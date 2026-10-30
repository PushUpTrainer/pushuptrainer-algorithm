"""
Собирает пакет для мобильщиков: один архив со всем, что нужно для модуля
точек и для переноса модуля отжиманий на Kotlin.

Что кладём в архив:
  README.md         оглавление пакета, версии и контрольные суммы
  docs/             памятка, контракт, развёртывание модели, выбор модели, эксперименты
  pushup_logic.py   эталон модуля отжиманий на Python
  replay_csv.py     прогон эталона по CSV с телефона (нужен только Python 3)
  models/           модели full и lite: .task и извлечённые из них .tflite для LiteRT
  params.json       параметры счётчика и числа конвейера модели
  samples/          записи с камеры в формате контракта (66 колонок)
  golden/           ответы эталона по каждому кадру этих записей

Запуск из папки проекта:
  python make_mobile_package.py

Результат: output/pushup_mobile_v<версия>.zip и рядом та же папка без архива.
Пакет собирается из файлов на диске. Если в git есть незакоммиченные правки,
скрипт предупреждает: в пакет попадёт версия, которой нет в истории.
"""

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from importlib import metadata
from pathlib import Path

import pushup_logic as logic
from extract_pose import MODEL_URL, csv_header, csv_row, get_model
from replay_csv import read_frames, read_labels

#Версия пакета совпадает с версией контракта данных
VERSION = "1.1"

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
NAME = f"pushup_mobile_v{VERSION}"

DOCS = ["mobile_handoff.md", "data_contract.md", "model_deployment.md", "model_choice.md",
        "modules_experiments.md"]
CODE = ["pushup_logic.py", "replay_csv.py"]
MODELS = ["full", "lite"]

#Колонки эталона: сначала ключ кадра, потом то, что считает модуль отжиманий
GOLDEN_COLUMNS = ["frame", "time_ms", "signal", "smoothed", "top", "phase", "reps", "hint",
                  "grip", "errors", "wide_reps", "narrow_reps"]

#Параметры счётчика для params.json: имя константы в pushup_logic.py, единица,
#смысл и эксперимент, в котором её подбирают. Значения берутся из самого модуля.
COUNTER_PARAMS = [
    ("MIN_VIS", "доля от 0 до 1", "точка с visibility ниже порога ненадёжна", "E1"),
    ("SMOOTH_WINDOW", "значений сигнала", "медиана последних значений сигнала", "E2"),
    ("DOWN_LEVEL", "доля от «верха»", "сглаженный сигнал ниже: человек «внизу»", "E4"),
    ("UP_LEVEL", "доля от «верха»", "сглаженный сигнал выше: человек «вверху», повтор засчитан", "E4"),
    ("TOP_WINDOW", "кадров с сигналом", "окно поиска «верха»", "E5"),
    ("TOO_FAR_WIDTH", "доля ширины кадра", "плечи уже: подсказка TOO_FAR, хват не оценивается", "E7"),
    ("TOO_CLOSE_WIDTH", "доля ширины кадра", "плечи шире: подсказка TOO_CLOSE", "E7"),
    ("GRIP_NARROW", "ширин плеч", "хват уже: ошибка HANDS_TOO_NARROW", "E8"),
    ("GRIP_WIDE", "ширин плеч", "хват шире: ошибка HANDS_TOO_WIDE", "E8"),
    ("GRIP_WINDOW", "кадров с хватом", "окно медианы хвата для errors", "E8"),
    ("GRIP_MIN", "значений хвата", "меньше стольких за окно или повтор: ошибку хвата не определяем", "E8"),
]


def counter_params():
    """Параметры счётчика из pushup_logic.py. Падает, если какой-то параметр не описан."""
    described = {name for name, _, _, _ in COUNTER_PARAMS}
    numbers = {name for name, value in vars(logic).items()
               if name.isupper() and isinstance(value, (int, float)) and not isinstance(value, bool)
               and not name.startswith(("L_", "R_"))}
    if numbers - described:
        print("Параметры без описания в COUNTER_PARAMS:", sorted(numbers - described))
        sys.exit(1)
    return [{"name": name, "value": getattr(logic, name), "unit": unit, "meaning": meaning,
             "experiment": experiment} for name, unit, meaning, experiment in COUNTER_PARAMS]


#Числа конвейера MediaPipe Pose Landmarker 1.1.1 для пути 2 (LiteRT напрямую).
#Источник: графы и калькуляторы MediaPipe, подробно в docs/model_deployment.md.
POSE_PIPELINE = {
    "thresholds": {"num_poses": 1, "min_pose_detection_confidence": 0.5,
                   "min_pose_presence_confidence": 0.5, "min_tracking_confidence": 0.5},
    "detector": {
        "file": "pose_detector.tflite",
        "input": [1, 224, 224, 3],
        "input_color": "RGB",
        "input_normalization": "(p-127.5)/127.5",
        "resize": "letterbox: keep aspect ratio, black border, bilinear",
        "outputs": {"boxes_and_keypoints": [1, 2254, 12], "score_logits": [1, 2254, 1]},
        "anchors": {"num_layers": 5, "strides": [8, 16, 32, 32, 32], "min_scale": 0.1484375,
                    "max_scale": 0.75, "offset_x": 0.5, "offset_y": 0.5, "aspect_ratios": [1.0],
                    "fixed_anchor_size": True, "count": 2254},
        "box_scale": 224,
        "num_keypoints": 4,
        "score_clipping": 100,
        "nms": {"algorithm": "weighted", "overlap": "intersection_over_union", "threshold": 0.5},
        "association": {"iou_threshold": 0.5, "source": "min_tracking_confidence",
                        "rule": "later rect removes earlier overlapping rects, take first"},
    },
    "roi": {"center_keypoint": 0, "scale_keypoint": 1, "target_angle_degrees": 90,
            "square": "long_side", "scale": 1.25,
            "tracking_center_landmark": 33, "tracking_scale_landmark": 34},
    "landmarks": {
        "file": "pose_landmarks_detector.tflite",
        "input": [1, 256, 256, 3],
        "input_color": "RGB",
        "input_normalization": "p / 255",
        "border": "replicate",
        "outputs": {"landmarks": [1, 195], "presence": [1, 1], "segmentation": [1, 256, 256, 1],
                    "heatmap": [1, 64, 64, 39], "world_landmarks": [1, 117]},
        "landmarks_total": 39,
        "pose_landmarks": 33,
        "used_mediapipe_ids": list(logic.MEDIAPIPE_IDS),
        "coordinate_divisor": 256,
        "presence_rule": "person present only if presence > 0.5",
        "visibility_presence_activation": "sigmoid",
        "heatmap_refinement": {"size": 64, "kernel": 7, "min_confidence": 0.5},
    },
    "smoothing": {
        "landmarks_one_euro": {"min_cutoff": 0.05, "beta": 80.0, "derivate_cutoff": 1.0,
                               "initial_frequency": 30.0,
                               "value_scale": "1 / ((roi_next.width * W + roi_next.height * H) / 2)"},
        "visibility_ema_alpha": 0.1,
        "presence": "not smoothed",
        "reset": {"landmarks_one_euro": "person lost or frame size changed",
                  "visibility_ema": "never"},
    },
}


def params_json():
    """Содержимое params.json: всё, что нужно модулю отжиманий, кроме самой логики."""
    return {
        "contract_version": VERSION,
        "status": "черновик: значения подбирает аналитик (эксперименты с E1 по E8), "
                  "окна пока в кадрах, в финальной версии будут в миллисекундах",
        "counter": counter_params(),
        "hints_in_order": [code.upper() for code in logic.HINTS],
        "hint_texts": {code.upper(): text for code, text in logic.HINTS.items()},
        "error_texts": {code.upper(): text for code, text in logic.ERRORS.items()},
        "pose_pipeline": POSE_PIPELINE,
    }


#Искусственная запись для веток логики, которых нет в настоящих записях
SYNTHETIC = "synthetic_edge_cases"
WIDTH, HEIGHT = 640, 480


def synthetic_points(signal, shoulders=0.2, wrist_vis=0.95, grip=1.4):
    """
    12 точек человека лицом к камере: сигнал, ширина плеч в долях кадра
    и хват (расстояние между запястьями в ширинах плеч).
    """
    left_x, right_x = 0.5 + shoulders / 2, 0.5 - shoulders / 2  #левое плечо человека справа
    shoulder_y = 0.2
    wrist_y = shoulder_y + signal * shoulders * WIDTH / HEIGHT
    wrist_half = grip * shoulders / 2
    layout = [(left_x, shoulder_y), (right_x, shoulder_y),
              (left_x + 0.03, 0.35), (right_x - 0.03, 0.35),
              (0.5 + wrist_half, wrist_y), (0.5 - wrist_half, wrist_y),
              (0.55, 0.6), (0.45, 0.6), (0.55, 0.75), (0.45, 0.75), (0.55, 0.9), (0.45, 0.9)]
    points = [(x, y, 0.0, 0.95, 0.99) for x, y in layout]
    for i in (logic.L_WRIST, logic.R_WRIST):
        points[i] = points[i][:3] + (wrist_vis, 0.99)
    return points


def synthetic_record(path):
    """
    Пишет искусственную запись. По порядку: человека нет; слишком близко;
    скачок сигнала, после которого «верх» завышен и человек считается «внизу»
    без настоящего повтора (защита от ложного повтора должна его не засчитать);
    снова нет человека; руки не видны; слишком далеко; один настоящий повтор;
    повтор с руками слишком широко; повтор с руками слишком узко; человек уходит
    (ошибка хвата гаснет); повтор, где исход решает правило медианы (хват
    чередуется чуть ниже и чуть выше GRIP_NARROW, значений чётное число);
    повтор с плохим хватом, но меньше GRIP_MIN его оценок (не оценивается).
    Номера кадров идут с разрывами, как в режиме LIVE_STREAM.
    """
    def rep(grip):
        return ([synthetic_points(1.2, grip=grip)] * 20 + [synthetic_points(0.5, grip=grip)] * 20
                + [synthetic_points(1.2, grip=grip)] * 30)

    low, high = logic.GRIP_NARROW - 0.1, logic.GRIP_NARROW + 0.05
    median_rep = ([synthetic_points(1.2)] * 20
                  + [synthetic_points(0.5, grip=(low, high)[i % 2]) for i in range(19)]
                  + [synthetic_points(1.2, grip=(low, high)[(19 + i) % 2]) for i in range(30)])
    few_grips_rep = ([synthetic_points(1.2)] * 20
                     + [synthetic_points(0.5, grip=2.4)] * 4
                     + [synthetic_points(0.5, shoulders=0.1)] * 15
                     + [synthetic_points(1.2, shoulders=0.1)] * 30
                     + [synthetic_points(1.2)] * 20)

    parts = ([None] * 10
             + [synthetic_points(1.4, shoulders=0.8)] * 40
             + [synthetic_points(3.0)] * 10
             + [synthetic_points(1.2)] * 200
             + [None] * 10
             + [synthetic_points(1.2, wrist_vis=0.3)] * 10
             + [synthetic_points(1.2, shoulders=0.1)] * 10
             + rep(1.4)
             + rep(2.4)
             + rep(0.8)
             + [None] * 20
             + median_rep
             + few_grips_rep)
    with open(path, "w", newline="", encoding="utf-8") as f:
        table = csv.writer(f)
        table.writerow(csv_header())
        frame_no = 0
        for i, points in enumerate(parts):
            table.writerow(csv_row(SYNTHETIC, frame_no, 1000 + frame_no * 33, WIDTH, HEIGHT, points))
            frame_no += 2 if i % 7 == 6 else 1       #каждый седьмой кадр модель «пропустила»


def rep_grips(rows):
    """Хват по каждому засчитанному повтору, как его собирает счётчик: от кадра перехода
    «вниз» до кадра перехода «вверх» включительно. Список (значения хвата, ошибки повтора)."""
    reps, current = [], None
    for before, row in zip([None] + rows, rows):
        if row[5] == "DOWN" and (before is None or before[5] == "UP"):
            current = []
        if current is not None and row[8]:
            current.append(float(row[8]))
        if before is not None and row[5] == "UP" and row[6] != before[6]:
            gained = {"HANDS_TOO_WIDE": row[10] - before[10], "HANDS_TOO_NARROW": row[11] - before[11]}
            reps.append((current or [], {code for code, n in gained.items() if n}))
            current = None
    return reps


def tricky_cases(rows):
    """Какие ловушки для переноса есть в эталоне: множество из median, few, cleared."""
    found = set()
    for grips, errors in rep_grips(rows):
        if len(grips) >= logic.GRIP_MIN and len(grips) % 2 == 0:
            middle = sorted(grips)[len(grips) // 2 - 1: len(grips) // 2 + 1]
            if logic.grip_errors(sum(middle) / 2) != {code.lower() for code in errors}:
                found.add("median")
        if 0 < len(grips) < logic.GRIP_MIN and logic.grip_errors(logic.median(grips)) and not errors:
            found.add("few")
    for before, row in zip(rows, rows[1:]):
        if before[7] == row[7] == "NO_PERSON" and before[9] and not row[9]:
            found.add("cleared")
    return found


def coverage(rows):
    """Что проверяет эталон: подсказки, ошибки техники и подъёмы, которые не дали повтор."""
    hints = {row[7] for row in rows if row[7]}
    errors = {code for row in rows for code in row[9].split("|") if code}
    blocked = sum(1 for before, after in zip(rows, rows[1:])
                  if before[5] == "DOWN" and after[5] == "UP" and after[6] == before[6])
    return hints | errors, blocked


def number(value):
    """Дробь для эталона: 6 знаков после точки или пустая клетка."""
    return "" if value is None else f"{value:.6f}"


def golden_rows(path):
    """Ответы эталона по каждому кадру записи, в порядке строк CSV."""
    counter = logic.PushupCounter()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        frame_numbers = [int(row["frame"]) for row in csv.DictReader(f)]
    for frame_no, (time_ms, width, height, points) in zip(frame_numbers, read_frames(path)):
        signal = logic.shoulder_height(points, width, height)
        grip = logic.grip_ratio(points, width, height)
        reps = counter.update(signal, grip)
        hint = logic.frame_hint(points, width, height)
        rows.append([frame_no, time_ms, number(signal), number(counter.smoothed),
                     number(counter.top), counter.state.upper(), reps,
                     "" if hint is None else hint.upper(), number(grip),
                     "|".join(sorted(code.upper() for code in counter.errors)),
                     counter.error_counts.get("hands_too_wide", 0),
                     counter.error_counts.get("hands_too_narrow", 0)])
    return rows


def sha256(path):
    """Контрольная сумма файла: по ней видно, что файл тот же самый."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    """(коммит, есть ли незакоммиченные правки) или (None, False) без git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return commit, bool(dirty)


def megabytes(size):
    """Размер файла в мегабайтах через запятую, как в остальных документах."""
    return f"{size / 1e6:.1f}".replace(".", ",")


def readme(commit, dirty, models, samples, labels):
    """Текст README.md пакета."""
    built = date.today().isoformat()
    source = "нет git" if commit is None else commit + (", с незакоммиченными правками" if dirty else "")
    model_rows = "\n".join(f"| `models/{name}` | {megabytes(size)} МБ | `{digest}` |"
                           for name, size, digest in models)
    sample_rows = "\n".join(
        f"| `samples/{name}.csv` | `golden/{name}.expected.csv` | {frames} | {reps} | "
        f"{'' if labels.get(name) is None else labels[name]} |"
        for name, frames, reps in samples)
    return f"""# Пакет для мобильщиков, версия {VERSION}

Всё, что нужно, чтобы сделать модуль точек на Android и iOS и перенести модуль отжиманий в общий код Kotlin Multiplatform. Собрано {built} из репозитория ML-части (коммит {source}), контракт данных версии {VERSION}.

## С чего начать

1. [docs/mobile_handoff.md](docs/mobile_handoff.md): короткая памятка, что сделать в приложении.
2. [docs/data_contract.md](docs/data_contract.md): главный документ. `PoseFrame`, `PushupState`, код контракта на Kotlin и формат CSV.
3. [docs/model_deployment.md](docs/model_deployment.md): как запустить модель через LiteRT, путь 1 (MediaPipe Tasks) и путь 2 (LiteRT напрямую).
4. [docs/model_choice.md](docs/model_choice.md): почему MediaPipe.
5. [docs/modules_experiments.md](docs/modules_experiments.md): зачем два модуля и как аналитик подбирает параметры. Для общего понимания.

## Что внутри

| Файл или папка | Зачем |
| --- | --- |
| `docs/` | документы, список выше |
| `pushup_logic.py` | эталон модуля отжиманий: сигнал, сглаживание, пороги, подсказки, ошибки хвата. Его логику переносите в `commonMain` |
| `replay_csv.py` | прогоняет эталон по CSV с телефона и предупреждает о повёрнутом или зеркальном кадре |
| `models/` | файлы моделей, на которых работаем мы: `.task` для пути 1 (MediaPipe Tasks) и `litert/<вариант>/*.tflite` для пути 2 (LiteRT напрямую). Кладите в приложение именно их |
| `params.json` | параметры: счётчика (пороги, окна, тексты подсказок и ошибок) и конвейера модели для пути 2 |
| `samples/` | записи в формате контракта: 66 колонок, строка на кадр. Две настоящие записи с веб-камеры (альбомный кадр 640 на 480) и одна искусственная `{SYNTHETIC}.csv` для случаев, которых в настоящих нет |
| `golden/` | ответы эталона по каждому кадру записей из `samples/`, для автотестов Kotlin |
| `SHA256SUMS.txt` | контрольные суммы всех файлов пакета |

В документах встречаются `extract_pose.py`, `make_mobile_package.py`, `requirements.txt`, `data/labels.csv`, `data/experiments.csv` и `output/`: они есть только в репозитории ML-части, вам не нужны.

## Модели

Ссылка на скачивание в коде ведёт на `latest`: Google может обновить файл, и тогда телефон и компьютер разойдутся. Поэтому в пакете лежат сами файлы, а по контрольной сумме видно, что модель та же.

| Файл | Размер | SHA-256 |
| --- | --- | --- |
{model_rows}

Ссылка, откуда скачаны: `{MODEL_URL.format(m="<full или lite>")}`. На компьютере их запускает MediaPipe для Python версии {metadata.version("mediapipe")} в режиме LIVE\\_STREAM, `numPoses = 1`, остальные пороги по умолчанию. На Android и iOS берите выпуск библиотеки MediaPipe с тем же номером, если он есть, одинаковый на обеих платформах, и пишите номер выпуска в `.meta.json` режима отладки (ниже): разные выпуски могут давать немного разные точки.

## Эталон для автотестов Kotlin

Для каждой записи из `samples/` в `golden/` лежит файл с ответами Python-эталона по каждому кадру. Строки идут в том же порядке, что в записи: строка N эталона относится к строке N записи.

| Колонка | Что значит |
| --- | --- |
| `frame`, `time_ms` | номер и время кадра, как в записи: по ним удобно сверять строки |
| `signal` | сигнал кадра: насколько плечи выше запястий, в ширинах плеч (`shoulder_height`). Пусто, если плечи или запястья видны хуже порога |
| `smoothed` | сглаженный сигнал: медиана последних `SMOOTH_WINDOW` значений сигнала точно как в коде, `sorted(values)[len(values) // 2]`. Пока значений чётное число, это большее из двух средних, а не их среднее. Пусто, пока сигнала не было ни разу |
| `top` | «верх»: самое большое сглаженное значение за последние `TOP_WINDOW` кадров с сигналом (кадр без сигнала окно не сдвигает). Пусто, пока сигнала не было ни разу |
| `phase` | `UP` или `DOWN`, как `Phase` в контракте |
| `reps` | сколько повторов насчитано к этому кадру |
| `hint` | подсказка как `Hint` в контракте: `NO_PERSON`, `ARMS_HIDDEN`, `TOO_FAR`, `TOO_CLOSE`. Пусто, если с кадром всё хорошо |
| `grip` | хват кадра (`grip_ratio`): расстояние между запястьями, делённое на ширину плеч. Пусто, если плечи или запястья видны хуже порога или плечи уже порога «слишком далеко» |
| `errors` | ошибки техники сейчас, как `PushupState.errors`: `HANDS_TOO_WIDE`, `HANDS_TOO_NARROW` или пусто. Если ошибок несколько, они через вертикальную черту |
| `wide_reps`, `narrow_reps` | сколько засчитанных повторов к этому кадру сделано с этой ошибкой, как `WorkoutSummary.errorCounts` |

| Запись | Эталон | Кадров | Повторов у эталона | Повторов на самом деле |
| --- | --- | --- | --- | --- |
{sample_rows}

Как пользоваться: автотест в `commonTest` читает запись из `samples/`, превращает каждую строку в `PoseFrame` (числа читаются в `Float`, как в контракте; формат в разделе «CSV для сверки телефона и компьютера»), отдаёт кадры Kotlin-модулю по порядку и после каждого кадра сравнивает его ответ со строкой эталона:

- `phase`, `reps`, `hint`, `errors`, `wide_reps` и `narrow_reps` должны совпасть точно;
- `signal`, `smoothed`, `top` и `grip` с допуском: разница не больше 0,0001, умноженного на большее из двух чисел, 1 и модуля значения эталона. Допуск растёт с числом, потому что на кадрах с очень узкими плечами (человек боком к камере) сигнал доходит до 68, и округление `Float` при чтении точек даёт там разницу около 0,0004;
- пустая клетка эталона значит, что значения нет, и у Kotlin-модуля его тоже не должно быть.

Первая строка, где ответ разошёлся, показывает, какой шаг перенесён не так. Внутри модуля считайте в `Double` и переводите точки в `Double` до любых вычислений.

`signal`, `smoothed`, `top` и `grip` это внутренние значения модуля отжиманий: в `PushupState` их не добавляем, контракт не меняется. Для автотеста модуль отдаёт их через отдельное отладочное свойство, которое видят только тесты, а экран им не пользуется.

Искусственная запись `{SYNTHETIC}.csv` проверяет то, чего нет в настоящих: подсказку `TOO_CLOSE`, разрывы в номерах кадров (как в LIVE\\_STREAM) и защиту от ложного повтора. После резкого скачка сигнала «верх» завышен, и человек считается «внизу». Когда скачок уходит из окна, фаза снова `UP`, но повтор не засчитывается: внизу сигнал не опускался ниже порога от нового «верха». Если Kotlin-модуль засчитывает повтор на каждом переходе `DOWN` в `UP`, на этой записи он насчитает на один больше. Дальше пять повторов и уход из кадра. Повтор с классическим хватом (без ошибки), с руками слишком широко, с руками слишком узко. Уход из кадра: ошибка хвата гаснет через `GRIP_WINDOW` кадров без хвата. Повтор, где хват чередуется чуть ниже и чуть выше `GRIP_NARROW` и значений чётное число: по правилу медианы ошибки нет, а по среднему двух средних была бы. Повтор с широким хватом, но меньше `GRIP_MIN` его оценок (дальше человек слишком далеко): он не оценивается.

Хват считается так же, как сигнал: медиана `sorted(values)[len(values) // 2]`. Для `errors` берутся последние `GRIP_WINDOW` значений хвата; после `GRIP_WINDOW` кадров подряд без хвата окно очищается. Для ошибки повтора берутся значения от кадра перехода в `DOWN` до кадра перехода в `UP` включительно, и повтор оценивается, только если значений не меньше `GRIP_MIN`.

Пороги и окна счётчика лежат в `params.json` в разделе `counter`, порядок и тексты подсказок в `hints_in_order` и `hint_texts`, тексты ошибок в `error_texts`: берите значения оттуда, а не из памяти. Числа в нём взяты из `pushup_logic.py` при сборке пакета.

Эталон описывает текущую, ещё не финальную версию счётчика: окна в нём пока в кадрах, а в финальной будут в миллисекундах. Когда аналитик закончит эксперименты, пакет пересоберём, и эталон обновится. Повторов у эталона на первой записи меньше, чем на самом деле: это известная ошибка базового счётчика, переносить нужно поведение эталона, а не правильный счёт.

## Сверка с телефона

Режим отладки в приложении пишет CSV в формате контракта и запоминает счёт Kotlin-модуля. Потом на компьютере:

```bash
python replay_csv.py <файл с телефона>.csv
```

На Mac команда обычно `python3`, а не `python`. Нужен только Python 3, ничего ставить не надо: `replay_csv.py` и `pushup_logic.py` используют одну стандартную библиотеку. Можно передать и папку: файлы эталона из `golden/` скрипт пропустит, это не записи.

Счёт должен совпасть со счётом телефона, а предупреждений про поворот и зеркало быть не должно. Удобно, если режим отладки рядом с `<video>.csv` пишет `<video>.meta.json`, например `{{"video": "phone_001", "model": "full", "mediapipe": "<номер выпуска>", "reps_counted": 7, "error_counts": {{"HANDS_TOO_WIDE": 1}}}}`. Тогда `replay_csv.py` сам покажет счёт телефона в колонке «вживую» рядом со своим в колонке «повторы» и напишет, если `error_counts` не совпали с его прогоном. Совпасть должны и повторы, и ошибки. Колонка «факт» нужна только аналитику, у вас она будет пустой.

## К кому с вопросами

- По точкам, CSV, модулю точек и этому пакету: к ML-инженеру.
- По логике счёта и параметрам: к аналитику.
"""


def check_links(folder):
    """Относительные ссылки во всех .md пакета ведут на файлы внутри пакета."""
    broken = []
    for md in folder.rglob("*.md"):
        for link in re.findall(r"\]\(([^)#\s]+)", md.read_text(encoding="utf-8")):
            if not link.startswith("http") and not (md.parent / link).exists():
                broken.append(f"{md.relative_to(folder)}: {link}")
    return broken


def main():
    commit, dirty = git_state()
    if dirty:
        print("Внимание: в git есть незакоммиченные правки, в пакет попадёт версия с ними")

    folder = OUT_DIR / NAME
    if folder.exists():
        shutil.rmtree(folder)
    for sub in ("docs", "models", "samples", "golden"):
        (folder / sub).mkdir(parents=True)

    for name in DOCS:
        shutil.copy2(ROOT / "docs" / name, folder / "docs" / name)
    for name in CODE:
        shutil.copy2(ROOT / name, folder / name)

    models = []
    for model in MODELS:
        path = Path(get_model(model))                #скачает модель, если её ещё нет
        shutil.copy2(path, folder / "models" / path.name)
        models.append((path.name, path.stat().st_size, sha256(path)))
        #Для пути 2 (LiteRT напрямую) нужны сами .tflite: .task это zip из двух моделей
        litert = folder / "models" / "litert" / model
        litert.mkdir(parents=True)
        with zipfile.ZipFile(path) as task:
            for inner in sorted(task.namelist()):
                target = litert / inner
                target.write_bytes(task.read(inner))
                models.append((f"litert/{model}/{inner}", target.stat().st_size, sha256(target)))

    labels = read_labels()
    labels[SYNTHETIC] = "5, так задумано"
    for path in sorted((ROOT / "samples").glob("*.csv")):
        shutil.copy2(path, folder / "samples" / path.name)
    synthetic_record(folder / "samples" / f"{SYNTHETIC}.csv")

    samples, hints, blocked, tricky = [], set(), 0, set()
    for path in sorted((folder / "samples").glob("*.csv")):
        rows = golden_rows(path)
        with open(folder / "golden" / f"{path.stem}.expected.csv", "w", newline="",
                  encoding="utf-8") as f:
            table = csv.writer(f, lineterminator="\n")
            table.writerow(GOLDEN_COLUMNS)
            table.writerows(rows)
        samples.append((path.stem, len(rows), rows[-1][6] if rows else 0))
        found, stopped = coverage(rows)
        hints |= found
        blocked += stopped
        tricky |= tricky_cases(rows)

    #Эталон должен проверять все подсказки, обе ошибки хвата и защиту от ложного повтора
    missing = {"NO_PERSON", "ARMS_HIDDEN", "TOO_FAR", "TOO_CLOSE",
               "HANDS_TOO_WIDE", "HANDS_TOO_NARROW"} - hints
    synthetic_reps = {name: reps for name, _, reps in samples}[SYNTHETIC]
    no_tricks = {"median", "few", "cleared"} - tricky
    if missing or not blocked or synthetic_reps != 5 or no_tricks:
        print(f"Эталон неполный: нет кодов {sorted(missing)}, подъёмов без повтора {blocked}, "
              f"повторов в искусственной записи {synthetic_reps} вместо 5, "
              f"нет ловушек {sorted(no_tricks)}")
        sys.exit(1)

    (folder / "params.json").write_text(
        json.dumps(params_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (folder / "README.md").write_text(readme(commit, dirty, models, samples, labels),
                                      encoding="utf-8", newline="\n")

    broken = check_links(folder)
    if broken:
        print("Ссылки в пакете ведут на несуществующие файлы:", *broken, sep="\n  ")
        sys.exit(1)

    files = sorted(p for p in folder.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p)}  {p.relative_to(folder).as_posix()}\n" for p in files)
    (folder / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")

    archive = OUT_DIR / f"{NAME}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            z.write(path, f"{NAME}/{path.relative_to(folder).as_posix()}")

    print(f"Готово: {archive} ({archive.stat().st_size / 1e6:.1f} МБ)")
    for name, frames, reps in samples:
        print(f"  {name}: {frames} кадров, эталон насчитал {reps}")


if __name__ == "__main__":
    main()
