"""
Собирает пакет для мобильщиков: один архив со всем, что нужно для модуля
точек и для переноса модуля отжиманий на Kotlin.

Что кладём в архив:
  README.md         оглавление пакета, версии и контрольные суммы
  docs/             памятка, контракт, выбор модели, модули и эксперименты
  pushup_logic.py   эталон модуля отжиманий на Python
  replay_csv.py     прогон эталона по CSV с телефона (нужен только Python 3)
  models/           файлы моделей MediaPipe full и lite, те же, что у нас
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
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from importlib import metadata
from pathlib import Path

import pushup_logic as logic
from extract_pose import MODEL_URL, get_model
from replay_csv import read_frames, read_labels

#Версия пакета совпадает с версией контракта данных
VERSION = "1.0"

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
NAME = f"pushup_mobile_v{VERSION}"

DOCS = ["mobile_handoff.md", "data_contract.md", "model_choice.md", "modules_experiments.md"]
CODE = ["pushup_logic.py", "replay_csv.py"]
MODELS = ["full", "lite"]

#Колонки эталона: сначала ключ кадра, потом то, что считает модуль отжиманий
GOLDEN_COLUMNS = ["frame", "time_ms", "signal", "smoothed", "top", "phase", "reps", "hint"]


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
        reps = counter.update(signal)
        hint = logic.frame_hint(points, width, height)
        rows.append([frame_no, time_ms, number(signal), number(counter.smoothed),
                     number(counter.top), counter.state.upper(), reps,
                     "" if hint is None else hint.upper()])
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


def readme(commit, dirty, models, samples, labels):
    """Текст README.md пакета."""
    built = date.today().isoformat()
    source = "нет git" if commit is None else commit + (", с незакоммиченными правками" if dirty else "")
    model_rows = "\n".join(f"| `models/{name}` | {size / 1e6:.1f} МБ | `{digest}` |"
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
3. [docs/model_choice.md](docs/model_choice.md): почему MediaPipe и как подключать его на iOS.
4. [docs/modules_experiments.md](docs/modules_experiments.md): зачем два модуля и как аналитик подбирает параметры. Для общего понимания.

## Что внутри

| Файл или папка | Зачем |
| --- | --- |
| `docs/` | документы, список выше |
| `pushup_logic.py` | эталон модуля отжиманий: сигнал, сглаживание, пороги, подсказки. Его логику переносите в `commonMain` |
| `replay_csv.py` | прогоняет эталон по CSV с телефона и предупреждает о повёрнутом или зеркальном кадре |
| `models/` | файлы моделей MediaPipe, на которых работаем мы. Кладите в приложение именно их |
| `samples/` | записи с веб-камеры в формате контракта: 66 колонок, строка на кадр |
| `golden/` | ответы эталона по каждому кадру записей из `samples/`, для автотестов Kotlin |
| `SHA256SUMS.txt` | контрольные суммы всех файлов пакета |

## Модели

Ссылка на скачивание в коде ведёт на `latest`: Google может обновить файл, и тогда телефон и компьютер разойдутся. Поэтому в пакете лежат сами файлы, а по контрольной сумме видно, что модель та же.

| Файл | Размер | SHA-256 |
| --- | --- | --- |
{model_rows}

Ссылка, откуда скачаны: `{MODEL_URL.format(m="<full или lite>")}`. На компьютере их запускает MediaPipe для Python версии {metadata.version("mediapipe")} в режиме LIVE\\_STREAM, `numPoses = 1`, остальные пороги по умолчанию.

## Эталон для автотестов Kotlin

Для каждой записи из `samples/` в `golden/` лежит файл с ответами Python-эталона по каждому кадру. Строки идут в том же порядке, что в записи: строка N эталона относится к строке N записи.

| Колонка | Что значит |
| --- | --- |
| `frame`, `time_ms` | номер и время кадра, как в записи: по ним удобно сверять строки |
| `signal` | сигнал кадра: насколько плечи выше запястий, в ширинах плеч (`shoulder_height`). Пусто, если плечи или запястья видны хуже порога |
| `smoothed` | сглаженный сигнал (медиана последних значений). Пусто, пока сигнала не было ни разу |
| `top` | «верх»: самое большое сглаженное значение в окне. Пусто, пока сигнала не было ни разу |
| `phase` | `UP` или `DOWN`, как `Phase` в контракте |
| `reps` | сколько повторов насчитано к этому кадру |
| `hint` | подсказка как `Hint` в контракте: `NO_PERSON`, `ARMS_HIDDEN`, `TOO_FAR`, `TOO_CLOSE`. Пусто, если с кадром всё хорошо |

| Запись | Эталон | Кадров | Повторов у эталона | Повторов на самом деле |
| --- | --- | --- | --- | --- |
{sample_rows}

Как пользоваться: автотест в `commonTest` читает запись из `samples/`, превращает каждую строку в `PoseFrame` (как в разделе «CSV для сверки телефона и компьютера» контракта), отдаёт кадры Kotlin-модулю по порядку и после каждого кадра сравнивает его ответ со строкой эталона. `phase`, `reps` и `hint` должны совпасть точно, `signal`, `smoothed` и `top` с точностью 0,0001. Первая строка, где ответ разошёлся, показывает, какой шаг перенесён не так.

Внутри модуля отжиманий считайте в `Double`: эталон считает в 64-битных дробях, и на кадрах у самого порога 32-битный `Float` может дать другую фазу.

Эталон описывает текущую, ещё не финальную версию счётчика: окна в нём пока в кадрах, а в финальной будут в миллисекундах. Когда аналитик закончит эксперименты, пакет пересоберём, и эталон обновится. Повторов у эталона на первой записи меньше, чем на самом деле: это известная ошибка базового счётчика, переносить нужно поведение эталона, а не правильный счёт.

## Сверка с телефона

Режим отладки в приложении пишет CSV в формате контракта и запоминает счёт Kotlin-модуля. Потом на компьютере:

```bash
python replay_csv.py <файл с телефона>.csv
```

Нужен только Python 3, ничего ставить не надо: `replay_csv.py` и `pushup_logic.py` используют одну стандартную библиотеку. Счёт должен совпасть со счётом телефона, а предупреждений про поворот и зеркало быть не должно.

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

    labels = read_labels()
    samples = []
    for path in sorted((ROOT / "samples").glob("*.csv")):
        shutil.copy2(path, folder / "samples" / path.name)
        rows = golden_rows(path)
        with open(folder / "golden" / f"{path.stem}.expected.csv", "w", newline="",
                  encoding="utf-8") as f:
            table = csv.writer(f, lineterminator="\n")
            table.writerow(GOLDEN_COLUMNS)
            table.writerows(rows)
        samples.append((path.stem, len(rows), rows[-1][6] if rows else 0))

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
