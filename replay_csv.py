"""
Прогоняет счётчик из pushup_logic.py по готовым CSV, без камеры и без модели.

Зачем нужен:
  эксперименты: меняешь формулу или параметры в pushup_logic.py и сразу видишь счёт;
  проверка правок: счёт по CSV должен остаться тем же, что был до правки;
  сверка с телефоном: на CSV из режима отладки приложения счёт должен совпасть
  со счётом телефона, а проверка поворота и зеркала не должна ругаться.

Запуск из папки проекта:
  python replay_csv.py                                  #все CSV из output/csv/
  python replay_csv.py output/csv/full                  #одна папка
  python replay_csv.py output/csv/full/webcam_1.csv     #один файл

Рядом с CSV ищется .meta.json: если он есть, в таблице видно, сколько
повторов насчитал счётчик вживую при записи. Если в data/labels.csv есть
строка для записи, видно и сколько повторов было на самом деле, а внизу
печатается ошибка счёта и доля точных записей.

Поворот и зеркало: у человека лицом к камере в правильном кадре линия плеч
ближе к горизонтали, чем к вертикали, плечи выше запястий, а левое плечо
лежит правее правого (x больше). Если больше половины кадров нарушают одно
из этих правил, скрипт предупреждает: кадр не повёрнут, перевёрнут на 180
градусов или зеркальный.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from pushup_logic import (ERRORS, L_SHOULDER, MIN_VIS, POINT_NAMES, R_SHOULDER, PushupCounter,
                          frame_hint, grip_ratio, median, shoulder_height)

FIELDS = ("x", "y", "z", "vis", "pres")

#Разметка записей: сколько повторов было на самом деле
LABELS = Path(__file__).resolve().parent / "data" / "labels.csv"

#Доля подозрительных кадров, после которой предупреждаем про поворот или зеркало
WARN_SHARE = 0.5


def read_frames(path):
    """Кадры из CSV по порядку: (time_ms, width, height, points или None)."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            width, height = int(row["width"]), int(row["height"])
            points = None
            if row["detected"] == "1":
                points = [tuple(float(row[f"{name}_{field}"]) for field in FIELDS)
                          for name in POINT_NAMES]
            yield int(row["time_ms"]), width, height, points


def shoulders_check(points, width, height):
    """(повёрнуто?, зеркально?) по линии плеч или None, если плечи плохо видны."""
    left, right = points[L_SHOULDER], points[R_SHOULDER]
    if left[3] < MIN_VIS or right[3] < MIN_VIS:
        return None
    dx = (left[0] - right[0]) * width
    dy = (left[1] - right[1]) * height
    rotated = abs(dy) > abs(dx)
    #Если линия плеч вертикальная, по знаку dx про зеркало ничего не понять
    return rotated, None if rotated else dx < 0


def share(count, total):
    """Доля от 0 до 1, без деления на ноль."""
    return count / total if total else 0


def replay(path):
    """Итоги одного CSV: кадры, доля кадров с человеком, длительность, повторы."""
    counter = PushupCounter()
    frames = detected = hints = 0
    shoulders = rotated = horizontal = mirrored = 0
    signals = upside_down = 0
    grips, error_frames = [], 0
    first_ms = last_ms = 0
    for time_ms, width, height, points in read_frames(path):
        if frames == 0:
            first_ms = time_ms
        last_ms = time_ms
        frames += 1
        detected += points is not None
        hints += frame_hint(points, width, height) is not None
        check = points and shoulders_check(points, width, height)
        if check:
            shoulders += 1
            rotated += check[0]
            if check[1] is not None:
                horizontal += 1
                mirrored += check[1]
        signal = shoulder_height(points, width, height)
        if signal is not None:
            signals += 1
            upside_down += signal < 0                #плечи ниже запястий
        grip = grip_ratio(points, width, height)
        if grip is not None:
            grips.append(grip)
        counter.update(signal, grip)
        error_frames += points is not None and bool(counter.errors)
    return {"frames": frames,
            "detected": share(detected, frames),
            "hints": share(hints, frames),
            "seconds": (last_ms - first_ms) / 1000,
            "reps": counter.reps,
            "grip": median(grips) if grips else None,
            "error_counts": counter.error_counts,
            "judged": counter.judged_reps,
            "error_frames": share(error_frames, detected),
            "rotated": share(rotated, shoulders),
            "upside_down": share(upside_down, signals),
            "mirrored": share(mirrored, horizontal)}


def read_meta(path):
    """Содержимое .meta.json рядом с CSV или пустой словарь."""
    meta = path.with_suffix(".meta.json")
    if not meta.exists():
        return {}
    return json.loads(meta.read_text(encoding="utf-8"))


def live_reps(path):
    """Число повторов из .meta.json рядом с CSV или None."""
    return read_meta(path).get("reps_counted")


def read_labels():
    """Разметка: имя записи и настоящее число повторов. Пусто, если файла нет."""
    if not LABELS.exists():
        return {}
    #utf-8-sig: Excel и Блокнот иногда пишут в начало файла служебные байты (BOM)
    try:
        with open(LABELS, newline="", encoding="utf-8-sig") as f:
            table = csv.DictReader(f)
            columns = set(table.fieldnames or [])
            rows = list(table)
    except UnicodeDecodeError:
        print("Внимание: data/labels.csv не в кодировке UTF-8. Сохрани его в UTF-8 "
              "(PyCharm или Блокнот), разделитель запятая")
        return {}
    if not {"video", "reps"} <= columns:
        print("Внимание: в data/labels.csv нет колонок video и reps. Первая строка "
              "должна быть video,reps,view,light,person,grip, разделитель запятая")
        return {}

    labels = {}
    for row in rows:
        video, reps = (row["video"] or "").strip(), (row["reps"] or "").strip()
        if not video or not reps:                    #счёт ещё не вписали
            continue
        try:
            value = float(reps.replace(",", "."))
        except ValueError:
            value = None
        if value is None or not value.is_integer():
            print(f"Внимание: в data/labels.csv у записи {video} в reps не целое число: {reps}")
            continue
        labels[video] = int(value)
    return labels


#Какую ошибку хвата ждём по разметке grip в labels.csv
GRIP_LABELS = {"classic": None, "wide": "hands_too_wide", "narrow": "hands_too_narrow"}


def read_grip_labels():
    """Разметка хвата: имя записи и classic, wide или narrow. Пусто, если колонки нет."""
    if not LABELS.exists():
        return {}
    try:
        with open(LABELS, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        return {}
    return {(row.get("video") or "").strip(): (row.get("grip") or "").strip()
            for row in rows if (row.get("grip") or "").strip() in GRIP_LABELS}


def grip_correct(grip_label, judged, error_counts):
    """Сколько оценённых по хвату повторов получили ту оценку, что в разметке."""
    expected = GRIP_LABELS[grip_label]
    if expected is None:
        return judged - sum(error_counts.values())
    return error_counts.get(expected, 0)


def csv_files(target):
    """Все CSV по пути: один файл или все файлы в папке и вложенных папках."""
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.csv"))


def is_record(path):
    """True, если CSV похож на запись с точками: в шапке есть служебные колонки."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    return {"video", "frame", "time_ms", "width", "height", "detected"} <= set(header)


def main():
    parser = argparse.ArgumentParser(description="Счётчик отжиманий по готовым CSV")
    parser.add_argument("path", nargs="?", default="output/csv",
                        help="CSV или папка с CSV (по умолчанию output/csv)")
    args = parser.parse_args()

    files = csv_files(Path(args.path))
    if not files:
        print("CSV не найдены:", args.path)
        print("Записи появляются после запуска python extract_pose.py. "
              "Без камеры можно взять готовые: python replay_csv.py samples")
        sys.exit(1)
    #Например файлы эталона из пакета для мобильщиков: в них нет точек
    skipped = [path for path in files if not is_record(path)]
    files = [path for path in files if path not in skipped]
    for path in skipped:
        print(f"Пропускаю {path.name}: это не запись с точками")
    if not files:
        sys.exit(1)

    labels = read_labels()                           #до таблицы, чтобы предупреждения не рвали её
    grip_labels = read_grip_labels()
    #Метрика E8 по каждому хвату: верно, оценено, не оценено; доля кадров с ошибкой на экране
    grip_stats = {label: [0, 0, 0] for label in GRIP_LABELS}
    classic_error_frames = []
    print(f"{'файл':<40} {'кадров':>7} {'секунд':>7} {'человек':>8} "
          f"{'подсказки':>10} {'повторы':>8} {'вживую':>7} {'факт':>5} {'хват':>5} "
          f"{'с ошибкой':>9}")
    warnings, errors, technique = [], [], []
    for path in files:
        result = replay(path)
        live = live_reps(path)
        truth = labels.get(path.stem)
        if truth is not None:
            errors.append(abs(result["reps"] - truth))
        print(f"{path.name:<40} {result['frames']:>7} {result['seconds']:>7.1f} "
              f"{result['detected']:>8.0%} {result['hints']:>10.0%} "
              f"{result['reps']:>8} {'' if live is None else live:>7} "
              f"{'' if truth is None else truth:>5} "
              f"{'' if result['grip'] is None else format(result['grip'], '.2f'):>5} "
              f"{sum(result['error_counts'].values()):>9}")
        label = grip_labels.get(path.stem)
        if label:
            stats = grip_stats[label]
            stats[0] += grip_correct(label, result["judged"], result["error_counts"])
            stats[1] += result["judged"]
            stats[2] += result["reps"] - result["judged"]
            if label == "classic":
                classic_error_frames.append(result["error_frames"])
        live_errors = read_meta(path).get("error_counts")
        if live_errors is not None:
            live = {code.lower(): count for code, count in live_errors.items() if count}
            if live != result["error_counts"]:
                technique.append(f"{path.name}: ошибки вживую из .meta.json {live} "
                                 f"не совпадают с прогоном {result['error_counts']}")
        for code, count in sorted(result["error_counts"].items()):
            technique.append(f"{path.name}: {ERRORS[code].lower()} в {count} повт.")
        #Проверки по порядку: при неповёрнутом или перевёрнутом кадре
        #проверка зеркала даёт случайный ответ, поэтому её не печатаем
        if result["rotated"] > WARN_SHARE:
            warnings.append(f"{path.name}: похоже, кадр не повёрнут вертикально "
                            f"({result['rotated']:.0%} кадров с вертикальной линией плеч)")
        elif result["upside_down"] > WARN_SHARE:
            warnings.append(f"{path.name}: похоже, кадр перевёрнут на 180 градусов "
                            f"({result['upside_down']:.0%} кадров с плечами ниже запястий)")
        elif result["mirrored"] > WARN_SHARE:
            warnings.append(f"{path.name}: похоже, кадр зеркальный "
                            f"({result['mirrored']:.0%} кадров с левым плечом слева)")
    if errors:
        exact = sum(error == 0 for error in errors) / len(errors)
        print(f"Размечено записей: {len(errors)}. Ошибка счёта: "
              f"{sum(errors) / len(errors):.2f} повтора на запись, точных записей: {exact:.0%}")
    for label, (ok, judged, unjudged) in grip_stats.items():
        if judged or unjudged:
            print(f"Хват {label}: верно оценён в {ok} из {judged} повторов"
                  + (f" ({ok / judged:.0%})" if judged else "")
                  + (f", не оценено (мало кадров с хватом): {unjudged}" if unjudged else ""))
    if classic_error_frames:
        print(f"Ложные ошибки на экране при классическом хвате: "
              f"{sum(classic_error_frames) / len(classic_error_frames):.1%} кадров с человеком")
    for line in technique:
        print("Ошибка техники:", line)
    for warning in warnings:
        print("Внимание:", warning)


if __name__ == "__main__":
    main()
