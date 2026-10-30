# Как читать CSV с точками

Короткая памятка для аналитика: где лежат записи, что в них за колонки и как достать данные в Python. Все примеры проверены на файлах из `samples/`. Полный формат описан в документе [Контракт данных](data_contract.md).

## Где лежат файлы

- Свои записи: `output/csv/<модель>/webcam_<дата>_<время>.csv`, например `output/csv/full/webcam_20261021_194714.csv`. Рядом лежит `.meta.json` с итогами записи.
- Готовые записи из репозитория: `samples/`.
- Разметка (сколько отжиманий было на самом деле): `data/labels.csv`.

Один файл это одна запись с камеры. Файлы никогда не правим: вся очистка точек делается в коде.

## Что в одной строке

Одна строка это один кадр, который обработала модель. В файле 66 колонок.

**6 служебных колонок:**

| Колонка | Что значит |
| --- | --- |
| `video` | имя записи, одинаковое во всех строках |
| `frame` | номер кадра камеры. Кадры, которые модель пропустила, в файл не попали, поэтому номера могут идти с разрывами |
| `time_ms` | время кадра в миллисекундах от запуска, строго растёт. Это главная ось времени |
| `width`, `height` | размер кадра в пикселях |
| `detected` | 1, если человек найден, и 0, если нет |

**60 колонок точек:** 12 точек по 5 чисел, имя колонки `<точка>_<число>`.

- Точки по порядку: `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`. «Левое» значит левое у самого человека: человек смотрит в камеру, поэтому его левое плечо на картинке справа (x больше).
- `_x` и `_y`: положение в кадре в долях ширины и высоты. Начало в левом верхнем углу, y растёт вниз. В пиксели: `x * width`, `y * height`. Обычно числа от 0 до 1, но если точка за краем кадра, будет меньше 0 или больше 1 (часто так у бёдер, когда они ниже кадра). Это нормально: такие значения не обрезай, смотри на `_vis`.
- `_z`: глубина, пока не используем.
- `_vis` и `_pres` от 0 до 1: насколько точка видна и насколько она вообще в кадре. Точку с `_vis` меньше 0,5 (`MIN_VIS`) счётчик считает ненадёжной.
- Если `detected = 0`, все 60 клеток пустые (в pandas это `NaN`).
- Очень малые числа могут быть записаны как `3e-05`: это обычная дробь, Python читает её сам.

Пример начала файла (первые колонки):

```text
video,frame,time_ms,width,height,detected,left_shoulder_x,left_shoulder_y,...
webcam_20261017_115256,0,896,640,480,0,,,...
```

## Где запускать свой код

Свой скрипт клади прямо в папку проекта (рядом с README) и запускай из неё. В терминале PyCharm это `python my_e4.py`, в обычном PowerShell `.venv\Scripts\python.exe my_e4.py` (подробно в README, раздел «Быстрый старт»). Для быстрых проб удобна консоль PyCharm: Tools, Python or Debug Console. Она сразу открывается в папке проекта.

Пути в примерах ниже считаются от папки проекта.

## Таблица: pandas

Одна запись:

```python
import pandas as pd

df = pd.read_csv("samples/webcam_20261017_115256.csv")
print(df.shape)  #(1159, 66): 1159 кадров, 66 колонок

people = df[df["detected"] == 1]  #только кадры, где человек найден
print(len(people), "кадров с человеком из", len(df))

#время в секундах от начала записи
df["t"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000

#координата в пикселях
df["left_wrist_px"] = df["left_wrist_x"] * df["width"]

#средняя видимость каждой точки по кадрам с человеком
print(people.filter(like="_vis").mean().round(2))

#сколько кадров модель пропустила между соседними строками (0 значит без пропуска)
print((df["frame"].diff() - 1).value_counts())
```

В `samples/` пропусков нет: эти записи сделаны прошлой версией, где модель обрабатывала каждый кадр. В новых записях пропуски бывают.

Сравнение с пустой клеткой всегда даёт «нет»: `df["left_wrist_vis"] < 0.5` для кадра без человека равно False. Поэтому доли по видимости считай по `people`, а не по всей таблице.

Много записей в одной таблице вместе с разметкой:

```python
from pathlib import Path

import pandas as pd

files = sorted(Path("samples").rglob("*.csv"))  #все записи из папки и вложенных папок
data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
labels = pd.read_csv("data/labels.csv")
data = data.merge(labels, on="video", how="left")  #к каждому кадру разметка его записи
print(data.groupby("video").agg(person=("person", "first"), reps=("reps", "first"),
                                share=("detected", "mean")))
```

## Сигнал счётчика для всех кадров

Счётчик смотрит на один сигнал: насколько плечи выше запястий, в ширинах плеч (подробно в README, раздел «Как работает базовый счётчик»). Вот он же сразу для всей таблицы. Результат совпадает с `shoulder_height` из `pushup_logic.py`, а там, где у кадра нет сигнала, стоит `NaN`.

```python
import numpy as np
import pandas as pd

import pushup_logic as logic

df = pd.read_csv("samples/webcam_20261017_115256.csv")

#плечи и запястья видны не хуже порога
seen = (df[["left_shoulder_vis", "right_shoulder_vis",
            "left_wrist_vis", "right_wrist_vis"]] >= logic.MIN_VIS).all(axis=1)

dx = (df["left_shoulder_x"] - df["right_shoulder_x"]) * df["width"]
dy = (df["left_shoulder_y"] - df["right_shoulder_y"]) * df["height"]
shoulder_px = np.hypot(dx, dy)  #ширина плеч в пикселях

shoulders_y = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2 * df["height"]
wrists_y = (df["left_wrist_y"] + df["right_wrist_y"]) / 2 * df["height"]
df["signal"] = ((wrists_y - shoulders_y) / shoulder_px).where(seen & (shoulder_px >= 1))
print(df["signal"].describe())
```

На этой записи сигнал обычно от 1 до 1,5, но в 16 кадрах он больше 3, а самый большой 68. В этих кадрах человек повернулся боком, плечи на картинке почти слились, и делить приходится на маленькую ширину плеч. Такие кадры как раз учатся отбрасывать в экспериментах E1 и E3.

## Как счётчик видит запись

Чтобы понять, почему повтор засчитан или нет, нарисуй сглаженный сигнал и оба порога. Их знает только сам счётчик, поэтому прогоняем его по кадрам и после каждого кадра запоминаем `counter.smoothed` и `counter.top`. Функция `read_frames` из `replay_csv.py` отдаёт кадры по порядку в виде `(time_ms, width, height, points)`: `points` это список из 12 точек в порядке выше, каждая точка `(x, y, z, vis, pres)`, или `None`, если человека нет.

```python
import matplotlib.pyplot as plt

import pushup_logic as logic
from replay_csv import read_frames

frames = list(read_frames("samples/webcam_20261017_115256.csv"))
start = frames[0][0]
counter = logic.PushupCounter()
rows = []
for time_ms, width, height, points in frames:
    counter.update(logic.shoulder_height(points, width, height))
    if counter.top is not None:  #пока сигнала не было, рисовать нечего
        rows.append(((time_ms - start) / 1000, counter.smoothed,
                     logic.DOWN_LEVEL * counter.top, logic.UP_LEVEL * counter.top))

t, smoothed, down, up = zip(*rows)
plt.plot(t, smoothed, label="сглаженный сигнал")
plt.plot(t, down, "--", label="порог «внизу»")
plt.plot(t, up, "--", label="порог «вверху»")
plt.ylim(0, 3)  #без этого выбросы в конце записи сплющивают график, и повторов не видно
plt.xlabel("секунды")
plt.legend()
plt.show()
```

На графике 8 глубоких провалов: это 8 повторов. Повтор засчитывается, когда сигнал ушёл ниже порога «внизу» и потом поднялся выше порога «вверху». Провал около 25,7 секунды счётчик не засчитал: сглаженный сигнал опустился только до 0,82, а порог «внизу» в тот момент был 0,79 (`DOWN_LEVEL` 0,55 от «верха» 1,43). Мелкий провал около 28 секунды это недоделанный повтор, его счётчик правильно пропустил. После 31 секунды выбросы поднимают «верх», и оба порога улетают вверх: один такой скачок может на несколько секунд «ослепить» счётчик (это задача E1 и E5). Ровная линия в самом конце значит, что сигнала нет и счётчик держит последнее значение.

## Прогон счётчика со своими параметрами и своей формулой

```python
import pushup_logic as logic
from replay_csv import read_frames

frames = list(read_frames("samples/webcam_20261017_115256.csv"))


def count(frames, signal=logic.shoulder_height):
    """Прогоняет счётчик по кадрам одной записи и возвращает число повторов."""
    counter = logic.PushupCounter()
    for time_ms, width, height, points in frames:
        counter.update(signal(points, width, height))
    return counter.reps


print(count(frames))  #7, а в разметке 8: один повтор счётчик пропустил

logic.DOWN_LEVEL = 0.45  #пробуем другой порог «внизу»
print(count(frames))  #0: при таком пороге счётчик не замечает ни одного повтора
logic.DOWN_LEVEL = 0.55  #возвращаем базовое значение


def by_arm(points, width, height):
    """Вариант 2 из E3: высота плеч над запястьями, делённая на длину руки."""
    if points is None or not logic.visible(points, logic.ARM_POINTS):
        return None
    xy = [logic.px(points, i, width, height) for i in range(len(points))]
    arm = (logic.distance(xy[logic.L_SHOULDER], xy[logic.L_ELBOW])
           + logic.distance(xy[logic.L_ELBOW], xy[logic.L_WRIST])
           + logic.distance(xy[logic.R_SHOULDER], xy[logic.R_ELBOW])
           + logic.distance(xy[logic.R_ELBOW], xy[logic.R_WRIST])) / 2
    if arm < 1:
        return None
    shoulders_y = (xy[logic.L_SHOULDER][1] + xy[logic.R_SHOULDER][1]) / 2
    wrists_y = (xy[logic.L_WRIST][1] + xy[logic.R_WRIST][1]) / 2
    return (wrists_y - shoulders_y) / arm


print(count(frames, by_arm))  #0: этот сигнал проседает слабее, до порога 0,55 не доходит
logic.DOWN_LEVEL, logic.UP_LEVEL = 0.85, 0.95  #для него нужны свои пороги
print(count(frames, by_arm))  #8, как в разметке
```

Параметры из `pushup_logic.py` можно так менять прямо из своего скрипта, не трогая сам файл. `MIN_VIS`, `DOWN_LEVEL`, `UP_LEVEL`, `GRIP_NARROW`, `GRIP_WIDE` и `GRIP_MIN` действуют сразу. `SMOOTH_WINDOW`, `TOP_WINDOW` и `GRIP_WINDOW` счётчик берёт при создании, поэтому после их смены нужен новый `PushupCounter()` (функция `count` создаёт его сама). Изменение живёт, пока работает скрипт или консоль.

Отдельная точка из `points`, например левое запястье: `points[logic.L_WRIST]`, его видимость `points[logic.L_WRIST][3]`.

## Перебор параметра по всем размеченным записям

Так выглядит эксперимент E4 в миниатюре: для каждого значения порога считаем метрики из журнала `data/experiments.csv`.

```python
from pathlib import Path

import pushup_logic as logic
from replay_csv import read_frames, read_labels

labels = read_labels()  #{"webcam_20261017_115256": 8, ...} из data/labels.csv
folder = Path("samples")  #сюда папку с настроечными записями
records = {path.stem: list(read_frames(path))
           for path in folder.rglob("*.csv") if path.stem in labels}
print(len(records), "размеченных записей в", folder)
if not records:
    raise SystemExit("Нет размеченных записей: проверь папку и data/labels.csv")


def count(frames, signal=logic.shoulder_height):
    counter = logic.PushupCounter()
    for time_ms, width, height, points in frames:
        counter.update(signal(points, width, height))
    return counter.reps


for level in [0.45, 0.50, 0.55, 0.60, 0.65]:
    logic.DOWN_LEVEL = level
    diffs = [count(frames) - labels[name] for name, frames in records.items()]
    error = sum(abs(d) for d in diffs) / len(diffs)  #ошибка счёта
    exact = sum(d == 0 for d in diffs) / len(diffs)  #доля точных записей
    extra = sum(d for d in diffs if d > 0)  #лишние повторы
    missed = -sum(d for d in diffs if d < 0)  #пропущенные повторы
    print(f"{level:.2f}  ошибка {error:.2f}  точные {exact:.0%}  лишние {extra}  пропущенные {missed}")
logic.DOWN_LEVEL = 0.55
```

На двух записях из `samples/` получается так:

```text
2 размеченных записей в samples
0.45  ошибка 4.00  точные 50%  лишние 0  пропущенные 8
0.50  ошибка 2.00  точные 50%  лишние 0  пропущенные 4
0.55  ошибка 0.50  точные 50%  лишние 0  пропущенные 1
0.60  ошибка 0.00  точные 100%  лишние 0  пропущенные 0
0.65  ошибка 0.00  точные 100%  лишние 0  пропущенные 0
```

Порог 0,6 здесь лучше нынешнего 0,55, но по двум записям одного человека выводов не делаем: нужны настроечные записи разных людей.

Файлы читаются один раз, дальше перебор идёт по кадрам в памяти и занимает секунды. Каждый запуск записывай строкой в `data/experiments.csv`. В папку `folder` контрольные записи не клади: их проверяют один раз в конце (подробно в документе [Модули, данные и эксперименты](modules_experiments.md)).

## Итоги записи: .meta.json

```python
import json
from pathlib import Path

meta = json.loads(Path("samples/webcam_20261017_115256.meta.json").read_text(encoding="utf-8"))
print(meta["reps_counted"])  #сколько повторов счётчик насчитал вживую
```

В новых записях есть поле `mode` (`LIVE_STREAM`), а ещё `frames` (сколько кадров дала камера), `processed` (сколько обработала модель, столько строк в CSV), `camera_fps`, `processing_fps` и `error_counts` (сколько засчитанных повторов сделано с каждой ошибкой хвата). Если поля `mode` нет, запись старая, как в `samples/`: там `frames` значит число строк CSV, а надёжно только `reps_counted`.

## Частые ошибки

- `ModuleNotFoundError: No module named 'pushup_logic'`: скрипт лежит не в папке проекта (например в подпапке).
- `FileNotFoundError`: скрипт запущен из другой папки, а пути считаются от текущей. Запускай из папки проекта или пиши полный путь.
- `No module named 'pandas'`: скрипт запущен не тем Python (в обычном PowerShell пиши `.venv\Scripts\python.exe`) или окружение ставилось до того, как pandas вернули в `requirements.txt`. Во втором случае выполни `python -m pip install -r requirements.txt` ещё раз.
- Время считай по `time_ms`, а не по `frame`: кадры идут неравномерно и с пропусками.
- Не сохраняй CSV из Excel: русский Excel меняет точки в дробях на запятые, и файл перестанет читаться.
