"""
Точки тела с веб-камеры: скелет и счётчик на экране, точки в CSV.

MediaPipe работает в режиме LIVE_STREAM, как на телефоне: модель считает
в фоне и пропускает кадры, пока занята. В CSV попадают только кадры,
которые модель обработала, поэтому номера кадров могут идти с пропусками.

Запуск из папки проекта:
  python extract_pose.py                 #веб-камера 0, модель full
  python extract_pose.py --input 1       #другая камера
  python extract_pose.py --model lite    #модель полегче

Выход по Esc или крестиком окна. Результат: output/csv/<модель>/webcam_<время>.csv
и рядом .meta.json с итогами.
"""

import argparse
import csv
import json
import os
import queue
import ssl
import time
import urllib.request
from datetime import datetime
from functools import lru_cache
from pathlib import Path

#Ускоряет запуск веб-камеры на Windows (на Mac ни на что не влияет)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import certifi
import cv2
import matplotlib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pushup_logic import (HINTS, MEDIAPIPE_IDS, MIN_VIS, POINT_NAMES, SKELETON,
                          PushupCounter, frame_hint, shoulder_height)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{m}/float16/latest/pose_landmarker_{m}.task")

#Папки считаем от места скрипта, чтобы запуск работал из любой папки
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

#Название окна только латиницей: русские буквы в заголовке Windows показывает криво
WINDOW = "Pushups (Esc to exit)"
ESC = 27

#Шрифт с кириллицей, он приходит вместе с matplotlib
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"

#Цвета в порядке OpenCV: синий, зелёный, красный
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def get_model(name):
    """Путь к файлу модели. Если файла нет, скачивает его в models/."""
    path = MODELS_DIR / f"pose_landmarker_{name}.task"
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю модель {name}...")
        #certifi: список сертификатов, чтобы на Mac не было ошибки CERTIFICATE_VERIFY_FAILED
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL.format(m=name), context=context) as response:
            data = response.read()
        #Сначала во временный файл: если скачивание оборвётся, битой модели не останется
        part = path.with_suffix(".part")
        part.write_bytes(data)
        part.replace(path)
        print("Модель скачана:", path)
    return str(path)


def to_points(result):
    """Наши 12 точек из ответа MediaPipe: [(x, y, z, vis, pres), ...] или None."""
    if not result.pose_landmarks:                    #человек не найден
        return None
    body = result.pose_landmarks[0]                  #все 33 точки
    return [(body[i].x, body[i].y, body[i].z, body[i].visibility or 0.0,
             body[i].presence or 0.0) for i in MEDIAPIPE_IDS]


def make_landmarker(model_path, results):
    """Детектор позы в режиме LIVE_STREAM. Ответы кладёт в очередь results."""
    def on_result(result, image, time_ms):
        #Вызывается в другом потоке, поэтому только кладём ответ в очередь
        results.put((time_ms, to_points(result)))

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_poses=1,                                 #ищем одного человека
        result_callback=on_result,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


@lru_cache(maxsize=None)
def font(size):
    """Шрифт нужного размера."""
    return ImageFont.truetype(str(FONT_PATH), size)


@lru_cache(maxsize=256)
def label(text, size, color):
    """Картинка с надписью на прозрачном фоне. Одинаковые надписи берём из кэша."""
    f = font(size)
    stroke = max(1, size // 12)
    left, top, right, bottom = f.getbbox(text, stroke_width=stroke)
    picture = Image.new("RGBA", (right - left, bottom - top))
    #Цвет в порядке BGR, как у кадра: Pillow просто кладёт числа в каналы по порядку
    ImageDraw.Draw(picture).text((-left, -top), text, font=f, fill=color,
                                 stroke_width=stroke, stroke_fill=BLACK)
    return picture


def write(image, text, x, y, size, color):
    """Кладёт надпись на кадр. Прозрачный фон надписи кадр не закрывает."""
    picture = label(text, max(12, size), color)
    image.paste(picture, (x, y), picture)


def draw(frame, points, reps, hint, fps):
    """Рисует на кадре 12 точек, скелет, счётчик, подсказку и скорость."""
    height, width = frame.shape[:2]
    scale = width / 800                              #размеры под размер кадра
    thick = max(2, int(3 * scale))

    if points:
        for a, b in SKELETON:                        #линии скелета
            if points[a][3] > MIN_VIS and points[b][3] > MIN_VIS:
                pa = (int(points[a][0] * width), int(points[a][1] * height))
                pb = (int(points[b][0] * width), int(points[b][1] * height))
                cv2.line(frame, pa, pb, GREEN, thick)
        for x, y, z, vis, pres in points:            #сами 12 точек
            if vis > MIN_VIS:
                cv2.circle(frame, (int(x * width), int(y * height)), thick + 2, RED, -1)

    #Надписи через Pillow: OpenCV не умеет русские буквы
    image = Image.fromarray(frame)
    write(image, f"Повторы: {reps}", 20, int(15 * scale), int(48 * scale), YELLOW)
    if hint:
        write(image, HINTS[hint], 20, int(80 * scale), int(32 * scale), RED)
    write(image, f"Модель: {fps:.0f} кадров/с", 20, height - int(45 * scale),
          int(26 * scale), WHITE)
    frame[:] = np.asarray(image)                     #возвращаем картинку с надписями в кадр


def csv_header():
    """Названия колонок CSV: 6 служебных и по 5 на каждую из 12 точек."""
    columns = ["video", "frame", "time_ms", "width", "height", "detected"]
    for name in POINT_NAMES:
        columns += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis", f"{name}_pres"]
    return columns


def csv_row(video, frame_no, time_ms, width, height, points):
    """Одна строка CSV для одного кадра."""
    row = [video, frame_no, time_ms, width, height, 1 if points else 0]
    if points:
        for point in points:
            row += [round(value, 5) for value in point]
    else:
        row += [""] * (len(POINT_NAMES) * 5)         #человек не найден: пустые клетки
    return row


def window_closed():
    """True, если окно закрыли крестиком (на Mac OpenCV этого не сообщает)."""
    return os.name == "nt" and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def run_webcam(camera, model_name, model_path):
    """Веб-камера: скелет и счётчик на экране, точки в CSV."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Камера {camera} не открылась. Проверь доступ к камере в настройках "
              f"или попробуй другой номер: --input {1 if camera == 0 else 0}")
        return

    csv_dir = OUTPUT_DIR / "csv" / model_name
    csv_dir.mkdir(parents=True, exist_ok=True)
    name = "webcam_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"{name}.csv"

    counter = PushupCounter()
    results = queue.SimpleQueue()                    #ответы модели из её потока
    sent = {}                                        #время кадра: (номер кадра, ширина, высота)
    points, hint = None, "no_person"
    frame_no = processed = 0
    first_ms = last_ms = -1
    start = time.monotonic()
    recent = []                                      #время последних ответов модели, для скорости

    def handle(table):
        """Забирает готовые ответы модели: строка CSV, счётчик, подсказка."""
        nonlocal points, hint, processed
        while not results.empty():
            time_ms, points = results.get()
            #Кадры раньше этого модель пропустила, ответа на них не будет
            while next(iter(sent)) < time_ms:
                del sent[next(iter(sent))]
            no, width, height = sent.pop(time_ms)
            table.writerow(csv_row(name, no, time_ms, width, height, points))
            counter.update(shoulder_height(points, width, height))
            hint = frame_hint(points, width, height)
            processed += 1
            recent.append(time.monotonic())
        del recent[:-30]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        table = csv.writer(f)
        table.writerow(csv_header())

        with make_landmarker(model_path, results) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Камера перестала отдавать кадры")
                    break
                height, width = frame.shape[:2]

                #Время кадра должно строго расти, иначе MediaPipe выдаст ошибку
                time_ms = max(int((time.monotonic() - start) * 1000), last_ms + 1)
                last_ms = time_ms
                if frame_no == 0:
                    first_ms = time_ms
                sent[time_ms] = (frame_no, width, height)

                #OpenCV хранит цвета как BGR, а MediaPipe ждёт RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                        time_ms)
                frame_no += 1

                handle(table)
                fps = 0
                if len(recent) > 1:
                    fps = (len(recent) - 1) / (recent[-1] - recent[0])
                draw(frame, points, counter.reps, hint, fps)
                cv2.imshow(WINDOW, frame)
                if (cv2.waitKey(1) & 0xFF) == ESC or window_closed():
                    break

        #После закрытия детектора забираем ответы, которые пришли в последний момент
        handle(table)

    cap.release()
    cv2.destroyAllWindows()

    #Скорость считаем только по времени кадров: запуск и закрытие модели не в счёт
    seconds = (last_ms - first_ms) / 1000
    camera_fps = (frame_no - 1) / seconds if seconds > 0 else 0
    meta = {"video": name, "model": model_name, "mode": "LIVE_STREAM",
            "frames": frame_no, "processed": processed,
            "camera_fps": round(camera_fps, 1),
            "processing_fps": round(camera_fps * processed / frame_no, 1) if frame_no else 0,
            "reps_counted": counter.reps}
    (csv_dir / f"{name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: насчитано отжиманий: {counter.reps}. Кадров с камеры: {frame_no}, "
          f"обработала модель: {processed}. Точки сохранены в {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Точки тела с веб-камеры")
    parser.add_argument("--input", type=int, default=0,
                        help="номер веб-камеры: 0, 1 и так далее (по умолчанию 0)")
    parser.add_argument("--model", default="full", choices=["lite", "full", "heavy"],
                        help="какая модель MediaPipe (по умолчанию full)")
    args = parser.parse_args()
    run_webcam(args.input, args.model, get_model(args.model))


if __name__ == "__main__":
    main()
