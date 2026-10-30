"""
Общая логика счётчика отжиманий для съёмки СПЕРЕДИ. Это Python-эталон:
мобильщики переносят его в Kotlin, когда аналитик закончит эксперименты.

MediaPipe находит точки тела, но нам нужны 12: плечи, локти, запястья,
бёдра, колени и щиколотки. Лицо, пальцы и стопы не храним и не рисуем.

Каждая точка хранится как пять чисел: (x, y, z, vis, pres)
  x, y  положение в кадре от 0 до 1 (доля ширины и высоты)
  z     глубина (нам не нужна)
  vis   насколько точка видна, от 0 до 1
  pres  насколько точка вообще в кадре, от 0 до 1
"""

import math
from collections import deque

#Наши 12 точек по порядку: номер в списке = номер точки у нас
POINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

#Номера этих же 12 точек у MediaPipe (у него всего 33 точки, с 0 по 32)
MEDIAPIPE_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

#Наши номера точек
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW, R_ELBOW = 2, 3
L_WRIST, R_WRIST = 4, 5
L_HIP, R_HIP = 6, 7
L_KNEE, R_KNEE = 8, 9
L_ANKLE, R_ANKLE = 10, 11
ARM_POINTS = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]

#Линии скелета: какие точки соединяем
SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),        #плечи и таз
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),       #левая рука
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),       #правая рука
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),        #бока корпуса
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),              #левая нога
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),              #правая нога
]

#Точка с видимостью ниже этого порога считается ненадёжной
MIN_VIS = 0.5

#Пороги счётчика: доли от «высоты» в верхней позиции.
#Это стартовые значения, их подбирают экспериментами (E4 и E5).
DOWN_LEVEL = 0.55   #ниже 55% от верхней высоты: человек «внизу»
UP_LEVEL = 0.85     #выше 85%: человек «вверху», повтор засчитан
TOP_WINDOW = 150    #«верхнюю высоту» ищем за последние 150 кадров с сигналом
SMOOTH_WINDOW = 5  #медиана последних 5 значений сигнала (E2)

#Пороги подсказок: ширина плеч в долях ширины кадра (E7)
TOO_FAR_WIDTH = 0.12  #уже этого: «слишком далеко»
TOO_CLOSE_WIDTH = 0.7  #шире этого: «слишком близко»

#Подсказки: код и текст, который человек видит на экране
HINTS = {
    "no_person": "Встаньте в кадр",
    "arms_hidden": "Не видно плеч или рук",
    "too_far": "Слишком далеко от камеры",
    "too_close": "Слишком близко к камере",
}

#Классический хват: руки чуть шире плеч. Хват меряем как расстояние между
#запястьями, делённое на ширину плеч (у классического примерно от 1,3 до 1,7).
#Пороги стартовые, их подбирают в эксперименте E8.
GRIP_NARROW = 1.1   #уже этого: руки слишком близко друг к другу
GRIP_WIDE = 2.0     #шире этого: руки слишком далеко друг от друга
GRIP_WINDOW = 15    #хват для экрана: медиана за последние 15 кадров с хватом;
                    #столько кадров подряд без хвата: окно очищается, ошибка гаснет
GRIP_MIN = 5        #меньше стольких значений хвата (за окно или за повтор): не судим, не меньше 1

#Ошибки техники: код и текст для экрана
ERRORS = {
    "hands_too_wide": "Руки слишком широко",
    "hands_too_narrow": "Руки слишком узко",
}


def px(points, i, width, height):
    """Точка номер i в пикселях: (x, y)."""
    return points[i][0] * width, points[i][1] * height


def visible(points, ids):
    """True, если все точки из списка ids хорошо видны."""
    return all(points[i][3] >= MIN_VIS for i in ids)


def distance(a, b):
    """Расстояние между двумя точками в пикселях."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shoulder_width(points, width, height):
    """Ширина плеч в пикселях."""
    return distance(px(points, L_SHOULDER, width, height),
                    px(points, R_SHOULDER, width, height))


def shoulder_height(points, width, height):
    """
    ГЛАВНЫЙ СИГНАЛ для съёмки спереди: насколько плечи выше запястий,
    измеренное в «ширинах плеч».
    Руки прямые (верх): число большое. Грудь у пола (низ): маленькое.
    Делим на ширину плеч, чтобы не зависеть от расстояния до камеры.
    Если плечи или запястья плохо видны, возвращает None.
    """
    if points is None or not visible(points, [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST]):
        return None
    sw = shoulder_width(points, width, height)
    if sw < 1:
        return None
    shoulders_y = (points[L_SHOULDER][1] + points[R_SHOULDER][1]) / 2 * height
    wrists_y = (points[L_WRIST][1] + points[R_WRIST][1]) / 2 * height
    #В картинке y растёт ВНИЗ, поэтому «плечи выше» значит y плеч меньше y запястий
    return (wrists_y - shoulders_y) / sw


def grip_ratio(points, width, height):
    """
    Ширина хвата: расстояние между запястьями, делённое на ширину плеч.
    Возвращает None, если плечи или запястья плохо видны, а ещё если плечи
    в кадре уже порога «слишком далеко»: человек далеко или стоит боком,
    и деление на почти нулевую ширину плеч даёт случайные большие числа.
    """
    if points is None or not visible(points, [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST]):
        return None
    sw = shoulder_width(points, width, height)
    if sw < 1 or sw / width < TOO_FAR_WIDTH:
        return None
    return distance(px(points, L_WRIST, width, height),
                    px(points, R_WRIST, width, height)) / sw


def median(values):
    """Медиана как в счётчике: при чётном числе значений большее из двух средних."""
    return sorted(values)[len(values) // 2]


def grip_errors(ratio):
    """Ошибки хвата по его ширине: множество кодов из ERRORS (пустое, если хват нормальный)."""
    if ratio is None:
        return set()
    if ratio > GRIP_WIDE:
        return {"hands_too_wide"}
    if ratio < GRIP_NARROW:
        return {"hands_too_narrow"}
    return set()


def angle(a, b, c):
    """Угол в точке b между отрезками b→a и b→c, в градусах."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    lengths = math.hypot(*ba) * math.hypot(*bc)
    if lengths == 0:
        return None
    cos = (ba[0] * bc[0] + ba[1] * bc[1]) / lengths
    cos = max(-1.0, min(1.0, cos))   #защита от ошибок округления
    return math.degrees(math.acos(cos))


def elbow_angle(points, width, height):
    """
    Средний угол в локтях по тем рукам, которые видно.
    Спереди этот угол неточный, поэтому для счёта не берём. Нужен для сравнения в E3.
    """
    if points is None:
        return None
    values = []
    for s, e, w in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
        if visible(points, [s, e, w]):
            a = angle(px(points, s, width, height), px(points, e, width, height),
                      px(points, w, width, height))
            if a is not None:
                values.append(a)
    if not values:
        return None
    return sum(values) / len(values)


def frame_hint(points, width, height):
    """
    Проверка кадра «человек стоит правильно?».
    Возвращает код подсказки из HINTS или None, если всё хорошо.
    Пороги стартовые, их подбирают в эксперименте E7.
    """
    if points is None:
        return "no_person"
    if not visible(points, ARM_POINTS):
        return "arms_hidden"
    sw = shoulder_width(points, width, height) / width   #ширина плеч в долях кадра
    if sw < TOO_FAR_WIDTH:
        return "too_far"
    if sw > TOO_CLOSE_WIDTH:
        return "too_close"
    return None


class PushupCounter:
    """
    Простой счётчик отжиманий.
    Класс: это «коробка», которая помнит данные между кадрами:
    где сейчас человек (вверху или внизу) и сколько повторов уже было.
    """

    def __init__(self):
        self.reps = 0                    #сколько повторов насчитали
        self.state = "up"                #считаем, что человек начинает вверху
        self.top = None                  #«высота» в верхней позиции, запоминаем сами
        self.recent = deque(maxlen=SMOOTH_WINDOW)  #последние значения сигнала
        self.smoothed = None             #сглаженное значение сигнала
        self.history = deque(maxlen=TOP_WINDOW)   #сглаженные значения за TOP_WINDOW кадров с сигналом
        self.lowest = None               #самая нижняя точка текущего повтора
        self.grips = deque(maxlen=GRIP_WINDOW)  #последние значения хвата
        self.rep_grips = []              #хват на кадрах текущего повтора, пока человек «внизу»
        self.errors = set()              #ошибки техники сейчас, для экрана
        self.error_counts = {}           #сколько засчитанных повторов сделано с каждой ошибкой
        self.judged_reps = 0             #сколько засчитанных повторов оценено по хвату (для E8)
        self.no_grip = 0                 #сколько кадров подряд без хвата

    def update(self, value, grip=None):
        """
        Даём счётчику сигнал и хват с нового кадра. Возвращает число повторов.
        Хват (grip_ratio) нужен только для ошибок техники, на счёт он не влияет.
        """
        if grip is not None:
            self.no_grip = 0
            self.grips.append(grip)
            if self.state == "down":
                self.rep_grips.append(grip)
        else:
            #Человек ушёл или руки долго не видны: старый хват больше не показываем
            self.no_grip += 1
            if self.no_grip >= GRIP_WINDOW:
                self.grips.clear()
        enough = len(self.grips) >= max(GRIP_MIN, 1)
        self.errors = grip_errors(median(self.grips)) if enough else set()

        if value is None:                #точек не видно: пропускаем кадр
            return self.reps

        #Сглаживание: медиана последних SMOOTH_WINDOW значений, чтобы убрать скачки точек
        self.recent.append(value)
        self.smoothed = sorted(self.recent)[len(self.recent) // 2]

        #Верхняя позиция: самая большая высота за последние TOP_WINDOW кадров.
        #Кадры без сигнала сюда не попадают и окно не сдвигают.
        #Если сигнал случайно «скакнул» (например, человек шёл к камере боком),
        #через TOP_WINDOW кадров счётчик это забудет.
        self.history.append(self.smoothed)
        self.top = max(self.history)

        if self.state == "up":
            if self.smoothed < DOWN_LEVEL * self.top:
                self.state = "down"      #опустился
                self.lowest = self.smoothed
                self.rep_grips = [] if grip is None else [grip]
        else:
            self.lowest = min(self.lowest, self.smoothed)
            if self.smoothed > UP_LEVEL * self.top:
                self.state = "up"        #поднялся обратно
                #Засчитываем, только если внизу правда опускался ниже порога.
                #Это защита от ложного повтора, когда «верх» пересчитался после скачка.
                if self.lowest < DOWN_LEVEL * self.top:
                    self.reps += 1
                    #Ошибка повтора: по медиане хвата, пока человек был «внизу»
                    #В rep_grips входят и кадр перехода «вниз», и кадр перехода «вверх»
                    if len(self.rep_grips) >= max(GRIP_MIN, 1):
                        self.judged_reps += 1
                        rep_errors = grip_errors(median(self.rep_grips))
                    else:
                        rep_errors = set()
                    for error in rep_errors:
                        self.error_counts[error] = self.error_counts.get(error, 0) + 1
        return self.reps
