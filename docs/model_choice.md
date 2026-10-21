# Сравнение моделей распознавания позы

Берём **MediaPipe Pose Landmarker**. Для приложения на Kotlin Multiplatform у него самый надёжный путь:

- официальные библиотеки для Android (Kotlin) и iOS (Swift Package Manager);
- одна и та же модель на телефонах и на компьютере, где мы подбираем параметры;
- все наши 12 точек, у каждой есть оценка видимости.

Официальной поддержки KMP нет ни у одной модели. Поэтому модуль точек в любом случае пишется отдельно под Android и под iOS, а общий код получает только готовые точки.

## Сравнение

| Решение | Точки | Android и iOS | Лицензия | Главный минус для нас |
| --- | --- | --- | --- | --- |
| [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) (Google) | 33, все наши 12 есть, у каждой видимость | Android: Kotlin, библиотека `tasks-vision`. iOS: Swift, через Swift Package Manager | Apache 2.0 | нет готовой KMP-библиотеки: модуль точек пишем под каждую платформу |
| [ML Kit Pose Detection](https://developers.google.com/ml-kit/vision/pose-detection) (Google) | 33, та же разметка, что у MediaPipe | Android: Gradle. iOS: только CocoaPods (`GoogleMLKit/PoseDetection`) | бесплатно, модель закрытая | статус «бета» без гарантий; нет версии для компьютера; CocoaPods со 2 декабря 2026 не принимает новые версии |
| [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) (Google) | 17, все наши 12 есть | через LiteRT (бывший TensorFlow Lite) на обеих | бесплатно | у LiteRT нет официальной поддержки KMP; подготовку кадра и разбор ответа модели писать самим, дважды |
| [YOLO26-pose](https://docs.ultralytics.com/tasks/pose/) (Ultralytics) | 17, все наши 12 есть | после экспорта в TFLite для Android и CoreML для iOS | [AGPL-3.0](https://www.ultralytics.com/license) | код всего приложения пришлось бы открыть; обвязку писать самим |
| [Apple Vision](https://developer.apple.com/documentation/vision/detecting-human-body-poses-in-images) | 19, все наши 12 есть | только iOS, встроено в систему | бесплатно | на Android не работает: пришлось бы держать две разные модели |

OpenPose из прошлой версии сравнения убрали: он рассчитан на компьютер и для телефона не подходит.

## Почему MediaPipe подходит именно нам

1. **Одна модель везде.** Одна и та же модель работает на Android, iOS и на компьютере в Python. Общий код KMP получает одинаковые точки с обеих платформ, а параметры, подобранные на компьютере по CSV, переносятся на телефон. У ML Kit нет версии для компьютера, а Apple Vision нет на Android.
2. **Официальные библиотеки на обеих платформах.** На Android это [`com.google.mediapipe:tasks-vision`](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android), на iOS [`MediaPipeTasksVision` через Swift Package Manager](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/ios). От CocoaPods не зависим, а он [со 2 декабря 2026 перестаёт принимать новые версии библиотек](https://blog.cocoapods.org/CocoaPods-Specs-Repo/).
3. **Реальное время из коробки.** Режим LIVE\_STREAM есть на обеих платформах: модель работает в фоне и сама пропускает кадры, пока занята, поэтому камера не тормозит.
4. **Все наши 12 точек с оценкой видимости.** На `visibility` и `presence` построены подсказки и очистка точек.
5. **Три размера модели.** lite, full и heavy подключаются одинаково, меняется только файл. Под слабый телефон можно взять модель полегче без переделки кода.
6. **Свободная лицензия.** Apache 2.0: бесплатно, и код приложения открывать не нужно.
7. **Путь уже пройден.** Есть [открытый KMP-проект](https://github.com/sonms/GoogleMediaPipeTest/pull/1), где MediaPipe работает на Android и iOS за одним общим интерфейсом. Там распознаётся лицо, но схема та же, что у нас.

## Как MediaPipe встраивается в KMP

На Android всё просто, выбирать приходится только на iOS.

| Платформа | Как подключаем | Плюсы и минусы |
| --- | --- | --- |
| Android | Kotlin в `androidMain`, библиотека `tasks-vision` | официальный прямой путь |
| iOS, вариант 1 (предлагаем) | класс на Swift в iOS-приложении реализует интерфейс `PoseSource` из общего кода; MediaPipe ставим через Swift Package Manager | официальный путь MediaPipe без промежуточных слоёв; нужен человек, который пишет на Swift |
| iOS, вариант 2 | Kotlin в `iosMain` через cinterop и CocoaPods | так сделано в открытом примере, но CocoaPods со 2 декабря 2026 не принимает новые версии библиотек |
| iOS, вариант 3 | Kotlin в `iosMain` через [импорт Swift-пакетов](https://kotlinlang.org/docs/multiplatform/multiplatform-spm-import.html) | весь код на Kotlin, но функция пока в альфе и требует плагин KMP 2.5.0-Beta1 |

Во всех вариантах общий код получает одно и то же: `PoseFrame` из документа [Контракт данных](data_contract.md). Поэтому вариант на iOS можно поменять позже, не трогая остальное приложение.

## Какую модель брать: lite, full или heavy

По умолчанию берём full, а lite оставляем запасным вариантом для слабых телефонов.

| Модель | Размер файла, МБ | Точность, % (PCK@0.2) | Время на кадр, мс (Pixel 3, GPU) | Когда брать |
| --- | --- | --- | --- | --- |
| lite | 5,8 | 93,5 | 20 | слабые телефоны |
| full | 9,4 | 95,7 | 25 | по умолчанию |
| heavy | 30,7 | 97,5 | 53 | только для сравнения на компьютере |

Точность и время взяты из [старого замера Google](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#pose-estimation-quality) на наборе с фитнес-упражнениями (HIIT) и телефоне Pixel 3. На телефонах команды проверяем сами, это эксперимент E6.

## Риски и что проверить

- **Два модуля точек могут разойтись.** Android и iOS могут по-разному поворачивать или зеркалить кадр. Ловим сверкой: CSV с телефона прогоняем через `replay_csv.py`, он сверяет счёт и предупреждает о повёрнутом или зеркальном кадре.
- **Видеокарта на iOS.** В руководстве MediaPipe для iOS про ускорение на видеокарте ничего не сказано. Скорость lite и full на iPhone команды нужно замерить.
- **Размер приложения.** Модель добавляет от 6 до 31 МБ, плюс сама библиотека MediaPipe.
- **Запасной вариант.** Если MediaPipe не заработает на каком-то телефоне, есть ML Kit с той же разметкой 33 точек: контракт и общий код не меняются. Но на iOS он ставится только через CocoaPods.
- **Следим за KMP.** Когда импорт Swift-пакетов выйдет из альфы, iOS-модуль точек можно будет перенести со Swift на Kotlin.

## Источники

- [MediaPipe Pose Landmarker: обзор](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Pose Landmarker: Android](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android)
- [MediaPipe Pose Landmarker: iOS, установка через Swift Package Manager](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/ios)
- [MediaPipe Pose: старый замер точности и скорости](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#pose-estimation-quality)
- [ML Kit Pose Detection: обзор и статус «бета»](https://developers.google.com/ml-kit/vision/pose-detection)
- [ML Kit Pose Detection: iOS, установка через CocoaPods](https://developers.google.com/ml-kit/vision/pose-detection/ios)
- [CocoaPods: план перевода репозитория в режим «только чтение»](https://blog.cocoapods.org/CocoaPods-Specs-Repo/)
- [Kotlin: импорт Swift-пакетов в KMP (альфа)](https://kotlinlang.org/docs/multiplatform/multiplatform-spm-import.html)
- [KMP-проект с MediaPipe на Android и iOS](https://github.com/sonms/GoogleMediaPipeTest/pull/1)
- [LiteRT и Kotlin Multiplatform: обсуждение на форуме Google AI](https://discuss.ai.google.dev/t/tf-lite-for-kotlin-multiplatform-jvm-js-ios/28550)
- [MoveNet: туториал TensorFlow](https://www.tensorflow.org/hub/tutorials/movenet)
- [Ultralytics: оценка позы, YOLO26](https://docs.ultralytics.com/tasks/pose/)
- [Ultralytics: лицензия](https://www.ultralytics.com/license)
- [Apple Vision: Detecting Human Body Poses in Images](https://developer.apple.com/documentation/vision/detecting-human-body-poses-in-images)
