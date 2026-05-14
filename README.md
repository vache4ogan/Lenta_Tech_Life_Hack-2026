# 🛒 ShelfVision: Lenta Tech CV Pipeline

**ShelfVision** — высокопроизводительный комплекс для аудита ценников. Система оптимизирована для работы на CPU с видеопотоком 4K в условиях активного движения робота-мерчандайзера.

---

## 🌟 Ключевые особенности
* YOLOv8 + PaddleOCR: Двухэтапная связка для мгновенной детекции и точного извлечения данных.
* ROI Optimization: Обработка только «активной зоны» кадра (ускорение в 15 раз).
* 4K Mapping: Алгоритм пересчета координат из 1080p в оригинальный 4K-эталон.
* Устойчивость к Blur: Препроцессинг Laplacian Sharpening для четкости текста в динамике.

---

## 🛠 Технический стек
* YOLOv8: Детекция Bounding Boxes.
* PaddleOCR v2.8.1: Распознавание кириллицы и цен.
* PaddlePaddle v2.6.2: Стабильный backend для CPU (Linux).
* OpenCV / Pandas: Препроцессинг и формирование отчетов.

---

## 📂 Структура проекта
```text
ShelfVision/
├── Scripts/
│   ├── main.py            # Основной конвейер обработки
│   ├── metrics_eval.py    # Расчет точности (IoU)
│   └── utils.py           # Маппинг и препроцессинг
├── Models/
│   └── custom_yolo.pt     # Веса модели YOLOv8
└── Data/
    ├── input/             # Видеофайлы (4K)
    └── output/            # Результаты (CSV)

```
---

## 🏗 Архитектура решения
1. Motion Control: Фильтрация статичных кадров для экономии ресурсов.
2. Detection: Поиск ценников через YOLOv8 в динамической зоне ROI.
3. OCR Phase: Обрезка, Padding и распознавание текста (PaddleOCR).
4. Logic Engine: Парсинг цен регулярными выражениями и расчет скидок.
5. Exporter: Генерация CSV (utf-8-sig) для корректного открытия в Excel.

---

## 📦 Установка и запуск
1. Установка стека: pip install ultralytics pandas opencv-python paddlepaddle==2.6.2 paddleocr==2.8.1
2. Запуск пайплайна: python Scripts/main.py --input data/2.mp4 --output results.csv

---



## 📄 Авторы
Vache — Lead CV Engineer / ML Developer
ShelfVision 2026 | Специально для Lenta Tech