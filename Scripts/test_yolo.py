import cv2
from ultralytics import YOLO

# --- НАСТРОЙКИ ---
# Путь к твоей обученной модели. Обычно YOLO сохраняет лучшую версию сюда:
MODEL_PATH = "runs/detect/train/weights/best.pt" 

# Возьми любую картинку из твоей папки frames для теста
IMAGE_PATH = "/home/vache/Projects/Lenta_TEch_CV_model/frames/25_12-20_frame_1.jpg" 

# 1. Загружаем модель
print("Загрузка модели...")
model = YOLO(MODEL_PATH)

# 2. Читаем фотографию
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"Ошибка: Не удалось найти картинку по пути {IMAGE_PATH}")
    exit()

h, w = frame.shape[:2]
print(f"Оригинальное разрешение: {w}x{h}")



y_start = int(h * 0.30)  
y_end = int(h * 0.90)   


x_start = int(w * 0.10)
x_end = int(w * 0.90)
# Вырезаем кусок с помощью среза numpy (сначала идет высота Y, потом ширина X)
cropped_frame = frame[y_start:y_end, x_start:x_end]
print(f"Размер после обрезки: {cropped_frame.shape[1]}x{cropped_frame.shape[0]}")

# 4. Отдаем кадр в YOLO
print("Поиск ценников...")
results = model(cropped_frame, conf=0.3)

# 5. Отрисовка результата
# Метод .plot() сам нарисует красивые рамочки и напишет уверенность над каждым ценником
annotated_frame = results[0].plot()

# 6. Сохраняем результат, чтобы посмотреть глазами
OUTPUT_PATH = "test_result.jpg"
cv2.imwrite(OUTPUT_PATH, annotated_frame)
print(f"Готово! Результат сохранен в {OUTPUT_PATH}. Открой файл и посмотри!")

