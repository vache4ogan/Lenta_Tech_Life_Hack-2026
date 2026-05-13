import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO
import re # Добавили библиотеку для очистки текста (регулярные выражения)

# --- НАСТРОЙКИ ---
VIDEO_PATH = "/home/vache/Projects/Lenta_TEch_CV_model/videos/25_12-20.mp4" # Путь к видео с робота
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Твоя обученная модель
CSV_OUTPUT = "supermarket_prices.csv"             # Итоговый файл

# Порог "мыла" (если четкость ниже этого числа, OCR даже не будет пытаться читать)
BLUR_THRESHOLD = 50 

print("1. Инициализация моделей...")
# Загружаем YOLO
model = YOLO(MODEL_PATH)
# Загружаем EasyOCR
reader = easyocr.Reader(['en', 'ru']) 

# Словарь для хранения самых четких ценников
best_price_tags = {}

print("2. Запуск обработки видео (Трекинг и поиск идеальных кадров)...")
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Видео закончилось

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Шаг А: УМНЫЙ КРОП (ROI) - обрезаем лишнее
    h, w = frame.shape[:2]
    roi = frame[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]

    # Шаг Б: Трекинг YOLO
    results = model.track(roi, persist=True, tracker="bytetrack.yaml", verbose=False)

    # Проверяем, нашел ли трекер хоть что-то и присвоил ли ID
    if results[0].boxes is not None and results[0].boxes.id is not None:
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            
            # Вырезаем сам ценник из ROI
            price_tag_img = roi[y1:y2, x1:x2]
            
            # Защита от ошибок (если рамка ушла за край кадра)
            if price_tag_img.size == 0: 
                continue

            # Шаг В: Оценка размытия (Фильтр Лапласиана)
            gray = cv2.cvtColor(price_tag_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Шаг Г: Логика сохранения (оставляем только самый четкий кадр)
            if track_id not in best_price_tags or sharpness > best_price_tags[track_id]["sharpness"]:
                best_price_tags[track_id] = {
                    "image": price_tag_img.copy(), 
                    "sharpness": sharpness
                }

cap.release()
print(f"Видео обработано! Найдено уникальных ценников: {len(best_price_tags)}")


print("3. Распознавание и очистка текста (OCR)...")
final_data = []

for track_id, data in best_price_tags.items():
    if data["sharpness"] < BLUR_THRESHOLD:
        continue

    img = data["image"]
    
    # --- МАГИЯ ПРЕПРОЦЕССИНГА ДЛЯ OCR ---
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # --- ЗАПУСК OCR ---
    text_results = reader.readtext(padded, detail=1)
    
    valid_texts = []
    for bbox, text, prob in text_results:
        # Убираем совсем уж неуверенные результаты (меньше 40%) и одиночные знаки препинания
        if prob > 0.4 and len(text.strip()) > 1: 
            valid_texts.append(text)
    
    if not valid_texts:
        continue
        
    full_text = " ".join(valid_texts)
    
    # === ОЧИСТКА ТЕКСТА (ПАРСИНГ) ===
    clean_words = []
    for word in full_text.split():
        # Оставляем слова длиннее 1 буквы, либо числа. Это удалит мусор от штрихкодов вроде "I", "l", "|"
        if (word.isalpha() and len(word) > 1) or word.isdigit() or word.isalnum():
            clean_words.append(word)
            
    clean_text = " ".join(clean_words)
    
    # Ищем цены (любые числа от 2 до 4 цифр). 
    # Часто OCR разбивает рубли и копейки, поэтому найдем все числа.
    prices = re.findall(r'\b\d{2,4}\b', clean_text)
    
    # Берем самое большое найденное число в качестве основной цены (обычно это рубли, а не копейки или граммы)
    if prices:
        # Конвертируем в числа для поиска максимума, потом возвращаем в строку
        likely_price = str(max([int(p) for p in prices])) 
    else:
        likely_price = "Не распознано"
    # =================================

    final_data.append({
        "ID ценника": track_id,
        "Товар (Очищенно)": clean_text,
        "Цена": likely_price,
        "Четкость кадра": round(data["sharpness"], 1),
        "Сырой текст (Дебаг)": full_text
    })
    
    print(f"ID {track_id} | Цена: {likely_price} | Товар: {clean_text}")

print("4. Сохранение в CSV...")
df = pd.DataFrame(final_data)
# Сортируем по ID для красоты
df = df.sort_values(by="ID ценника")
df.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')
print(f"Готово! Вся информация сохранена в {CSV_OUTPUT}")