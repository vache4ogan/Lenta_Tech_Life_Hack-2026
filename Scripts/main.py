import os
import cv2
from paddleocr import PaddleOCR 
import pandas as pd
import numpy as np
from ultralytics import YOLO
import re
from pyzbar.pyzbar import decode

# --- НАСТРОЙКИ ---
VIDEO_PATH = "/home/vache/Projects/Lenta_TEch_CV_model/start_material/videos/25_12-20.mp4" 
MODEL_PATH = "/home/vache/Projects/Lenta_TEch_CV_model/runs/detect/train/weights/best.pt"  
CSV_OUTPUT = "supermarket_prices_pro.csv"             

BLUR_THRESHOLD = 50 
MOTION_THRESHOLD = 5.0 # Порог движения. Если меньше - робот стоит, кадр пропускаем

print("1. Инициализация моделей (YOLO + EasyOCR RU/EN)...")
model = YOLO(MODEL_PATH)

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR) # Глушим системный мусор от Paddle

# Инициализируем свежую версию с правильными параметрами
# Никаких предупреждений, классический синтаксис
ocr_reader = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False)

best_price_tags = {}
prev_gray_roi = None # Для детектора движения

print("2. Запуск обработки видео (Трекинг + Оптический поток)...")
cap = cv2.VideoCapture(VIDEO_PATH)
filename = os.path.basename(VIDEO_PATH)

frames_skipped = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    h, w = frame.shape[:2]
    
    # Координаты нашей "Золотой зоны"
    roi_y_start, roi_y_end = int(h * 0.3), int(h * 0.9)
    roi_x_start, roi_x_end = int(w * 0.1), int(w * 0.9)
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # --- ФИШКА 1: Детектор движения (Пропускаем кадры, если робот стоит) ---
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi_small = cv2.resize(gray_roi, (0, 0), fx=0.1, fy=0.1) # Уменьшаем для скорости
    
    if prev_gray_roi is not None:
        mse = np.mean((gray_roi_small - prev_gray_roi) ** 2)
        if mse < MOTION_THRESHOLD:
            frames_skipped += 1
            continue # Пропускаем кадр, робот не сдвинулся!
    prev_gray_roi = gray_roi_small

    # Шаг Б: Трекинг YOLO
    results = model.track(roi, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            
            # --- ФИШКА 2: Padding (Защита штрихкодов от обрезания) ---
            pad = 20
            y1_pad = max(0, y1 - pad)
            y2_pad = min(roi.shape[0], y2 + pad)
            x1_pad = max(0, x1 - pad)
            x2_pad = min(roi.shape[1], x2 + pad)
            
            price_tag_img = roi[y1_pad:y2_pad, x1_pad:x2_pad]
            if price_tag_img.size == 0: continue

            # Оценка размытия
            gray_tag = cv2.cvtColor(price_tag_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_tag, cv2.CV_64F).var()

            if track_id not in best_price_tags or sharpness > best_price_tags[track_id]["sharpness"]:
                best_price_tags[track_id] = {
                    "image": price_tag_img.copy(), 
                    "sharpness": sharpness,
                    "timestamp": round(timestamp_sec, 2),
                    "x_min": x1 + roi_x_start, "y_min": y1 + roi_y_start,
                    "x_max": x2 + roi_x_start, "y_max": y2 + roi_y_start
                }

cap.release()
print(f"Видео обработано! Уникальных ценников: {len(best_price_tags)}. Кадров сэкономлено: {frames_skipped}")

print("3. Распознавание OCR и Парсинг Штрихкодов...")
final_data = []

# Инициализируем CLAHE (выравниватель теней)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for track_id, data in best_price_tags.items():
    if data["sharpness"] < BLUR_THRESHOLD: continue

    img = data["image"]
    
    # --- МОДУЛЬ ШТРИХКОДОВ (Трюк с Красным Каналом + Морфология) ---
    red_channel = img[:, :, 2] # Оранжевый фон становится белым
    barcode_padded = cv2.copyMakeBorder(red_channel, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    
    # Адаптивный порог для спасения штрихкодов в тени
    thresh = cv2.adaptiveThreshold(barcode_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    
    # Лечим разорванные линии
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    healed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    barcode_val, qr_raw_val = "нет", "нет"
    for variant in [barcode_padded, healed_thresh]:
        decoded_objects = decode(variant)
        if not decoded_objects:
            # Пробуем повернуть
            decoded_objects = decode(cv2.rotate(variant, cv2.ROTATE_90_CLOCKWISE))
            
        if decoded_objects:
            for obj in decoded_objects:
                if obj.type == 'QRCODE': qr_raw_val = obj.data.decode("utf-8")
                else: barcode_val = obj.data.decode("utf-8")
            break # Нашли - выходим из цикла

    # --- МОДУЛЬ OCR (С применением CLAHE) ---
   # --- МОДУЛЬ OCR (PaddleOCR) ---
    # --- МОДУЛЬ OCR (PaddleOCR 2.8.1) ---
    # --- МОДУЛЬ OCR (PaddleOCR 2.8.1 - ИДЕАЛ) ---
    # 1. Возвращаем увеличение (x2.0). Без него русские буквы слипаются.
    img_ocr = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Хак резкости (Sharpening): убираем "мыло" от видео, чтобы "Л" не казалась "N"
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_ocr, -1, kernel)
    
    padded_ocr = cv2.copyMakeBorder(img_sharp, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Запускаем распознавание
    text_results = ocr_reader.ocr(padded_ocr, cls=True)
    
    valid_texts = []
    if text_results and text_results[0] is not None:
        for line in text_results[0]:
            text = line[1][0]
            prob = line[1][1]
            
            # Берем текст с уверенностью больше 50%
            if prob > 0.5: 
                valid_texts.append(text)
                
    full_text = " ".join(valid_texts)
    
    # === ГЛАВНЫЙ ФИКС ЦЕН ===
    # Ищем цены в сыром тексте ДО того, как Python удалит точки и запятые!
    prices = re.findall(r'\b\d{2,4}\b', full_text)
    
    # Чистим текст для колонки "product_name" (удаляем одиночные левые символы)
    clean_words = []
    for w in full_text.split():
        # Оставляем слова длиннее 1 буквы, либо если это просто цифра
        if len(w) > 1 or w.isdigit():
            clean_words.append(w)
            
    clean_text = " ".join(clean_words)
    
    # --- ФИШКА 3: Деривация полей (Математика цен и скидок) ---
    prices = re.findall(r'\b\d{2,4}\b', clean_text)
    price_default = "нет"
    price_card = "нет"
    discount_amount = "нет"
    
    if prices:
        raw_prices = list(set([int(p) for p in prices]))
        
        # Если есть хотя бы одна цена больше 100 рублей, то "99", "90" и "00" - это точно копейки, удаляем их!
        if any(p >= 100 for p in raw_prices):
            raw_prices = [p for p in raw_prices if p not in [99, 90, 00]]
            
        unique_prices = sorted(raw_prices, reverse=True)
        # Уникальные цены, отсортированные по убыванию (например, [144, 129])
        
        price_default = str(unique_prices[0])
        if len(unique_prices) > 1:
            price_card = str(unique_prices[1])
            # Высчитываем скидку математически!
            calc_discount = int((1 - (unique_prices[1] / unique_prices[0])) * 100)
            if calc_discount > 0:
                discount_amount = f"-{calc_discount}%"

    final_data.append({
        "filename": filename,
        "product_name": clean_text,
        "price_default": price_default,
        "price_card": price_card,
        "price_discount": "нет",
        "barcode": barcode_val,
        "discount_amount": discount_amount,
        "id_sku": "нет",
        "print_datetime": "нет",
        "code": "нет",
        "additional_info": "нет",
        "color": "нет",
        "special_symbols": "нет",
        "frame_timestamp": data["timestamp"],
        "x_min": data["x_min"], "y_min": data["y_min"],
        "x_max": data["x_max"], "y_max": data["y_max"],
        "qr_code_barcode": qr_raw_val,
        "price1_qr": "нет", "price2_qr": "нет", "price3_qr": "нет", "price4_qr": "нет",
        "wholesale_level_1_coun": "нет", "wholesale_level_1_price": "нет",
        "wholesale_level_2_count": "нет", "wholesale_level_2_price": "нет",
        "action_price_qr": "нет", "action_code_qr": "нет"
    })
    
    print(f"ID {track_id} | OCR: {clean_text} | Штрих: {barcode_val} | БЕЗ карты: {price_default} | ПО карте: {price_card} | Скидка: {discount_amount}")

print("4. Сохранение в CSV...")
df = pd.DataFrame(final_data)
df = df.sort_values(by="ID ценника", errors='ignore') # errors='ignore' на случай если колонка не добавлена
df.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig', sep='\t')
print(f"Успешно! Эпичная таблица сохранена в {CSV_OUTPUT}")