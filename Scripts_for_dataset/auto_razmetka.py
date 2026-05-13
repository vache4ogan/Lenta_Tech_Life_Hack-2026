import os
import cv2
import supervision as sv
from pathlib import Path
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# 1. Настройка путей
HOME = Path.cwd()
INPUT_DIR = "/home/vache/Projects/Lenta_TEch_CV_model/frames"
OUTPUT_DIR = HOME / "ClearDataset"

# Создаем папку вывода, если ее нет
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Онтология с широким охватом (синонимы)
# Мы перечисляем разные варианты, но все они сохранятся как 'price_tag' (ID 0)
ontology = CaptionOntology({
    "price tag": "price_tag",
    "yellow price tag": "price_tag",
    "white label on shelf": "price_tag",
    "barcode on price tag": "price_tag",
    "qr code": "price_tag"
})

# 3. Инициализация модели
# Снижаем пороги, чтобы модель была более "чувствительной" к мелким ценникам
base_model = GroundingDINO(ontology=ontology)

# 4. Функция ручной разметки с фильтрацией дубликатов (NMS)
def label_dataset(input_folder, output_folder, extension=".jpg", proximity_threshold=0.5):
    images = list(Path(input_folder).glob(f"*{extension}"))
    print(f"Найдено изображений: {len(images)}")
    
    for img_path in images:
        # Загружаем картинку
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        # Предсказание (используем низкий порог для захвата всех ценников)
        # box_threshold=0.20 поможет найти ценники в тени или под углом [cite: 180, 182, 183]
        detections = base_model.predict(image)
        
        # Убираем дубликаты (NMS)
        # Если две рамки пересекаются более чем на 50%, оставляем только одну
        detections = detections.with_nms(threshold=proximity_threshold)
        
        # Сохраняем результат в формате YOLO
        img_name = img_path.stem
        label_path = Path(output_folder) / f"{img_name}.txt"
        
        # Записываем аннотации
        with open(label_path, "w") as f:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = detections.confidence[i]
                class_id = detections.class_id[i]
                
                # Переводим в формат YOLO (нормализованный)
                h, w, _ = image.shape
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Записываем строку (всегда класс 0, так как у нас один целевой класс)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Копируем саму картинку в папку датасета (опционально)
        cv2.imwrite(str(Path(output_folder) / img_path.name), image)
        
    print(f"Разметка завершена. Результаты в папке: {output_folder}")

# 5. Запуск

label_dataset(INPUT_DIR, OUTPUT_DIR)