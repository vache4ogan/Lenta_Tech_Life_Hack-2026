import cv2
import os
import glob

# Настройки папок
INPUT_FOLDER = "videos"
OUTPUT_FOLDER = "frames"
SECONDS_INTERVAL = 1  # Сохранять 1 кадр каждую 1 секунду

# --- НАСТРОЙКА ПОВОРОТА ---
# Попробуй один из этих вариантов, если картинка все еще не та:
# cv2.ROTATE_90_CLOCKWISE (на 90 градусов по часовой)
# cv2.ROTATE_90_COUNTERCLOCKWISE (на 90 градусов против часовой) - Обычно это!
# cv2.ROTATE_180 (на 180 градусов)
ROTATION_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE 
# ---------------------------

# Создаем папку для кадров, если её еще нет
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Ищем все mp4 видео в папке
video_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))

if not video_files:
    print(f"В папке {INPUT_FOLDER} не найдено видео файлов!")

for video_path in video_files:
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"Обработка видео: {video_name}...")

    cap = cv2.VideoCapture(video_path)
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        continue

    frame_interval = fps * SECONDS_INTERVAL
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # === ВОТ ЭТА СТРОЧКА ПОВОРАЧИВАЕТ КАДР ===
            frame = cv2.rotate(frame, ROTATION_CODE)
            # ========================================

            filename = os.path.join(OUTPUT_FOLDER, f"{video_name}_frame_{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Готово! Из {video_name} сохранено {saved_count} вертикальных кадров.")

print("Все видео успешно обработаны!")