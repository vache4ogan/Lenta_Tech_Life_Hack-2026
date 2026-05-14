import cv2
from pyzbar.pyzbar import decode

# Загружаем твою картинку
img = cv2.imread("/home/vache/Projects/Lenta_TEch_CV_model/debug_tag_1.jpg")

# 1. ТРЮК С КАНАЛОМ: Берем только Красный канал (индекс 2 в BGR)
# Оранжевый фон станет белым, черные линии - черными!
red_channel = img[:, :, 2]

# 2. Добавляем "Тихую зону" (белые поля)
padded = cv2.copyMakeBorder(red_channel, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

# 3. Увеличиваем без сглаживания
zoomed = cv2.resize(padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

# 4. Умная бинаризация Оцу (алгоритм сам найдет идеальный порог черного и белого)
_, thresh = cv2.threshold(zoomed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Сохраним, чтобы ты посмотрел, какая идеальная зебра получилась
cv2.imwrite("magic_barcode.jpg", thresh)

# Сканируем!
results = decode(thresh)
for obj in results:
    print("НАЙДЕНО:", obj.type, obj.data.decode("utf-8"))