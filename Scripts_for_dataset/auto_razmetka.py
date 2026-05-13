from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology



# Слева пишем текстовый промпт для нейросети  (что искать).
# Справа пишем название класса, которое пойдет в YOLO.
ontology = CaptionOntology({
    "price tag": "price_tag",
    "yellow price tag": "price_tag",
    "white price tag": "price_tag", 
    "red price tag": "price_tag",
})

# 2. Инициализируем модель с нашим словарем
base_model = GroundingDINO(ontology=ontology)

print("Начинаем автоматическую разметку. Это может занять время...")

# 3. Натравливаем модель на папку с твоими кадрами
dataset = base_model.label(
    input_folder="./home/vache/Projects/Lenta_TEch_CV_model/frames",       # Папка, где лежат твои картинки
    extension=".jpg",              # Формат картинок
    output_folder="./home/vache/Projects/Lenta_TEch_CV_model/ClearDataset" # Как назвать папку с готовым датасетом
)
