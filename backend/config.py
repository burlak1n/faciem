import os

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "vk_faces"
EMBEDDING_DIMENSION = 512  # Изменено для DeepFace Facenet512 модели
URL_FILE_PATH = "vk_photos_data.db"  # Файл с URL-адресами от Rust-кода

# Настройки для DeepFace
DEEPFACE_MODEL_NAME = "Facenet512"  # Используем Facenet512
DEEPFACE_DETECTOR_BACKEND = "retinaface" # Примеры: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'

# Настройки обработки
MIN_DET_SCORE = 0.0  # Сниженный порог для уверенности детектора лиц (было 0.98)

# Настройки поиска
SEARCH_TOP_K = 10
# SEARCH_THRESHOLD_IP = 0.85 # <-- Убираем старый порог IP

# Пороги поиска для разных метрик
SEARCH_THRESHOLDS = {
    "cosine": 0.4,    # Меньше - более похожи (0-1, где 0 - идентичны)
    "euclidean": 0.8, # Меньше - более похожи
    "euclidean_l2": 0.8, # Синоним для euclidean
    "L2": 0.8,        # Синоним для euclidean
}

# По умолчанию, если не указана метрика
SEARCH_THRESHOLD_L2 = 0.8 

MILVUS_NPROBE = 64 # <-- Еще увеличиваем nprobe

USE_MILVUS_LITE = False
MILVUS_LITE_DATA_PATH="milvus.db"
# Настройки для скачивания
DOWNLOAD_TIMEOUT = 10 # Секунды 

PROCESSING_CHUNK_SIZE = 10
PROCESSING_CHUNK_TIMEOUT = 10

# Настройки логирования Loguru
LOGURU_LEVEL = "INFO" # Уровни: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# --- Настройки локального кэширования изображений ---
DOWNLOAD_FILES_LOCALLY = True  # Установите True для включения, False для отключения
LOCAL_DOWNLOAD_PATH = "data/downloaded_images"  # Путь для сохранения скачанных изображений
OVERWRITE_LOCAL_FILES = False # Установите True, если хотите перезаписывать файлы при скачивании, даже если они уже есть локально 

# --- Настройки сохранения извлеченных лиц ---
SAVE_EXTRACTED_FACES = True  # Включить сохранение извлеченных лиц
EXTRACTED_FACES_PATH = "data/extracted_faces"  # Путь для сохранения извлеченных лиц
FACE_IMAGE_QUALITY = 100  # Качество JPEG для сохраняемых лиц (0-100)
FACE_IMAGE_SIZE = None  # Размер (ширина и высота) для сохраняемых лиц, если None - сохраняется оригинальный размер
FACE_COLOR_CONVERT = True  # Преобразовывать RGB->BGR при сохранении (DeepFace обычно возвращает RGB, OpenCV использует BGR)
DISABLE_DARK_CHECK = True

DEEPFACE_CONCURRENCY_LIMIT = 12 # Например, по числу ядер CPU или 4 по умолчанию 

# --- Настройки асинхронного пайплайна ---
# Максимальный размер очередей между этапами
QUEUE_MAX_SIZE = 10
# Количество параллельных задач для скачивания
DOWNLOAD_WORKERS = 5
# Количество параллельных задач для детекции лиц (обычно <= DEEPFACE_CONCURRENCY_LIMIT)
EXTRACTION_WORKERS = DEEPFACE_CONCURRENCY_LIMIT 
# Количество параллельных задач для извлечения эмбеддингов (обычно <= DEEPFACE_CONCURRENCY_LIMIT)
EMBEDDING_WORKERS = DEEPFACE_CONCURRENCY_LIMIT 
# Размер пакета для вставки в Milvus
MILVUS_INSERT_BATCH_SIZE = 10
# Таймаут ожидания новых данных для Milvus worker'а перед принудительной вставкой (в секундах)
MILVUS_INSERT_TIMEOUT = 5.0 

# --- Наборы конфигураций моделей ---
MODEL_CONFIGS = {
    "retina_facenet512_cosine_aligned": {
        "model_name": "Facenet512",
        "detector_backend": "retinaface",
        "distance_metric": "cosine",
        "alignment": True, # Предполагается, что retinaface выполняет выравнивание или оно включено по умолчанию
        "search_threshold": 0.4 # Примерный порог для cosine (меньше = более похожи)
    },
    "ghost_opencv_cosine_unaligned": {
        "model_name": "GhostFaceNet",
        "detector_backend": "opencv",
        "distance_metric": "cosine",
        "alignment": False,
        "search_threshold": 0.65 # Примерный порог для GhostFaceNet + cosine (может требовать подстройки)
    },
    "ssd_facenet512_cosine": {
        "model_name": "Facenet512",
        "detector_backend": "ssd",
        "distance_metric": "cosine",
        "alignment": False,
        "search_threshold": 0.4 # Примерный порог для cosine (меньше = более похожи)
    }
}

# --- Активная конфигурация модели (можно выбрать одну из MODEL_CONFIGS) ---
# Устанавливаем одну из конфигураций как активную по умолчанию
ACTIVE_MODEL_CONFIG_NAME = "ssd_facenet512_cosine"
ACTIVE_MODEL_CONFIG = MODEL_CONFIGS[ACTIVE_MODEL_CONFIG_NAME]

# Переопределяем основные настройки на основе активной конфигурации
DEEPFACE_MODEL_NAME = ACTIVE_MODEL_CONFIG["model_name"]
DEEPFACE_DETECTOR_BACKEND = ACTIVE_MODEL_CONFIG["detector_backend"]

# Устанавливаем порог поиска на основе выбранной метрики
SEARCH_THRESHOLD = ACTIVE_MODEL_CONFIG.get("search_threshold")

# Если в конфигурации не задан порог, используем порог из SEARCH_THRESHOLDS по метрике
if SEARCH_THRESHOLD is None:
    distance_metric = ACTIVE_MODEL_CONFIG.get("distance_metric", "euclidean")
    SEARCH_THRESHOLD = SEARCH_THRESHOLDS.get(distance_metric, SEARCH_THRESHOLD_L2)

# Примечание: DeepFace.represent по умолчанию использует евклидово расстояние для Facenet и Facenet512,
# и косинусное для VGG-Face, ArcFace, Dlib, SFace, GhostFaceNet.
# Если метрика в Milvus отличается от той, что DeepFace использует для вычисления "расстояния" в результатах represent,
# то пороги нужно будет carefully настраивать и понимать, что сравнивается.

# --- Настройки для OpenCV ---
OPENCV_DATA_PATH = None
