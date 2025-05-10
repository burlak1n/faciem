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
MIN_DET_SCORE = 0.98  # Минимальный порог для уверенности детектора лиц (может потребовать подстройки для DeepFace)

# Настройки поиска
SEARCH_TOP_K = 10
# SEARCH_THRESHOLD_IP = 0.85 # <-- Убираем старый порог IP
SEARCH_THRESHOLD_L2 = 0.8 # Порог для L2 метрики (меньше = более похожи). Типично 1.0-1.2 для Facenet.
# Для Facenet512 порог обычно выше, чем для SFace. Например, ~0.7-0.8 для высокой уверенности.
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
LOCAL_DOWNLOAD_PATH = "C:/projects/face/data/downloaded_images"  # Путь для сохранения скачанных изображений
OVERWRITE_LOCAL_FILES = False # Установите True, если хотите перезаписывать файлы при скачивании, даже если они уже есть локально 

DEEPFACE_CONCURRENCY_LIMIT = os.cpu_count() or 12 # Например, по числу ядер CPU или 4 по умолчанию 

# --- Настройки асинхронного пайплайна ---
# Максимальный размер очередей между этапами
QUEUE_MAX_SIZE = 100 
# Количество параллельных задач для скачивания
DOWNLOAD_WORKERS = 20 
# Количество параллельных задач для детекции лиц (обычно <= DEEPFACE_CONCURRENCY_LIMIT)
EXTRACTION_WORKERS = DEEPFACE_CONCURRENCY_LIMIT 
# Количество параллельных задач для извлечения эмбеддингов (обычно <= DEEPFACE_CONCURRENCY_LIMIT)
EMBEDDING_WORKERS = DEEPFACE_CONCURRENCY_LIMIT 
# Размер пакета для вставки в Milvus
MILVUS_INSERT_BATCH_SIZE = 128
# Таймаут ожидания новых данных для Milvus worker'а перед принудительной вставкой (в секундах)
MILVUS_INSERT_TIMEOUT = 5.0 