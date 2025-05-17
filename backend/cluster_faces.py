from loguru import logger
import argparse
import sys

# Импортируем только FaceManager
from face_manager import FaceManager

# --- Константы (Параметры по умолчанию для argparse) ---
DEFAULT_CLUSTER_EPS = 0.35
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_HDBSCAN_MIN_SAMPLES = 3
LOG_FILE = "cluster_faces.log"

# --- Настройка Логгера (такая же, как была) ---
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE, rotation="5 MB", level="DEBUG")

# --- Удалены все вспомогательные функции и run_clustering_and_update --- 
# --- Вся логика теперь внутри FaceManager.cluster_faces() --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кластеризация лиц в Milvus с помощью FaceManager.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "incremental"],
        default="full", 
        help="Режим кластеризации: 'full' (перекластеризовать все), 'incremental' (только новые)."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_CLUSTER_EPS, 
        help=f"Параметр eps для инкрементального присвоения к существующим кластерам. По умолчанию: {DEFAULT_CLUSTER_EPS}"
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=f"Параметр min_cluster_size для HDBSCAN (минимальный размер кластера). По умолчанию: {DEFAULT_MIN_CLUSTER_SIZE}"
    )
    parser.add_argument(
        "--hdbscan_min_samples",
        type=int,
        default=DEFAULT_HDBSCAN_MIN_SAMPLES,
        help=f"Параметр min_samples для HDBSCAN (влияет на обработку шума, обычно равен min_cluster_size если None). По умолчанию: {DEFAULT_HDBSCAN_MIN_SAMPLES}"
    )
    # Можно добавить параметры для модели/детектора, если нужно переопределить
    # parser.add_argument("--model", type=str, help="Модель DeepFace (если отличается от FaceManager по умолчанию)")
    # parser.add_argument("--detector", type=str, help="Детектор DeepFace (если отличается от FaceManager по умолчанию)")
    
    args = parser.parse_args()
    
    logger.info("Инициализация FaceManager для кластеризации...")
    # Создаем экземпляр менеджера. Можно передать model/detector из args, если они есть.
    # manager_kwargs = {}
    # if args.model: manager_kwargs['model_name'] = args.model
    # if args.detector: manager_kwargs['detector_backend'] = args.detector
    # face_manager = FaceManager(**manager_kwargs)
    face_manager = FaceManager()

    # Проверяем, успешно ли инициализирован менеджер
    if not face_manager.client:
         logger.error("Не удалось инициализировать FaceManager (проблема с Milvus). Завершение работы.")
         sys.exit(1)
         
    # Запускаем кластеризацию через метод менеджера
    face_manager.cluster_faces(args.mode, args.eps, args.min_cluster_size, args.hdbscan_min_samples)

    logger.info(f"Скрипт cluster_faces.py (режим {args.mode}) завершил работу.") 