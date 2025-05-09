import cv2
import numpy as np
# import requests # Будет заменен на aiohttp для асинхронной загрузки
import asyncio
import aiohttp
from deepface import DeepFace # <--- Изменено: импорт DeepFace
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema
import sys # Для настройки Loguru
import sqlite3 # <--- Добавлено для SQLite
# import argparse # Удален
import base64 # <--- Добавлено для кодирования изображений
# import io # Удален
from typing import Callable, Optional, Any, List, Tuple, Dict # <--- Добавлено Dict
import functools # <--- Добавлено для partial
import urllib.parse # <--- Добавлено для работы с URL
import time # <--- Добавлено для измерения времени
import datetime # <--- Добавлено для работы с датой и временем
import pytz # <--- Добавлено для работы с часовыми поясами
import os # <--- Добавлено для работы с файловой системой
import tensorflow as tf # <--- Добавлено для проверки GPU

# --- Импорт Loguru --- 
from loguru import logger

# --- Импорт функций из новых модулей ---
from download_utils import get_local_path_for_url, get_max_quality_url, download_image_async
from db_utils import read_urls_from_file, get_urls_for_photo_ids
from milvus_utils import init_milvus_connection, search_similar_faces_in_milvus, search_similar_faces_in_milvus_by_bytes
from image_processing import check_tensorflow_gpu, init_deepface_model, run_extraction_task_with_semaphore, run_represent_task_with_semaphore
from pipeline_core import process_and_store_faces_async # <--- Импортируем основную функцию пайплайна
# TODO: Добавить импорты из других утилитных модулей по мере их создания

# --- Импорт конфигурации ---
# Оставляем относительный импорт для случаев, когда этот модуль используется как часть пакета
# Если web_server.py импортирует его напрямую из корня, то такой импорт может вызвать проблемы.
# Для консистентности, можно использовать from . import config, если запускать как модуль,
# или import config, если PYTHONPATH настроен на корень проекта.
# При запуске через uvicorn web_server:app из корня, import config должен работать.
try:
    from . import config # Если milvus_vk_pipeline - часть пакета
except ImportError:
    import config # Если запускается как отдельный скрипт или импортируется из web_server, находящегося в корне

# --- Настройка Loguru --- 
logger.remove() # Удаляем стандартный обработчик

# Добавляем обработчик для вывода в stderr (консоль)
logger.add(
    sys.stderr, 
    level=config.LOGURU_LEVEL, # Уровень для консоли берем из конфига (INFO, DEBUG, etc.)
    format=config.LOGURU_FORMAT,
    colorize=True # Включаем цвета для консоли
)

# Добавляем обработчик для вывода в файл
log_file_path = "logs/pipeline_{time:YYYY-MM-DD}.log" # Имя файла с датой
logger.add(
    log_file_path,
    level="DEBUG",  # Уровень для файла (пишем все, начиная с DEBUG)
    format=config.LOGURU_FORMAT, # Используем тот же формат, но можно и другой
    rotation="100 MB", # Ротация при достижении 100 MB
    retention="7 days", # Хранить логи за последние 7 дней
    compression="zip", # Сжимать старые логи в zip
    encoding="utf-8", # Явно указываем кодировку
    enqueue=True, # Асинхронная запись логов (рекомендуется для производительности)
    backtrace=True, # Расширенные трейсбеки
    diagnose=True   # Расширенная диагностика переменных при ошибках
)

logger.info(f"Логирование настроено. Уровень консоли: {config.LOGURU_LEVEL}, Уровень файла ({log_file_path}): DEBUG")

# --- Инициализация ---
def init_deepface_model():
    """Инициализирует (прогревает) модель DeepFace."""
    check_tensorflow_gpu() # <--- Добавлен вызов функции проверки GPU
    try:
        # Этот вызов загрузит модель и детектор, если они еще не загружены
        DeepFace.build_model(config.DEEPFACE_MODEL_NAME)
        # Также можно прогреть детектор, хотя он часто загружается с первой extract_faces
        # DeepFace.extract_faces(np.zeros((100,100,3), dtype=np.uint8), detector_backend=config.DEEPFACE_DETECTOR_BACKEND,-
        #                        enforce_detection=False) # enforce_detection=False для быстрого вызова
        logger.info(f"DeepFace модель '{config.DEEPFACE_MODEL_NAME}' и детектор '{config.DEEPFACE_DETECTOR_BACKEND}' готовы (или будут загружены при первом использовании).")
    except Exception as e:
        logger.error(f"Ошибка при инициализации/прогреве DeepFace: {e}")
        # Не прерываем выполнение, DeepFace может загрузить модели позже
        # raise # Раскомментировать, если критично иметь модели загруженными сразу

# --- Основная логика ---

# --- Точка входа (Пример) ---
async def main():
    logger.info("Запуск основного пайплайна...")
    
    # 1. Инициализация моделей и соединений
    # Вызываем функции инициализации из соответствующих модулей
    logger.info("Инициализация DeepFace модели...")
    init_deepface_model() # Из image_processing.py
    
    logger.info("Инициализация соединения с Milvus...")
    try:
        milvus_collection = init_milvus_connection() # Из milvus_utils.py
    except Exception as e:
        logger.critical(f"Не удалось инициализировать Milvus. Ошибка: {e}")
        return # Прерываем выполнение

    # 2. Получение данных для обработки
    # Пример: Чтение URL из файла SQLite
    # Вам нужно будет адаптировать это под ваш источник данных
    logger.info(f"Чтение URL из источника: {config.URL_FILE_PATH}")
    # Используем функцию из db_utils.py
    # Так как read_urls_from_file - генератор, преобразуем его в список для process_and_store_faces_async
    try:
        # Определяем тип источника на основе расширения файла или конфигурации
        source_type = "sqlite" if config.URL_FILE_PATH.endswith('.db') else "txt"
        image_data_generator = read_urls_from_file(
            input_path=config.URL_FILE_PATH, 
            source_type=source_type
            # Остальные параметры db_* будут по умолчанию
        )
        image_data_tuples = list(image_data_generator)
        if not image_data_tuples:
             logger.warning(f"Не найдено URL для обработки в источнике: {config.URL_FILE_PATH}")
             return
        logger.info(f"Найдено {len(image_data_tuples)} записей для обработки.")
    except Exception as e:
        logger.critical(f"Ошибка при чтении данных из источника {config.URL_FILE_PATH}: {e}")
        return

    # 3. Запуск основного пайплайна обработки
    logger.info("Запуск асинхронной обработки и сохранения лиц...")
    await process_and_store_faces_async(
        milvus_collection=milvus_collection,
        image_data_tuples=image_data_tuples,
        progress_callback=None, # Передайте ваш callback, если он есть
        skip_existing=True # Пример: пропускать существующие (или False)
    )

    logger.info("Основной пайплайн завершен.")

if __name__ == "__main__":
    # Запуск асинхронной функции main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Выполнение прервано пользователем.")
    except Exception as e_main:
        logger.critical(f"Критическая ошибка в главном потоке выполнения: {e_main}")
