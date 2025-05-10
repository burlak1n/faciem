import asyncio
import os
import time
import shutil # Для очистки директорий
import sys # <--- Добавлено для настройки Loguru
from typing import List, Tuple, Dict, Any, Optional

import aiohttp
import cv2
import numpy as np
from pymilvus import Collection
from loguru import logger

# Предполагается, что эти модули находятся в той же директории или PYTHONPATH
try:
    import config
    from download_utils import download_image_async, get_local_path_for_url
    from image_processing import run_extraction_task_with_semaphore, run_represent_task_with_semaphore, init_deepface_model
    from db_utils import read_urls_from_file # Для примера в main
    from milvus_utils import init_milvus_connection # Для примера в main
    # Импорт get_dummy_embedding, если будем поддерживать маркеры отсутствия лиц также детально
    from pipeline_core import get_dummy_embedding, NO_FACE_MARKER_FACE_INDEX

except ImportError as e:
    logger.error(f"Ошибка импорта необходимых модулей: {e}. Убедитесь, что все зависимости доступны.")
    raise

# --- Конфигурация логирования --- 
logger.remove() # Удаляем стандартный обработчик, если он был добавлен где-то еще (например, в config)

# Добавляем обработчик для вывода в stderr (консоль)
logger.add(
    sys.stderr, 
    level=getattr(config, 'LOGURU_LEVEL', "INFO"), # Уровень для консоли берем из конфига или INFO по умолчанию
    format=getattr(config, 'LOGURU_FORMAT', "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"),
    colorize=True
)

# Добавляем обработчик для вывода в файл
log_file_path = os.path.join("logs", "sequential_pipeline_{time:YYYY-MM-DD}.log")
os.makedirs("logs", exist_ok=True) # Создаем директорию logs, если ее нет

logger.add(
    log_file_path,
    level="DEBUG",  # Уровень для файла (пишем все, начиная с DEBUG)
    format=getattr(config, 'LOGURU_FORMAT', "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"),
    rotation="100 MB", 
    retention="7 days", 
    compression="zip", 
    encoding="utf-8", 
    enqueue=True, 
    backtrace=True, 
    diagnose=True   
)

logger.info(f"Логирование для sequential_pipeline настроено. Уровень консоли: {getattr(config, 'LOGURU_LEVEL', 'INFO')}, Файл: {log_file_path} (уровень DEBUG)")

# --- Определение путей ---
# Путь для скачанных изображений берется из config
DOWNLOAD_BASE_PATH = getattr(config, 'LOCAL_DOWNLOAD_PATH', 'data/sequential_downloaded_images')
# Путь для извлеченных лиц. Желательно добавить в config.py как LOCAL_EXTRACTED_FACES_PATH
EXTRACTED_FACES_BASE_PATH = getattr(config, 'LOCAL_EXTRACTED_FACES_PATH', os.path.join(os.path.dirname(DOWNLOAD_BASE_PATH), "sequential_extracted_faces"))


async def download_all_images_locally(
    image_data_tuples: List[Tuple[Any, str, Optional[str]]],
    download_path_base: str,
    session: aiohttp.ClientSession,
    batch_size: int = 0 # 0 или отсутствие будет означать использование значения из config
) -> List[Dict[str, Any]]:
    """
    Этап 1: Скачивает все изображения пачками и сохраняет их локально.
    Возвращает список словарей с информацией о скачанных файлах.
    """
    logger.info(f"--- ЭТАП 1: Скачивание {len(image_data_tuples)} изображений в {download_path_base} ---")
    os.makedirs(download_path_base, exist_ok=True)

    if batch_size <= 0:
        # Попробуем взять из конфига, если не задан явно или некорректен
        # Предположим, что в config.py есть DOWNLOAD_WORKERS или DOWNLOAD_BATCH_SIZE
        # Используем DOWNLOAD_WORKERS как примерный ориентир для размера пачки одновременных задач
        effective_batch_size = getattr(config, 'DOWNLOAD_WORKERS', 20) 
        # Или можно ввести новый параметр типа config.SEQUENTIAL_DOWNLOAD_BATCH_SIZE
        logger.info(f"Размер пачки для скачивания не указан явно, используется значение: {effective_batch_size}")
    else:
        effective_batch_size = batch_size
        logger.info(f"Размер пачки для скачивания: {effective_batch_size}")
    
    downloaded_image_records = []
    total_images = len(image_data_tuples)

    for i in range(0, total_images, effective_batch_size):
        batch_tuples = image_data_tuples[i:i + effective_batch_size]
        logger.info(f"Обработка пачки изображений {i+1}-{min(i + effective_batch_size, total_images)} из {total_images}...")
        
        download_tasks = []
        # Временный список для записей, которые уже существуют локально в текущей пачке
        skipped_in_batch = [] 

        for photo_id, url, photo_date in batch_tuples:
            local_file_path = get_local_path_for_url(photo_id, url, download_path_base)
            overwrite = getattr(config, 'OVERWRITE_LOCAL_FILES', False)
            if os.path.exists(local_file_path) and not overwrite:
                logger.debug(f"Файл {local_file_path} для ID {photo_id} уже существует, пропуск скачивания.")
                # Сразу добавляем в skipped_in_batch, чтобы не создавать задачу
                skipped_in_batch.append({
                    "photo_id": photo_id, "original_url": url, "photo_date": photo_date,
                    "local_path": local_file_path, "status": "skipped_existing", "error": None
                })
                continue
            download_tasks.append(download_image_direct(session, photo_id, url, photo_date, local_file_path))

        # Добавляем пропущенные записи из текущей пачки в общий список результатов
        downloaded_image_records.extend(skipped_in_batch)

        if download_tasks: # Если в пачке есть что скачивать
            results_from_gather = await asyncio.gather(*download_tasks, return_exceptions=True)
            for result in results_from_gather:
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при скачивании в пакете: {result}")
                    # В реальном сценарии, нужно было бы создать запись об ошибке с photo_id и т.д.
                    # Здесь мы просто логируем, т.к. download_image_direct сам формирует запись об ошибке.
                    # Эта ветка сработает, если gather сам по себе выбросит исключение, что маловероятно
                    # для return_exceptions=True, но для полноты.
                elif result: # download_image_direct возвращает dict или None (хотя сейчас всегда dict)
                    downloaded_image_records.append(result)
        logger.info(f"Завершена обработка пачки. Всего обработано записей: {len(downloaded_image_records)}.")
            
    successful_downloads = sum(1 for r in downloaded_image_records if r['status'] == 'success' or r['status'] == 'skipped_existing')
    logger.info(f"Завершено скачивание. Успешно/пропущено: {successful_downloads} из {total_images}")
    return downloaded_image_records

async def download_image_direct(session: aiohttp.ClientSession, photo_id: Any, url: str, photo_date: Optional[str], local_path: str) -> Optional[Dict[str, Any]]:
    """Вспомогательная функция для скачивания и сохранения одного изображения."""
    try:
        async with session.get(url, timeout=getattr(config, 'DOWNLOAD_TIMEOUT', 10)) as response:
            response.raise_for_status()
            image_data = await response.read()
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(image_data)
            logger.debug(f"Изображение для ID {photo_id} успешно скачано и сохранено в {local_path}")
            return {
                "photo_id": photo_id, "original_url": url, "photo_date": photo_date,
                "local_path": local_path, "status": "success", "error": None
            }
    except Exception as e:
        logger.error(f"Ошибка скачивания {url} (ID: {photo_id}): {e}")
        return {
            "photo_id": photo_id, "original_url": url, "photo_date": photo_date,
            "local_path": local_path, "status": "failed", "error": str(e)
        }

async def extract_all_faces_from_local_images(
    downloaded_image_records: List[Dict[str, Any]],
    extracted_faces_path_base: str,
    deepface_semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop
) -> List[Dict[str, Any]]:
    """
    Этап 2: Обрабатывает локальные изображения, извлекает лица и сохраняет их.
    """
    logger.info(f"--- ЭТАП 2: Извлечение лиц из {len(downloaded_image_records)} изображений в {extracted_faces_path_base} ---")
    os.makedirs(extracted_faces_path_base, exist_ok=True)
    
    extracted_face_records = []
    extraction_tasks = []

    for record in downloaded_image_records:
        if record['status'] != 'success' and record['status'] != 'skipped_existing':
            logger.warning(f"Пропуск извлечения лиц для ID {record['photo_id']} из-за ошибки скачивания: {record.get('error', 'N/A')}")
            continue
        
        extraction_tasks.append(
            process_single_image_for_faces(
                record, extracted_faces_path_base, deepface_semaphore, loop
            )
        )

    results_list_of_lists = await asyncio.gather(*extraction_tasks, return_exceptions=True)

    for result_item in results_list_of_lists:
        if isinstance(result_item, Exception):
            logger.error(f"Ошибка в задаче извлечения лиц: {result_item}")
        elif isinstance(result_item, list): # Ожидаем список (может быть пустым для no_face_marker)
            extracted_face_records.extend(result_item)
        elif result_item is not None: # Если не список, но и не None/Exception, это неожиданно
             logger.warning(f"Неожиданный результат от process_single_image_for_faces: {result_item}")


    total_faces_extracted_count = sum(1 for r in extracted_face_records if not r.get('is_no_face_marker'))
    total_no_face_markers = sum(1 for r in extracted_face_records if r.get('is_no_face_marker'))
    logger.info(f"Завершено извлечение лиц. Извлечено реальных лиц: {total_faces_extracted_count}. Маркеров 'нет лиц': {total_no_face_markers}.")
    return extracted_face_records

async def process_single_image_for_faces(
    image_record: Dict[str, Any],
    extracted_faces_path_base: str,
    deepface_semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop
) -> List[Dict[str, Any]]:
    """Обрабатывает одно изображение, извлекает и сохраняет лица."""
    photo_id = image_record['photo_id']
    local_path = image_record['local_path']
    results_for_image = []
    
    try:
        image_np = cv2.imread(local_path)
        if image_np is None:
            logger.error(f"Не удалось прочитать локальный файл: {local_path} для ID {photo_id}")
            # Создаем маркер "ошибка чтения"
            results_for_image.append({
                "photo_id": photo_id, "original_url": image_record['original_url'], "photo_date": image_record['photo_date'],
                "original_local_path": local_path, "is_no_face_marker": True, "marker_type": "read_error", "error_message": "Failed to read image"
            })
            return results_for_image

        # Используем run_extraction_task_with_semaphore, он возвращает кортеж
        # (photo_id, url, image_base64, photo_date, extraction_result_or_exc)
        # Нам нужно адаптировать параметры или обернуть вызов
        _ph_id, _url, _img_b64, _ph_date, extraction_result_or_exc = await run_extraction_task_with_semaphore(
            loop, deepface_semaphore,
            photo_id_param=photo_id,
            url_param=image_record['original_url'], # передаем оригинальный url для консистентности
            image_np_param=image_np,
            image_base64_cb_param=None, # base64 не нужен здесь
            photo_date_param=image_record['photo_date']
        )

        if isinstance(extraction_result_or_exc, Exception):
            logger.error(f"Ошибка извлечения лиц для ID {photo_id} из {local_path}: {extraction_result_or_exc}")
            results_for_image.append({
                "photo_id": photo_id, "original_url": image_record['original_url'], "photo_date": image_record['photo_date'],
                "original_local_path": local_path, "is_no_face_marker": True, "marker_type": "extraction_error", "error_message": str(extraction_result_or_exc)
            })
            return results_for_image

        detected_faces_list = extraction_result_or_exc
        faces_kept_count = 0
        if detected_faces_list:
            for face_idx, face_data in enumerate(detected_faces_list):
                confidence = face_data.get('confidence', 0)
                if confidence >= getattr(config, 'MIN_DET_SCORE', 0.98):
                    face_np_data = face_data['face']
                    # Сохраняем вырезанное лицо
                    extracted_face_filename = f"{photo_id}_face{face_idx}_conf{confidence:.2f}.jpg"
                    extracted_face_local_path = os.path.join(extracted_faces_path_base, extracted_face_filename)
                    
                    try:
                        cv2.imwrite(extracted_face_local_path, face_np_data)
                        results_for_image.append({
                            "photo_id": photo_id, "original_url": image_record['original_url'],
                            "photo_date": image_record['photo_date'], "original_local_path": local_path,
                            "face_idx": face_idx, "extracted_face_local_path": extracted_face_local_path,
                            "confidence": confidence, "is_no_face_marker": False
                        })
                        faces_kept_count +=1
                    except Exception as e_write:
                        logger.error(f"Ошибка сохранения извлеченного лица {extracted_face_local_path}: {e_write}")
                else:
                    logger.debug(f"Лицо {face_idx} на ID {photo_id} пропущено (confidence: {confidence:.2f})")
        
        if faces_kept_count == 0 and not isinstance(extraction_result_or_exc, Exception): # Если лиц не было или ни одно не прошло порог
             logger.info(f"Подходящие лица не найдены/не извлечены для ID {photo_id} из {local_path}.")
             results_for_image.append({
                "photo_id": photo_id, "original_url": image_record['original_url'], "photo_date": image_record['photo_date'],
                "original_local_path": local_path, "is_no_face_marker": True, "marker_type": "no_faces_kept"
            })
        return results_for_image

    except Exception as e:
        logger.exception(f"Неперехваченная ошибка при обработке изображения {local_path} (ID: {photo_id}): {e}")
        results_for_image.append({
            "photo_id": photo_id, "original_url": image_record['original_url'], "photo_date": image_record['photo_date'],
            "original_local_path": local_path, "is_no_face_marker": True, "marker_type": "processing_exception", "error_message": str(e)
        })
        return results_for_image


async def embed_and_store_all_faces(
    extracted_face_records: List[Dict[str, Any]],
    milvus_collection: Collection,
    deepface_semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop
) -> List[Dict[str, Any]]:
    """
    Этап 3: Получает эмбеддинги для извлеченных лиц и сохраняет их в Milvus.
    """
    logger.info(f"--- ЭТАП 3: Получение эмбеддингов и сохранение в Milvus для {len(extracted_face_records)} записей ---")
    
    milvus_batch_data = []
    milvus_insertion_results = []
    
    # Получение фиктивного эмбеддинга для маркеров (если нужно)
    # dummy_embedding = await get_dummy_embedding(loop, deepface_semaphore)

    for record_idx, record in enumerate(extracted_face_records):
        photo_id = record['photo_id']

        if record.get('is_no_face_marker'):
            marker_type = record.get("marker_type", "unknown")
            logger.info(f"Обработка маркера 'нет лица' для ID {photo_id} (тип: {marker_type}).")
            # Если мы хотим сохранять маркеры в Milvus, нам нужен dummy_embedding
            # if dummy_embedding:
            #     milvus_batch_data.append({
            #         "photo_id": photo_id,
            #         "embedding": dummy_embedding,
            #         "face_index": NO_FACE_MARKER_FACE_INDEX,
            #     })
            #     milvus_insertion_results.append({
            #         "photo_id": photo_id, "face_idx": NO_FACE_MARKER_FACE_INDEX, "status": "marker_prepared", 
            #         "is_no_face_marker": True, "error": record.get("error_message")
            #     })
            # else:
            #     logger.error(f"Не удалось получить фиктивный эмбеддинг для маркера ID {photo_id}.")
            #     milvus_insertion_results.append({
            #         "photo_id": photo_id, "status": "marker_failed_dummy_emb", "is_no_face_marker": True, 
            #         "error": f"Failed to get dummy embedding. Original error: {record.get('error_message')}"
            #     })
            # Пока просто логируем и пропускаем добавление в Milvus для маркеров в этом упрощенном пайплайне
            milvus_insertion_results.append({
                    "photo_id": photo_id, "status": "marker_skipped", "is_no_face_marker": True, 
                    "error": record.get("error_message"), "marker_type": marker_type
                })
            continue

        face_idx = record['face_idx']
        extracted_face_local_path = record['extracted_face_local_path']
        
        try:
            face_np = cv2.imread(extracted_face_local_path)
            if face_np is None:
                logger.error(f"Не удалось прочитать извлеченное лицо: {extracted_face_local_path} для ID {photo_id}, FaceIdx {face_idx}")
                milvus_insertion_results.append({"photo_id": photo_id, "face_idx": face_idx, "status": "failed_read_face", "error": "Failed to read extracted face file", "is_no_face_marker": False})
                continue

            # Используем run_represent_task_with_semaphore
            # Возвращает Dict: {"status": "success" or "error", ..., "embedding_obj_list": ..., "error": ...}
            represent_result = await run_represent_task_with_semaphore(
                loop, deepface_semaphore,
                photo_id_meta_param=photo_id,
                url_meta_param=record['original_url'], # для консистентности
                face_idx_mapping_param=face_idx,
                face_np_param=face_np,
                photo_date_meta_param=record['photo_date']
            )

            if represent_result["status"] == "error":
                error_msg = str(represent_result.get("error", "Unknown represent error"))
                logger.error(f"Ошибка эмбеддинга для ID {photo_id}, FaceIdx {face_idx}: {error_msg}")
                milvus_insertion_results.append({"photo_id": photo_id, "face_idx": face_idx, "status": "failed_embedding", "error": error_msg, "is_no_face_marker": False})
                continue

            embedding_obj_list = represent_result.get("embedding_obj_list")
            if not embedding_obj_list or not isinstance(embedding_obj_list, list) or \
               len(embedding_obj_list) == 0 or not embedding_obj_list[0].get("embedding"):
                logger.error(f"Некорректный результат эмбеддинга для ID {photo_id}, FaceIdx {face_idx}")
                milvus_insertion_results.append({"photo_id": photo_id, "face_idx": face_idx, "status": "failed_embedding_nodata", "error": "No embedding data in result", "is_no_face_marker": False})
                continue
            
            embedding = embedding_obj_list[0]["embedding"]
            milvus_batch_data.append({
                "photo_id": photo_id,
                "embedding": embedding,
                "face_index": face_idx
            })
            milvus_insertion_results.append({"photo_id": photo_id, "face_idx": face_idx, "status": "embedding_success", "is_no_face_marker": False})

        except Exception as e:
            logger.exception(f"Неперехваченная ошибка при обработке эмбеддинга для ID {photo_id}, FaceIdx {face_idx}: {e}")
            milvus_insertion_results.append({"photo_id": photo_id, "face_idx": face_idx, "status": "failed_embedding_exception", "error": str(e), "is_no_face_marker": False})
            continue # к следующей записи
        
        # Вставка в Milvus пачками
        if len(milvus_batch_data) >= getattr(config, 'MILVUS_INSERT_BATCH_SIZE', 128) or \
           (record_idx == len(extracted_face_records) - 1 and milvus_batch_data): # последняя запись
            try:
                logger.info(f"Вставка {len(milvus_batch_data)} эмбеддингов в Milvus...")
                insert_start_time = time.monotonic()
                # Выполняем в executor, т.к. insert - блокирующая операция
                insert_result = await loop.run_in_executor(None, milvus_collection.insert, milvus_batch_data)
                insert_duration_ms = (time.monotonic() - insert_start_time) * 1000
                inserted_pks = insert_result.primary_keys
                logger.info(f"Успешно вставлено {len(inserted_pks)} эмбеддингов. Время: {insert_duration_ms:.2f} мс.")
                
                # Обновляем статус для вставленных записей
                # Это упрощенное обновление, в реальности нужно сопоставлять PK с исходными записями,
                # но для пакетной вставки Milvus обычно возвращает PK в том же порядке.
                # Здесь мы просто предполагаем, что все в батче успешно, если нет исключения.
                # Более точное отслеживание требует сопоставления ID.
                
                milvus_batch_data.clear()
            except Exception as e_milvus:
                logger.error(f"Ошибка Milvus при вставке батча: {e_milvus}")
                # Помечаем все записи в текущем батче как неудачные (упрощенно)
                for res_idx in range(len(milvus_insertion_results) - len(milvus_batch_data), len(milvus_insertion_results)):
                    if milvus_insertion_results[res_idx]['status'] == 'embedding_success': # Обновляем только те, что были готовы к вставке
                        milvus_insertion_results[res_idx]['status'] = 'failed_milvus_insert'
                        milvus_insertion_results[res_idx]['error'] = str(e_milvus)
                milvus_batch_data.clear() # Очищаем батч после ошибки

    # Финальный flush Milvus (если требуется)
    try:
        logger.info("Финальный Flush данных Milvus...")
        await loop.run_in_executor(None, milvus_collection.flush)
        logger.info("Данные Milvus успешно сброшены на диск.")
    except Exception as e_flush:
        logger.error(f"Ошибка при финальном вызове milvus_collection.flush(): {e_flush}")

    successful_inserts = sum(1 for r in milvus_insertion_results if r['status'] == 'embedding_success' or r['status'] == 'marker_prepared') # Предполагаем, что embedding_success означает успешную вставку (нужно улучшить)
    logger.info(f"Завершено получение эмбеддингов и сохранение. Успешно обработано для Milvus (приблизительно): {successful_inserts} из {len(extracted_face_records)} извлеченных лиц/маркеров.")
    return milvus_insertion_results


async def run_sequential_pipeline(
    image_data_tuples: List[Tuple[Any, str, Optional[str]]],
    clear_previous_run_data: bool = True,
    download_batch_s: Optional[int] = None
):
    """
    Основная функция для запуска всех этапов последовательного пайплайна.
    """
    logger.info("--- НАЧАЛО ЗАПУСКА ПОСЛЕДОВАТЕЛЬНОГО ПАЙПЛАЙНА (run_sequential_pipeline) ---")
    overall_start_time = time.monotonic()

    # Инициализация ресурсов
    loop = asyncio.get_event_loop()
    deepface_semaphore = asyncio.Semaphore(getattr(config, 'DEEPFACE_CONCURRENCY_LIMIT', 4))
    
    # Инициализация DeepFace модели (прогрев)
    init_deepface_model()

    # Инициализация Milvus
    try:
        milvus_collection = init_milvus_connection()
    except Exception as e:
        logger.critical(f"Не удалось инициализировать Milvus. Ошибка: {e}. Пайплайн прерван.")
        return

    # Очистка директорий от предыдущих запусков (если включено)
    if clear_previous_run_data:
        for path_to_clear in [DOWNLOAD_BASE_PATH, EXTRACTED_FACES_BASE_PATH]:
            if os.path.exists(path_to_clear):
                logger.info(f"Очистка директории: {path_to_clear}")
                try:
                    shutil.rmtree(path_to_clear)
                except Exception as e_clear:
                    logger.error(f"Не удалось очистить директорию {path_to_clear}: {e_clear}")
            os.makedirs(path_to_clear, exist_ok=True)
    else:
        os.makedirs(DOWNLOAD_BASE_PATH, exist_ok=True)
        os.makedirs(EXTRACTED_FACES_BASE_PATH, exist_ok=True)

    # Логика для возобновления
    actual_image_data_to_process = image_data_tuples
    pre_skipped_records = []
    start_index = 0

    # Проверяем необходимость возобновления только если не очищаем данные и не перезаписываем файлы
    should_overwrite = getattr(config, 'OVERWRITE_LOCAL_FILES', False)
    if not clear_previous_run_data and not should_overwrite:
        logger.info(f"Режим возобновления: clear_previous_run_data=False, OVERWRITE_LOCAL_FILES=False. Поиск точки возобновления...")
        for idx, (photo_id, url, photo_date) in enumerate(image_data_tuples):
            local_file_path = get_local_path_for_url(photo_id, url, DOWNLOAD_BASE_PATH)
            if not os.path.exists(local_file_path):
                logger.info(f"Точка возобновления найдена. Начинаем обработку с индекса {idx} (ID: {photo_id}, URL: {url}).")
                start_index = idx
                break # Нашли первый не скачанный файл
            else:
                # Файл существует, добавляем его в pre_skipped_records
                pre_skipped_records.append({
                    "photo_id": photo_id, "original_url": url, "photo_date": photo_date,
                    "local_path": local_file_path, "status": "skipped_existing_before_resume", "error": None
                })
                if idx == len(image_data_tuples) - 1: # Если все файлы уже существуют
                    logger.info("Все файлы из списка уже существуют локально. Скачивание не требуется.")
                    start_index = len(image_data_tuples) # Устанавливаем индекс за пределы списка
        
        actual_image_data_to_process = image_data_tuples[start_index:]
        if pre_skipped_records:
            logger.info(f"{len(pre_skipped_records)} файлов было пропущено до точки возобновления (уже существуют локально).")
    else:
        logger.info("Возобновление не активно (clear_previous_run_data=True или OVERWRITE_LOCAL_FILES=True). Обработка всех записей с начала.")

    # --- ЭТАП 1: Скачивание ---
    processed_download_records = []
    if actual_image_data_to_process: # Если есть что обрабатывать после определения точки возобновления
        async with aiohttp.ClientSession() as session:
            processed_download_records = await download_all_images_locally(
                actual_image_data_to_process, DOWNLOAD_BASE_PATH, session, batch_size=download_batch_s if download_batch_s is not None else 0
            )
    else:
        logger.info("Нет новых изображений для скачивания на ЭТАПЕ 1.")
    
    # Объединяем предварительно пропущенные записи с результатами текущего скачивания
    downloaded_records = pre_skipped_records + processed_download_records
    
    # Фильтруем только успешно скачанные или ранее существовавшие для следующего этапа
    successful_downloads_for_extraction = [
        r for r in downloaded_records if r['status'] == 'success' or r['status'] == 'skipped_existing' or r['status'] == 'skipped_existing_before_resume'
    ]
    if not successful_downloads_for_extraction:
        logger.warning("Нет успешно скачанных изображений для продолжения пайплайна.")
        return

    # --- ЭТАП 2: Извлечение лиц ---
    extracted_records = await extract_all_faces_from_local_images(
        successful_downloads_for_extraction, EXTRACTED_FACES_BASE_PATH, deepface_semaphore, loop
    )

    # Фильтруем только успешно извлеченные лица (не маркеры ошибок) для следующего этапа
    # или все записи, если хотим обрабатывать маркеры в embed_and_store_all_faces
    records_for_embedding = [
        r for r in extracted_records # if not r.get('is_no_face_marker') or r.get('marker_type') == 'no_faces_kept' # Пример фильтрации
    ]
    if not records_for_embedding:
        logger.warning("Нет извлеченных лиц или маркеров для передачи на этап эмбеддинга.")
        return

    # --- ЭТАП 3: Эмбеддинг и сохранение в Milvus ---
    final_results = await embed_and_store_all_faces(
        records_for_embedding, milvus_collection, deepface_semaphore, loop
    )

    overall_duration = time.monotonic() - overall_start_time
    logger.info(f"--- ПОСЛЕДОВАТЕЛЬНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН ---")
    logger.info(f"Общее время выполнения: {overall_duration:.2f} сек.")
    # Здесь можно вывести более детальную статистику по final_results
    
    # Пример детальной статистики
    stats = {"total_input": len(image_data_tuples), "downloaded_success_skipped": 0, "download_failed":0,
             "faces_extracted_real":0, "markers_created_extraction":0, "embeddings_milvus_success":0, "embeddings_milvus_failed_or_skipped_marker":0}

    for dr in downloaded_records:
        if dr['status'] == 'success' or dr['status'] == 'skipped_existing' or dr['status'] == 'skipped_existing_before_resume':
            stats['downloaded_success_skipped'] += 1
        else:
            stats['download_failed'] +=1
    
    for er in extracted_records:
        if not er.get('is_no_face_marker'):
            stats['faces_extracted_real'] +=1
        else:
            stats['markers_created_extraction'] +=1
            
    for fr in final_results:
        # Уточнить логику подсчета успешных вставок в Milvus
        if fr['status'] == 'embedding_success' and not fr.get('is_no_face_marker'): # Если это была реальная вставка
             stats['embeddings_milvus_success'] +=1
        else: # Ошибки или маркеры, которые не пошли в Milvus
             stats['embeddings_milvus_failed_or_skipped_marker'] +=1
             
    logger.info(f"Статистика: {stats}")


async def example_main():
    """Пример запуска пайплайна."""
    # Получение данных для обработки (например, из SQLite)
    source_type = "sqlite" if config.URL_FILE_PATH.endswith('.db') else "txt"
    image_data_gen = read_urls_from_file(
        input_path=config.URL_FILE_PATH, 
        source_type=source_type
    )
    image_data_list = list(image_data_gen)
    
    # Возьмем только первые N для теста, если нужно
    # image_data_list = image_data_list[:20] 

    if not image_data_list:
        logger.error("Нет данных для обработки. Проверьте URL_FILE_PATH в config.py")
        return

    await run_sequential_pipeline(image_data_list, clear_previous_run_data=False, download_batch_s=50) # Пример: размер пачки 50

if __name__ == "__main__":
    # Перед запуском убедитесь, что LOCAL_DOWNLOAD_PATH и (опционально) LOCAL_EXTRACTED_FACES_PATH
    # существуют или могут быть созданы, и что Milvus доступен.
    
    # Убедимся, что базовые директории существуют, если clear_previous_run_data=False
    if not getattr(config, 'clear_previous_run_data_on_startup', True): # Пример новой переменной в config
        os.makedirs(DOWNLOAD_BASE_PATH, exist_ok=True)
        os.makedirs(EXTRACTED_FACES_BASE_PATH, exist_ok=True)
        
    asyncio.run(example_main()) 