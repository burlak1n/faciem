import asyncio
import datetime
import os
import time
import shutil
from typing import Callable, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from pymilvus import Collection
from loguru import logger
import sys # Для настройки логгера

# Предполагается, что эти модули находятся в той же директории или PYTHONPATH
try:
    import config
    from image_processing import run_extraction_task_with_semaphore, run_represent_task_with_semaphore, init_deepface_model
    # get_dummy_embedding и NO_FACE_MARKER_FACE_INDEX могут понадобиться, если мы решим детально обрабатывать маркеры
    # from pipeline_core import get_dummy_embedding, NO_FACE_MARKER_FACE_INDEX 
    from milvus_utils import init_milvus_connection # Для примера в main
    from db_utils import read_urls_from_file # Для примера в main (хотя этот пайплайн не читает URL)
except ImportError as e:
    logger.error(f"Ошибка импорта необходимых модулей в local_album_pipeline: {e}.")
    raise

# --- Конфигурация логирования (аналогично sequential_pipeline.py) ---
logger.remove()
logger.add(
    sys.stderr, 
    level=getattr(config, 'LOGURU_LEVEL', "INFO"),
    format=getattr(config, 'LOGURU_FORMAT', "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"),
    colorize=True
)
log_file_path_lap = os.path.join("logs", "local_album_pipeline_{time:YYYY-MM-DD}.log")
os.makedirs("logs", exist_ok=True)
logger.add(
    log_file_path_lap,
    level="DEBUG",
    format=getattr(config, 'LOGURU_FORMAT', "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"),
    rotation="50 MB", retention="5 days", compression="zip", encoding="utf-8", 
    enqueue=True, backtrace=True, diagnose=True   
)
logger.info(f"Логирование для local_album_pipeline настроено. Файл: {log_file_path_lap}")

# --- Определение путей (можно использовать те же, что и в sequential_pipeline или новые из config) ---
EXTRACTED_FACES_BASE_PATH_LAP = getattr(config, 'LOCAL_EXTRACTED_FACES_PATH', 
                                        os.path.join(getattr(config, 'LOCAL_DOWNLOAD_PATH', 'data/downloaded_images'), "../local_album_extracted_faces"))

async def extract_faces_from_local_files(
    image_data_tuples: List[Tuple[Any, str, Optional[str]]], # (photo_id, local_file_path, photo_date)
    extracted_faces_path_base: str,
    deepface_semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    progress_callback: Optional[Callable] = None # Добавлен progress_callback
) -> List[Dict[str, Any]]:
    """
    Этап 1 (для локального пайплайна): Обрабатывает локальные изображения, извлекает лица и сохраняет их.
    """
    logger.info(f"--- LOCAL ALBUM PIPELINE - ЭТАП 1: Извлечение лиц из {len(image_data_tuples)} локальных файлов в {extracted_faces_path_base} ---")
    os.makedirs(extracted_faces_path_base, exist_ok=True)
    
    extracted_face_records = []
    extraction_tasks = []
    total_files = len(image_data_tuples)

    for idx, (photo_id, local_file_path, photo_date) in enumerate(image_data_tuples):
        # photo_id здесь уже должен быть строкой из get_image_data_by_album_from_sqlite
        extraction_tasks.append(
            process_single_local_image_for_faces(
                photo_id, local_file_path, photo_date, 
                extracted_faces_path_base, deepface_semaphore, loop, progress_callback, idx, total_files
            )
        )

    results_list_of_lists = await asyncio.gather(*extraction_tasks, return_exceptions=True)

    for result_item in results_list_of_lists:
        if isinstance(result_item, Exception):
            logger.error(f"Ошибка в задаче извлечения лиц (local_album_pipeline): {result_item}")
            # Здесь можно решить, нужно ли создавать запись об ошибке для статистики
        elif isinstance(result_item, list):
            extracted_face_records.extend(result_item)
        elif result_item is not None:
             logger.warning(f"Неожиданный результат от process_single_local_image_for_faces: {result_item}")

    total_faces_extracted_count = sum(1 for r in extracted_face_records if not r.get('is_no_face_marker'))
    total_no_face_markers = sum(1 for r in extracted_face_records if r.get('is_no_face_marker'))
    logger.info(f"LOCAL ALBUM PIPELINE - Завершено извлечение лиц. Реальных лиц: {total_faces_extracted_count}. Маркеров: {total_no_face_markers}.")
    return extracted_face_records

async def process_single_local_image_for_faces(
    photo_id: Any,
    local_file_path: str,
    photo_date: Optional[str],
    extracted_faces_path_base: str,
    deepface_semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    progress_callback: Optional[Callable],
    current_idx: int,
    total_count: int
) -> List[Dict[str, Any]]:
    """Обрабатывает одно локальное изображение, извлекает и сохраняет лица."""
    results_for_image = []
    base_image_for_callback = None # Для передачи миниатюры, если потребуется
    
    try:
        if progress_callback: # Начало обработки файла
            await progress_callback(photo_id=photo_id, url=local_file_path, image_base64=None, status="processing_local_start", 
                                    processed_count=current_idx, total_count=total_count, photo_date=photo_date)

        image_np = cv2.imread(local_file_path)
        if image_np is None:
            logger.error(f"Не удалось прочитать локальный файл: {local_file_path} для ID {photo_id}")
            results_for_image.append({
                "photo_id": photo_id, "original_local_path": local_file_path, "photo_date": photo_date,
                "is_no_face_marker": True, "marker_type": "read_error", "error_message": "Failed to read image"
            })
            if progress_callback:
                await progress_callback(photo_id=photo_id, url=local_file_path, status="error_read_local", error_message="Failed to read image", photo_date=photo_date)
            return results_for_image

        # Адаптируем вызов run_extraction_task_with_semaphore
        # url_param может быть local_file_path для идентификации в логах/коллбэках
        _ph_id, _url_cb, _img_b64_cb, _ph_date_cb, extraction_result_or_exc = await run_extraction_task_with_semaphore(
            loop, deepface_semaphore,
            photo_id_param=photo_id,
            url_param=local_file_path, 
            image_np_param=image_np,
            image_base64_cb_param=None, # Пока не генерируем base64 здесь для экономии
            photo_date_param=photo_date
        )
        faces_found_on_image_count = 0
        face_confidences = []
        error_on_extraction_msg = None

        if isinstance(extraction_result_or_exc, Exception):
            error_on_extraction_msg = str(extraction_result_or_exc)
            logger.error(f"Ошибка извлечения лиц для ID {photo_id} из {local_file_path}: {error_on_extraction_msg}")
            results_for_image.append({
                "photo_id": photo_id, "original_local_path": local_file_path, "photo_date": photo_date,
                "is_no_face_marker": True, "marker_type": "extraction_error", "error_message": error_on_extraction_msg
            })
        else:
            detected_faces_list = extraction_result_or_exc
            faces_kept_count = 0
            if detected_faces_list:
                faces_found_on_image_count = len(detected_faces_list)
                for face_idx, face_data in enumerate(detected_faces_list):
                    confidence = face_data.get('confidence', 0)
                    if confidence >= getattr(config, 'MIN_DET_SCORE', 0.98):
                        face_np_data = face_data['face']
                        face_confidences.append(round(confidence, 4))
                        extracted_face_filename = f"{photo_id}_face{face_idx}_conf{confidence:.2f}.jpg"
                        extracted_face_local_path = os.path.join(extracted_faces_path_base, extracted_face_filename)
                        try:
                            cv2.imwrite(extracted_face_local_path, face_np_data)
                            results_for_image.append({
                                "photo_id": photo_id, "original_local_path": local_file_path, "photo_date": photo_date,
                                "face_idx": face_idx, "extracted_face_local_path": extracted_face_local_path,
                                "confidence": confidence, "is_no_face_marker": False
                            })
                            faces_kept_count +=1
                        except Exception as e_write:
                            logger.error(f"Ошибка сохранения извлеченного лица {extracted_face_local_path}: {e_write}")
                    else:
                        logger.debug(f"Лицо {face_idx} на ID {photo_id} (из {local_file_path}) пропущено (confidence: {confidence:.2f})")    
            
            if faces_kept_count == 0: # Если лиц не было или ни одно не прошло порог
                 logger.info(f"Подходящие лица не найдены/не извлечены для ID {photo_id} из {local_file_path}.")
                 results_for_image.append({
                    "photo_id": photo_id, "original_local_path": local_file_path, "photo_date": photo_date,
                    "is_no_face_marker": True, "marker_type": "no_faces_kept"
                })
        
        if progress_callback:
            status_cb = "extraction_completed_local"
            if error_on_extraction_msg:
                status_cb = "error_extraction_local"
            elif faces_kept_count == 0 and not error_on_extraction_msg:
                status_cb = "extraction_no_faces_kept_local"
            await progress_callback(
                status=status_cb, photo_id=photo_id, url=local_file_path, 
                faces_count=faces_found_on_image_count, face_confidences=face_confidences if face_confidences else None,
                error_message=error_on_extraction_msg, photo_date=photo_date,
                processed_count=current_idx + 1, total_count=total_count
            )
        return results_for_image

    except Exception as e:
        logger.exception(f"Неперехваченная ошибка при обработке файла {local_file_path} (ID: {photo_id}): {e}")
        results_for_image.append({
            "photo_id": photo_id, "original_local_path": local_file_path, "photo_date": photo_date,
            "is_no_face_marker": True, "marker_type": "processing_exception", "error_message": str(e)
        })
        if progress_callback:
             await progress_callback(status="error_processing_local", photo_id=photo_id, url=local_file_path, error_message=str(e), photo_date=photo_date,
                                   processed_count=current_idx + 1, total_count=total_count)
        return results_for_image

async def embed_and_store_extracted_faces(
    extracted_face_records: List[Dict[str, Any]],
    milvus_collection: Collection,
    deepface_semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    progress_callback: Optional[Callable] = None,
    offset_count: int = 0, # для корректного processed_count в callback
    total_count_overall: int = 0 # для корректного total_count в callback
) -> Tuple[int, Dict[str, int]]: # inserted_count, error_counts
    """
    Этап 2 (для локального пайплайна): Получает эмбеддинги для извлеченных лиц и сохраняет их в Milvus.
    """
    logger.info(f"--- LOCAL ALBUM PIPELINE - ЭТАП 2: Эмбеддинг и сохранение в Milvus для {len(extracted_face_records)} записей ---")
    
    milvus_batch_data = []
    # milvus_insertion_results = [] # Заменим на счетчики
    successfully_inserted_count = 0
    embedding_errors = 0
    milvus_insert_errors = 0
    skipped_markers_count = 0

    # dummy_embedding = await get_dummy_embedding(loop, deepface_semaphore) # Если будем сохранять маркеры

    for record_idx, record in enumerate(extracted_face_records):
        photo_id = record['photo_id']
        actual_processed_count = offset_count + record_idx + 1

        if record.get('is_no_face_marker'):
            marker_type = record.get("marker_type", "unknown_marker")
            logger.info(f"Пропуск маркера '{marker_type}' для ID {photo_id} на этапе эмбеддинга.")
            skipped_markers_count +=1
            # Не отправляем индивидуальный callback для пропущенного маркера здесь,
            # т.к. он уже был обработан на этапе extraction.
            # Финальная статистика учтет это.
            continue

        face_idx = record['face_idx']
        extracted_face_local_path = record['extracted_face_local_path']
        
        if progress_callback: 
            await progress_callback(photo_id=photo_id, url=record.get('original_local_path'), 
                                    status="embedding_start_local", face_index_processed=face_idx, 
                                    processed_count=actual_processed_count, total_count=total_count_overall, 
                                    photo_date=record.get('photo_date'))
        try:
            face_np = cv2.imread(extracted_face_local_path)
            if face_np is None:
                logger.error(f"Не удалось прочитать извлеченное лицо: {extracted_face_local_path} (ID {photo_id}, FaceIdx {face_idx})")
                embedding_errors +=1
                if progress_callback: await progress_callback(photo_id=photo_id, url=record.get('original_local_path'), status="error_embedding_local", error_message="Failed read extracted face", face_index_processed=face_idx, photo_date=record.get('photo_date'))
                continue

            represent_result = await run_represent_task_with_semaphore(
                loop, deepface_semaphore,
                photo_id_meta_param=photo_id,
                url_meta_param=record.get('original_local_path', extracted_face_local_path),
                face_idx_mapping_param=face_idx,
                face_np_param=face_np,
                photo_date_meta_param=record.get('photo_date')
            )

            embedding_duration_ms = represent_result.get("duration_ms", -1.0)
            if represent_result["status"] == "error":
                error_msg = str(represent_result.get("error", "Unknown represent error"))
                logger.error(f"Ошибка эмбеддинга для ID {photo_id}, FaceIdx {face_idx}: {error_msg}")
                embedding_errors += 1
                if progress_callback: await progress_callback(photo_id=photo_id, url=record.get('original_local_path'), status="error_embedding_local", error_message=error_msg, face_index_processed=face_idx, embedding_duration_ms=embedding_duration_ms, photo_date=record.get('photo_date'))
                continue

            embedding_obj_list = represent_result.get("embedding_obj_list")
            if not embedding_obj_list or not isinstance(embedding_obj_list, list) or \
               len(embedding_obj_list) == 0 or not embedding_obj_list[0].get("embedding"):
                logger.error(f"Некорректный результат эмбеддинга для ID {photo_id}, FaceIdx {face_idx}")
                embedding_errors += 1
                if progress_callback: await progress_callback(photo_id=photo_id, url=record.get('original_local_path'), status="error_embedding_local", error_message="No embedding data in result", face_index_processed=face_idx, embedding_duration_ms=embedding_duration_ms, photo_date=record.get('photo_date'))
                continue
            
            embedding = embedding_obj_list[0]["embedding"]
            milvus_batch_data.append({"photo_id": photo_id, "embedding": embedding, "face_index": face_idx})
            if progress_callback: await progress_callback(photo_id=photo_id, url=record.get('original_local_path'), status="embedding_extracted_local_face", face_index_processed=face_idx, embedding_duration_ms=embedding_duration_ms, photo_date=record.get('photo_date'))

        except Exception as e:
            logger.exception(f"Неперехваченная ошибка (ID {photo_id}, FaceIdx {face_idx}): {e}")
            embedding_errors += 1
            if progress_callback: await progress_callback(photo_id=photo_id, url=record.get('original_local_path'), status="error_embedding_local", error_message=str(e), face_index_processed=face_idx, photo_date=record.get('photo_date'))
            continue
        
        if len(milvus_batch_data) >= getattr(config, 'MILVUS_INSERT_BATCH_SIZE', 128) or \
           (record_idx == len(extracted_face_records) - 1 and milvus_batch_data):
            try:
                logger.info(f"Вставка {len(milvus_batch_data)} эмбеддингов в Milvus...")
                insert_result = await loop.run_in_executor(None, milvus_collection.insert, milvus_batch_data)
                inserted_count_batch = len(insert_result.primary_keys)
                successfully_inserted_count += inserted_count_batch
                logger.info(f"Успешно вставлено {inserted_count_batch} эмбеддингов.")
                milvus_batch_data.clear()
            except Exception as e_milvus:
                logger.error(f"Ошибка Milvus при вставке батча: {e_milvus}")
                milvus_insert_errors += len(milvus_batch_data) # Считаем весь батч ошибочным
                milvus_batch_data.clear()
                if progress_callback: await progress_callback(photo_id=None, url="BATCH INSERT", status="error_milvus_insert_batch", error_message=str(e_milvus))

    try:
        logger.info("Финальный Flush данных Milvus (local_album_pipeline)...")
        await loop.run_in_executor(None, milvus_collection.flush)
        logger.info("Данные Milvus успешно сброшены на диск (local_album_pipeline).")
    except Exception as e_flush:
        logger.error(f"Ошибка Milvus flush (local_album_pipeline): {e_flush}")
        # Это не увеличивает счетчик milvus_insert_errors, т.к. это ошибка flush, а не insert

    error_summary = {'embedding': embedding_errors, 'milvus_insert': milvus_insert_errors, 'skipped_markers': skipped_markers_count}
    logger.info(f"LOCAL ALBUM PIPELINE - Завершено эмбеддинг и сохранение. Вставлено: {successfully_inserted_count}. Ошибки: {error_summary}")
    return successfully_inserted_count, error_summary

async def run_local_album_pipeline(
    image_data_tuples: List[Tuple[Any, str, Optional[str]]], # (photo_id, local_file_path, photo_date)
    milvus_collection: Collection, 
    progress_callback: Optional[Callable] = None,
    skip_existing_milvus_check: bool = False # Новый параметр
):
    """
    Основная функция для запуска пайплайна обработки локальных файлов альбома.
    """
    logger.info(f"--- ЗАПУСК LOCAL ALBUM PIPELINE для {len(image_data_tuples)} файлов ---")
    overall_start_time = time.monotonic()
    loop = asyncio.get_event_loop()
    deepface_semaphore = asyncio.Semaphore(getattr(config, 'DEEPFACE_CONCURRENCY_LIMIT', 4))
    
    init_deepface_model() # Прогрев, если еще не было

    # Очистка директории с извлеченными лицами (опционально, можно сделать параметром)
    # Сейчас очистка не делается, предполагая, что имена файлов уникальны благодаря photo_id и face_idx
    # if os.path.exists(EXTRACTED_FACES_BASE_PATH_LAP):
    #     logger.info(f"Очистка предыдущих извлеченных лиц из: {EXTRACTED_FACES_BASE_PATH_LAP}")
    #     shutil.rmtree(EXTRACTED_FACES_BASE_PATH_LAP)
    os.makedirs(EXTRACTED_FACES_BASE_PATH_LAP, exist_ok=True)

    # Фильтрация уже существующих в Milvus (если skip_existing_milvus_check=True)
    input_tuples_for_processing = []
    skipped_count_milvus = 0
    if skip_existing_milvus_check:
        logger.info("Проверка существующих photo_id в Milvus...")
        photo_ids_to_check = [item[0] for item in image_data_tuples]
        existing_ids_in_milvus = set()
        query_batch_size = 500 

        for i in range(0, len(photo_ids_to_check), query_batch_size):
            batch_ids = photo_ids_to_check[i:i+query_batch_size]
            if not batch_ids: continue

            str_ids_values = []
            num_ids_values = []

            for pid in batch_ids:
                if isinstance(pid, str):
                    # Для строк в Milvus нужны двойные кавычки внутри выражения IN
                    str_ids_values.append(f'\"{pid}\"') 
                elif isinstance(pid, (int, float)):
                    num_ids_values.append(str(pid))
                else: # Если тип не строка и не число, пробуем привести к строке и экранировать
                    try:
                        str_ids_values.append(f'\"{str(pid)}\"')
                    except Exception:
                        logger.warning(f"Не удалось преобразовать photo_id {pid} (тип: {type(pid)}) в строку для Milvus-запроса. Пропуск.")
                        continue
            
            expr_parts = []
            if str_ids_values:
                expr_parts.append(f"photo_id IN [{', '.join(str_ids_values)}]")
            if num_ids_values:
                expr_parts.append(f"photo_id IN [{', '.join(num_ids_values)}]")
            
            if not expr_parts: 
                logger.debug(f"В пачке ID для Milvus-запроса не найдено подходящих типизированных ID (исходные: {batch_ids})")
                continue
                
            expr = " or ".join(expr_parts)
            logger.debug(f"Сформировано Milvus выражение для проверки ID: {expr}")

            try:
                check_res = await loop.run_in_executor(None, milvus_collection.query, expr=expr, output_fields=["photo_id"], limit=len(batch_ids) + 10)
                for res_item in check_res:
                    existing_ids_in_milvus.add(str(res_item['photo_id'])) 
            except Exception as e_milvus_check:
                logger.error(f"Ошибка проверки photo_id в Milvus с выражением '{expr}': {e_milvus_check}. Пропуск проверки для этой пачки.")

        for item_tuple in image_data_tuples:
            # item_tuple[0] это photo_id, который должен быть строкой
            if str(item_tuple[0]) not in existing_ids_in_milvus:
                input_tuples_for_processing.append(item_tuple)
            else:
                skipped_count_milvus += 1
                if progress_callback: # Сообщаем о пропуске
                    await progress_callback(photo_id=item_tuple[0], url=item_tuple[1], status="skipped_existing_milvus", photo_date=item_tuple[2])
        logger.info(f"Проверка Milvus завершена. Пропущено {skipped_count_milvus} записей (уже в Milvus).")
    else:
        input_tuples_for_processing = image_data_tuples

    if not input_tuples_for_processing:
        logger.info("Нет файлов для обработки после проверки Milvus (или проверка была отключена, а список пуст).")
        if progress_callback:
            await progress_callback(photo_id=None, url=None, status="finished_no_files_to_process", total_count=len(image_data_tuples), skipped_count=skipped_count_milvus)
        return

    # --- ЭТАП 1: Извлечение лиц ---
    extracted_records = await extract_faces_from_local_files(
        input_tuples_for_processing, EXTRACTED_FACES_BASE_PATH_LAP, 
        deepface_semaphore, loop, progress_callback
    )
    # Статистика извлечения уже логируется внутри extract_faces_from_local_files
    # и передается через progress_callback
    faces_extracted_count = sum(1 for r in extracted_records if not r.get('is_no_face_marker'))
    no_face_markers_count = sum(1 for r in extracted_records if r.get('is_no_face_marker'))
    
    # --- ЭТАП 2: Эмбеддинг и сохранение в Milvus ---
    # Передаем offset_count для корректного отображения прогресса, если часть была пропущена Milvus-проверкой
    # total_count_overall остается общим количеством изначальных файлов для корректного % прогресса.
    milvus_inserted_count, milvus_error_counts = await embed_and_store_extracted_faces(
        extracted_records, milvus_collection, deepface_semaphore, loop, progress_callback,
        offset_count=skipped_count_milvus, 
        total_count_overall=len(image_data_tuples) 
    )

    overall_duration = time.monotonic() - overall_start_time
    logger.info(f"--- LOCAL ALBUM PIPELINE ЗАВЕРШЕН --- Времся: {overall_duration:.2f} сек.")

    if progress_callback:
        # Собираем финальную статистику
        # processed_count в finished должен быть равен total_count изначальному, если все обработано
        # skipped_count - это те, что были пропущены из-за Milvus
        # error_counts - ошибки эмбеддинга и вставки в Milvus
        # faces_extracted_count - реальные лица
        # no_face_markers_count - маркеры (ошибки чтения, не найдено лиц и т.д.)
        # milvus_inserted_total_count - успешно вставленные в Milvus
        await progress_callback(
            status="finished",
            total_count=len(image_data_tuples), # Общее количество фото на входе пайплайна
            processed_count=len(image_data_tuples), # Сколько всего было обработано до стадии finished
            skipped_count=skipped_count_milvus, 
            faces_extracted_count=faces_extracted_count,
            no_face_markers_count=no_face_markers_count, 
            embeddings_extracted_count=milvus_inserted_count, # Это фактически вставленные
            milvus_inserted_total_count=milvus_inserted_count,
            error_counts=milvus_error_counts, # Словарь ошибок {embedding: X, milvus_insert: Y}
            total_duration_sec=round(overall_duration, 2)
            # avg_throughput_photos_sec можно посчитать, если нужно
        )

# --- Пример использования (можно удалить или адаптировать) ---
async def example_main_local_album():
    logger.info("Пример запуска local_album_pipeline")
    
    # 1. Инициализация Milvus
    try:
        milvus_collection_obj = init_milvus_connection()
    except Exception as e_milvus:
        logger.critical(f"Не удалось инициализировать Milvus для example_main_local_album: {e_milvus}")
        return

    # 2. Формирование image_data_tuples (это должен делать web_server)
    # Здесь для примера создадим несколько фиктивных записей
    # Предположим, у нас есть скачанные файлы в config.LOCAL_DOWNLOAD_PATH
    dummy_download_path = getattr(config, 'LOCAL_DOWNLOAD_PATH', 'data/downloaded_images')
    if not os.path.exists(dummy_download_path) or not os.listdir(dummy_download_path):
        logger.warning(f"Для примера нужны файлы в {dummy_download_path}. Создайте их или измените путь.")
        # return
        # Создадим пару пустых файлов для теста, если их нет
        os.makedirs(dummy_download_path, exist_ok=True)
        if not os.listdir(dummy_download_path):
            try:
                with open(os.path.join(dummy_download_path, "test_img1.jpg"), 'w') as f: f.write("dummy")
                with open(os.path.join(dummy_download_path, "test_img2.png"), 'w') as f: f.write("dummy")
                logger.info(f"Созданы фиктивные файлы в {dummy_download_path} для теста.")
            except Exception as e_create_dummy: logger.error(f"Не удалось создать фиктивные файлы: {e_create_dummy}")

    example_image_tuples = []
    idx = 0
    for fname in os.listdir(dummy_download_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            fpath = os.path.join(dummy_download_path, fname)
            example_image_tuples.append((
                f"example_photo_{idx}", 
                fpath, 
                datetime.datetime.now().isoformat()
            ))
            idx +=1
            if idx >= 5: break # Возьмем не больше 5 для примера
    
    if not example_image_tuples:
        logger.error(f"Не найдено примеров изображений в {dummy_download_path} для запуска example_main_local_album.")
        return

    # 3. Определяем progress_callback (простой логгер для примера)
    async def _example_progress_callback(**kwargs):
        logger.info(f"[PROGRESS_CALLBACK_LAP]: {kwargs}")

    # 4. Запуск пайплайна
    await run_local_album_pipeline(
        image_data_tuples=example_image_tuples, 
        milvus_collection=milvus_collection_obj, 
        progress_callback=_example_progress_callback,
        skip_existing_milvus_check=True # Пропускать уже существующие в Milvus
    )

if __name__ == "__main__":
    # asyncio.run(example_main_local_album())
    logger.info("local_album_pipeline.py загружен. Для запуска используйте web_server.py или вызовите run_local_album_pipeline из другого скрипта.") 