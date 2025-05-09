import asyncio
import time
import datetime
import pytz
import os
import base64
from typing import Callable, Optional, Any, List, Tuple, Dict
import functools # Добавлен импорт functools

import aiohttp
import cv2
import numpy as np
from pymilvus import Collection, MilvusException
from loguru import logger
# Импорт DeepFace напрямую для get_dummy_embedding
from deepface import DeepFace 

# Предполагается, что config.py и другие утилиты находятся там же или в PYTHONPATH
try:
    from . import config
    from .download_utils import download_image_async, get_local_path_for_url # get_local_path_for_url нужен для проверки кэша в downloader
    from .image_processing import run_extraction_task_with_semaphore, run_represent_task_with_semaphore
except ImportError:
    import config
    from download_utils import download_image_async, get_local_path_for_url
    from image_processing import run_extraction_task_with_semaphore, run_represent_task_with_semaphore

# --- Константы и кэш для фиктивного эмбеддинга ---
NO_FACE_MARKER_FACE_INDEX = -1
# Размер изображения для генерации фиктивного эмбеддинга, можно вынести в config
DUMMY_IMAGE_SIZE = 16 
DUMMY_IMAGE_FOR_NO_FACE_REPRESENTATION = np.zeros((DUMMY_IMAGE_SIZE, DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
_CACHED_DUMMY_EMBEDDING: Optional[List[float]] = None
_dummy_embedding_lock = asyncio.Lock()

async def get_dummy_embedding(loop: asyncio.AbstractEventLoop, deepface_semaphore: asyncio.Semaphore) -> Optional[List[float]]:
    """
    Генерирует и кэширует фиктивный эмбеддинг для использования с маркерами отсутствия лиц.
    Использует глобальный кэш и лок для предотвращения многократной генерации.
    """
    global _CACHED_DUMMY_EMBEDDING
    if _CACHED_DUMMY_EMBEDDING is not None:
        return _CACHED_DUMMY_EMBEDDING
    
    async with _dummy_embedding_lock:
        if _CACHED_DUMMY_EMBEDDING is not None: # Повторная проверка после захвата лока
            return _CACHED_DUMMY_EMBEDDING
        
        logger.info("Попытка генерации фиктивного эмбеддинга...")
        try:
            async with deepface_semaphore: # Уважаем общий лимит параллельных вызовов DeepFace
                # Используем functools.partial для передачи аргументов в run_in_executor
                # DeepFace.represent принимает numpy массив напрямую как img_path
                # detector_backend='skip' критичен, т.к. мы не хотим детектировать лица на нашем фиктивном изображении
                partial_represent = functools.partial(
                    DeepFace.represent,
                    img_path=DUMMY_IMAGE_FOR_NO_FACE_REPRESENTATION,
                    model_name=config.DEEPFACE_MODEL_NAME,
                    enforce_detection=False, # Важно, т.к. лиц нет
                    detector_backend='skip'  # Критически важно для предварительно обработанных/фиктивных лиц
                )
                # Выполняем в executor, т.к. DeepFace.represent - блокирующая операция
                embedding_objs = await loop.run_in_executor(None, partial_represent)

                if embedding_objs and isinstance(embedding_objs, list) and \
                   len(embedding_objs) > 0 and isinstance(embedding_objs[0], dict) and \
                   "embedding" in embedding_objs[0] and isinstance(embedding_objs[0]["embedding"], list):
                    _CACHED_DUMMY_EMBEDDING = embedding_objs[0]["embedding"]
                    logger.info(f"Фиктивный эмбеддинг успешно сгенерирован и кэширован глобально (размерность: {len(_CACHED_DUMMY_EMBEDDING)}).")
                else:
                    logger.error("Не удалось сгенерировать фиктивный эмбеддинг! DeepFace.represent вернул неожиданный результат.")
                    # Можно вернуть None или возбудить исключение, чтобы обработать выше
                    return None 
            return _CACHED_DUMMY_EMBEDDING
        except Exception as e_dummy:
            logger.exception(f"Исключение во время генерации фиктивного эмбеддинга: {e_dummy}")
            return None # Возвращаем None в случае ошибки

async def process_and_store_faces_async(
    milvus_collection: Collection,
    image_data_tuples: list[tuple[any, str, Optional[str]]],
    progress_callback: Optional[Callable] = None,
    skip_existing: bool = False
):
    """Асинхронно обрабатывает изображения с использованием конвейера очередей."""
    overall_start_time = time.monotonic()
    total_images_to_process_initially = len(image_data_tuples)

    loop = asyncio.get_event_loop()
    moscow_tz = pytz.timezone('Europe/Moscow')

    # --- Инициализация Очередей ---
    download_queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
    process_queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
    embed_queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE * 5) # Может быть больше лиц
    milvus_queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE * 5)
    logger.info(f"Очереди инициализированы с размером {config.QUEUE_MAX_SIZE} (embed/milvus * 5).")

    # --- Инициализация Семафора для DeepFace ---
    deepface_semaphore = asyncio.Semaphore(config.DEEPFACE_CONCURRENCY_LIMIT)
    logger.info(f"Инициализирован семафор DeepFace с лимитом: {config.DEEPFACE_CONCURRENCY_LIMIT}")

    # --- Управление состоянием пайплайна ---
    active_tasks = set()
    processed_counter = 0
    inserted_counter = 0
    skipped_counter = 0
    error_counter = { 'download': 0, 'processing': 0, 'embedding': 0, 'milvus': 0, 'logic': 0 }
    # Используем Future для сигнализации о завершении milvus_inserter
    pipeline_finished_future = loop.create_future()
    # --- Инициализация счетчиков для статистики пайплайна ---
    processed_images_download_count = 0 
    total_faces_extracted_count = 0
    total_faces_embedded_count = 0
    total_no_face_markers_created_final_count = 0 # <--- Вот здесь его нужно инициализировать

    # --- Определения Worker Функций ---

    async def downloader_worker(
        worker_id: int,
        download_queue: asyncio.Queue,
        extraction_queue: asyncio.Queue,
        session: aiohttp.ClientSession,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Асинхронный воркер для загрузки изображений."""
        nonlocal processed_counter, skipped_counter, error_counter, processed_images_download_count
        logger.info(f"[Downloader-{worker_id}] Запущен.")
        failed_images_download_count = 0
        skipped_existing_count = 0
        total_processed_in_worker = 0
        while True:
            try:
                item = await download_queue.get()
                if item is None:
                    await process_queue.put(None)
                    download_queue.task_done()
                    logger.info(f"[Downloader-{worker_id}] Получен сигнал завершения.")
                    break

                photo_id, url, photo_date = item
                logger.debug(f"[Downloader-{worker_id}] Взял в работу ID: {photo_id}, URL: {url}")

                should_skip = False
                # --- Начало проверки на существование ---
                if skip_existing and (not isinstance(photo_id, str) or not photo_id.startswith('txt_')):
                    # Проверка локального кэша (оптимизация - не качать, если есть)
                    # Эта проверка использует config и get_local_path_for_url
                    use_local_cache = hasattr(config, 'DOWNLOAD_FILES_LOCALLY') and config.DOWNLOAD_FILES_LOCALLY
                    local_file_exists = False
                    if use_local_cache:
                         if hasattr(config, 'LOCAL_DOWNLOAD_PATH') and hasattr(config, 'OVERWRITE_LOCAL_FILES'):
                              local_path = get_local_path_for_url(photo_id, url, config.LOCAL_DOWNLOAD_PATH)
                              if os.path.exists(local_path) and not config.OVERWRITE_LOCAL_FILES:
                                   local_file_exists = True # Файл есть, но надо проверить Milvus
                         else:
                              logger.warning(f"[Downloader-{worker_id}] Локальное кэширование включено, но LOCAL_DOWNLOAD_PATH/OVERWRITE_LOCAL_FILES не настроены.")

                    # Проверка в Milvus ТОЛЬКО если файл НЕ найден локально (или кэш выключен/перезапись)
                    if not local_file_exists:
                        try:
                            # Исправляем экранирование символов для строковых ID
                            milvus_expr = f'photo_id == {photo_id}' if isinstance(photo_id, (int, float)) else f'photo_id == "{str(photo_id).replace("\\", "\\\\").replace('"', '\\"')}"'
                            query_start_time = time.monotonic()
                            check_res = await loop.run_in_executor(None, milvus_collection.query, expr=milvus_expr, output_fields=["photo_id"], limit=1)
                            query_duration_ms = (time.monotonic() - query_start_time) * 1000
                            logger.debug(f"[BENCHMARK][Downloader-{worker_id}] Milvus check exists (ID: {photo_id}) заняла {query_duration_ms:.2f} мс.")
                            if check_res:
                                should_skip = True
                                skipped_counter += 1
                                logger.info(f"[Downloader-{worker_id}] Пропуск фото ID {photo_id} (найден в Milvus).")
                                if progress_callback:
                                    await progress_callback(status="skipped_existing", photo_id=photo_id, url=url, photo_date=photo_date, timestamp_msk=datetime.datetime.now(moscow_tz).isoformat())
                        except Exception as e_check:
                            logger.error(f"[Downloader-{worker_id}] Ошибка проверки ID {photo_id} в Milvus: {e_check}. Обработка продолжится.")
                    else:
                        # Файл есть локально, значит он уже был скачан ранее.
                        # Если мы доверяем, что раз он скачан, то он и обработан (и skip_existing=True),
                        # то можно пропустить его.
                        # Однако, если процесс упал между скачиванием и записью в Milvus, это приведет к пропуску.
                        # Более надежно - всегда проверять Milvus, но это медленнее.
                        # Оставим текущую логику: пропускаем, только если НАЙДЕН В MILVUS.
                        # Наличие локального файла само по себе не причина пропускать при skip_existing=True.
                        pass


                # --- Конец проверки на существование ---

                result_data = None
                if not should_skip:
                    dl_photo_id, image_np = await download_image_async(session, photo_id, url)
                    # Важно: download_image_async возвращает исходный photo_id

                    image_base64_for_callback = None
                    status = "error_download" # Статус по умолчанию
                    if image_np is not None:
                         status = "downloaded_processing"
                         processed_images_download_count +=1 # Счетчик успешных загрузок/декодирований
                         if progress_callback:
                            try:
                                _, buffer = cv2.imencode('.jpg', image_np)
                                image_base64_for_callback = base64.b64encode(buffer).decode('utf-8')
                            except Exception as e_enc:
                                logger.warning(f"[Downloader-{worker_id}] Ошибка base64 для {url} (ID: {photo_id}): {e_enc}")

                    result_data = {
                        "photo_id": dl_photo_id, # Используем ID, возвращенный download_image_async
                        "url": url,
                        "image_np": image_np,
                        "image_base64": image_base64_for_callback,
                        "photo_date": photo_date
                    }
                    await process_queue.put(result_data)
                    if progress_callback:
                         await progress_callback(status=status, photo_id=dl_photo_id, url=url, image_base64=image_base64_for_callback, photo_date=photo_date, timestamp_msk=datetime.datetime.now(moscow_tz).isoformat())

                processed_counter += 1
                download_queue.task_done()
                logger.debug(f"[Downloader-{worker_id}] Завершил работу над ID: {photo_id}. Обработано всего: {processed_counter}")

            except asyncio.CancelledError:
                logger.info(f"[Downloader-{worker_id}] Задача отменена.")
                break
            except Exception as e:
                logger.exception(f"[Downloader-{worker_id}] Неперехваченная ошибка: {e}")
                if 'item' in locals() and item is not None: # Пытаемся пометить задачу в очереди выполненной
                     download_queue.task_done()
                error_counter['download'] += 1


    async def face_extraction_worker(worker_id: int):
        nonlocal error_counter, total_faces_extracted_count
        logger.info(f"[Extractor-{worker_id}] Запущен.")
        while True:
            try:
                item = await process_queue.get()
                if item is None:
                    await embed_queue.put(None)
                    process_queue.task_done()
                    logger.info(f"[Extractor-{worker_id}] Получен сигнал завершения.")
                    break

                photo_id = item["photo_id"]
                url = item["url"]
                image_np = item["image_np"]
                image_base64_cb = item["image_base64"]
                photo_date = item["photo_date"]

                logger.debug(f"[Extractor-{worker_id}] Взял в работу ID: {photo_id}, URL: {url}")

                if image_np is None:
                    logger.warning(f"[Extractor-{worker_id}] Пропуск обработки ID {photo_id}, т.к. изображение не было загружено.")
                    process_queue.task_done()
                    # Статистика для callback, если изображение не было загружено
                    if progress_callback:
                        await progress_callback(
                            status="error_extraction_no_image",
                            photo_id=photo_id, url=url, image_base64=image_base64_cb, photo_date=photo_date,
                            faces_count=0,
                            error_message="Image data was None at extraction stage",
                            timestamp_msk=datetime.datetime.now(moscow_tz).isoformat()
                        )
                    continue

                # ВАЖНО: run_extraction_task_with_semaphore использует блокирующий DeepFace внутри executor'а
                # Она уже обернута семафором
                ext_photo_id, ext_url, ext_img_base64, ext_photo_date, extraction_result_or_exc = await run_extraction_task_with_semaphore(
                    loop, deepface_semaphore, photo_id, url, image_np, image_base64_cb, photo_date
                )

                faces_found_on_image_count = 0
                error_on_extraction_msg = None
                face_confidences = []
                faces_put_to_queue = 0

                if isinstance(extraction_result_or_exc, Exception):
                    e = extraction_result_or_exc
                    error_on_extraction_msg = str(e)
                    error_counter['processing'] += 1
                    # Логирование ошибки есть в helper'е
                else:
                    detected_faces_list = extraction_result_or_exc
                    if detected_faces_list:
                        faces_found_on_image_count = len(detected_faces_list)
                        for face_idx, face_data in enumerate(detected_faces_list):
                            confidence = face_data.get('confidence', 0)
                            if confidence >= config.MIN_DET_SCORE:
                                face_np_data = face_data['face']
                                face_confidences.append(round(confidence, 4))
                                total_faces_extracted_count += 1 # Считаем успешно извлеченные лица
                                embed_data = {
                                    "photo_id": ext_photo_id,
                                    "url": ext_url,
                                    "face_idx": face_idx,
                                    "face_np": face_np_data,
                                    "photo_date": ext_photo_date,
                                    "image_base64": ext_img_base64,
                                    "is_no_face_marker": False # Явно указываем, что это не маркер
                                }
                                await embed_queue.put(embed_data)
                                faces_put_to_queue += 1
                            else:
                                logger.debug(f"[Extractor-{worker_id}] Лицо {face_idx} на {ext_url} (ID: {ext_photo_id}) пропущено (confidence: {confidence:.2f})")
                        logger.debug(f"[Extractor-{worker_id}] ID {ext_photo_id}: Найдено {faces_found_on_image_count} лиц, добавлено в embed_queue: {faces_put_to_queue}")
                    else:
                         logger.info(f"[Extractor-{worker_id}] Лица не найдены на изображении: {ext_url} (ID: {ext_photo_id})")
                
                # Если ни одно лицо не было отправлено в очередь эмбеддинга (ошибка, не найдено, не прошли порог)
                if faces_put_to_queue == 0 and not isinstance(extraction_result_or_exc, Exception):
                    # Отправляем маркер только если не было ошибки на этапе extraction_result_or_exc,
                    # т.к. ошибка уже залогирована и посчитана.
                    # Если была ошибка, то extraction_result_or_exc будет Exception, и мы не должны сюда попадать для маркера.
                    # Исключение - если extraction_result_or_exc это пустой список (нет лиц), тогда это не ошибка.
                    logger.info(f"[Extractor-{worker_id}] ID {ext_photo_id}: Подходящих лиц не найдено/не извлечено. Отправка NO_FACE_MARKER в embed_queue.")
                    no_face_marker_item = {
                        "photo_id": ext_photo_id,
                        "url": ext_url,
                        "face_idx": NO_FACE_MARKER_FACE_INDEX, # Специальный индекс
                        "face_np": None, # Нет реального лица
                        "photo_date": ext_photo_date,
                        "image_base64": ext_img_base64, # Для callback
                        "is_no_face_marker": True # Явный флаг
                    }
                    await embed_queue.put(no_face_marker_item)
                    # Этот маркер не считается "извлеченным лицом" в статистике total_faces_extracted_count
                    # но должен быть отражен в progress_callback
                    # Статус для коллбэка будет extraction_completed, но с faces_count=0 и faces_put_to_queue=0

                if progress_callback: 
                    status_cb = "extraction_completed"
                    if error_on_extraction_msg:
                        status_cb = "error_extraction"
                    elif faces_put_to_queue == 0 and not error_on_extraction_msg: # Если не было ошибки, но и лиц не добавили
                        status_cb = "extraction_no_faces_kept" # Новый статус для этого случая

                    await progress_callback(
                        status=status_cb,
                        photo_id=ext_photo_id, url=ext_url, image_base64=ext_img_base64, photo_date=ext_photo_date,
                        faces_count=faces_found_on_image_count, # Общее количество найденных до фильтрации
                        face_confidences=face_confidences if face_confidences else None, 
                        error_message=error_on_extraction_msg,
                        timestamp_msk=datetime.datetime.now(moscow_tz).isoformat()
                    )

                process_queue.task_done()
                logger.debug(f"[Extractor-{worker_id}] Завершил обработку ID: {ext_photo_id}")

            except asyncio.CancelledError:
                logger.info(f"[Extractor-{worker_id}] Задача отменена.")
                break
            except Exception as e:
                logger.exception(f"[Extractor-{worker_id}] Неперехваченная ошибка: {e}")
                if 'item' in locals() and item is not None:
                     process_queue.task_done()
                error_counter['processing'] += 1

    async def embedding_worker(worker_id: int):
        nonlocal error_counter, total_faces_embedded_count, total_no_face_markers_created_final_count
        # Убираем локальный счетчик, будем использовать общий
        # processed_no_face_markers_count = 0 

        logger.info(f"[Embedder-{worker_id}] Запущен.")
        while True:
            try:
                item = await embed_queue.get()
                if item is None:
                    await milvus_queue.put(None)
                    embed_queue.task_done()
                    logger.info(f"[Embedder-{worker_id}] Получен сигнал завершения.")
                    break

                photo_id = item["photo_id"]
                url = item["url"]
                face_idx = item["face_idx"]
                # face_np может быть None для маркера
                face_np = item.get("face_np") # Используем .get() для безопасности
                photo_date = item["photo_date"]
                image_base64_cb = item["image_base64"]
                is_marker = item.get("is_no_face_marker", False)

                current_time_msk = datetime.datetime.now(moscow_tz).isoformat() # Определяем здесь для обоих путей

                if is_marker:
                    logger.debug(f"[Embedder-{worker_id}] Обработка NO_FACE_MARKER для ID: {photo_id}")
                    dummy_emb = await get_dummy_embedding(loop, deepface_semaphore)
                    
                    if dummy_emb is None:
                        logger.error(f"[Embedder-{worker_id}] Не удалось получить фиктивный эмбеддинг для NO_FACE_MARKER ID: {photo_id}. Пропуск Milvus для этого маркера.")
                        error_counter['embedding'] +=1 # Считаем как ошибку этапа embedding
                        if progress_callback:
                            await progress_callback(
                                status="error_creating_no_face_marker",
                                photo_id=photo_id, url=url, image_base64=image_base64_cb,
                                face_index_processed=NO_FACE_MARKER_FACE_INDEX,
                                error_message="Failed to generate/retrieve dummy embedding for no_face_marker",
                                photo_date=photo_date,
                                timestamp_msk=current_time_msk
                            )
                        embed_queue.task_done()
                        continue

                    milvus_data = {
                        "photo_id": photo_id,
                        "embedding": dummy_emb,
                        "face_index": NO_FACE_MARKER_FACE_INDEX,
                        # "photo_date": photo_date, # Если photo_date хранится на уровне вектора в Milvus
                    }
                    await milvus_queue.put(milvus_data)
                    # processed_no_face_markers_count +=1 # <--- Заменяем на общий счетчик
                    total_no_face_markers_created_final_count +=1
                    logger.debug(f"[Embedder-{worker_id}] NO_FACE_MARKER для ID {photo_id} (face_idx {NO_FACE_MARKER_FACE_INDEX}) добавлен в milvus_queue.")
                    
                    if progress_callback:
                         await progress_callback(
                            status="no_face_marker_created",
                            photo_id=photo_id, url=url, image_base64=image_base64_cb,
                            face_index_processed=NO_FACE_MARKER_FACE_INDEX,
                            photo_date=photo_date,
                            timestamp_msk=current_time_msk
                         )
                    embed_queue.task_done()
                    continue # Переходим к следующему элементу из embed_queue

                # --- Если это не маркер, а реальное лицо --- 
                logger.debug(f"[Embedder-{worker_id}] Взял в работу лицо {face_idx} с ID: {photo_id}")

                # ВАЖНО: run_represent_task_with_semaphore использует блокирующий DeepFace внутри executor'а
                # Она уже обернута семафором
                result_data = await run_represent_task_with_semaphore(
                    loop, deepface_semaphore, photo_id, url, face_idx, face_np, photo_date
                )

                embedding_duration_ms = result_data.get("duration_ms", -1.0)
                current_time_msk = datetime.datetime.now(moscow_tz).isoformat()
                error_on_embedding_msg = None
                embedding_extracted = False

                if result_data["status"] == "error":
                    e_repr = result_data["error"]
                    error_on_embedding_msg = f"Exception during represent for face {face_idx}: {e_repr}"
                    error_counter['embedding'] += 1
                    # Логирование есть в helper'е
                else:
                    embedding_obj_list = result_data["embedding_obj_list"]
                    if not embedding_obj_list or not isinstance(embedding_obj_list, list) or not embedding_obj_list[0].get("embedding"):
                        error_on_embedding_msg = f"Failed to get embedding for face {face_idx} (represent result invalid)"
                        error_counter['embedding'] += 1
                        logger.warning(f"[Embedder-{worker_id}] {error_on_embedding_msg} для ID {photo_id}")
                    else:
                        embedding = embedding_obj_list[0]["embedding"]
                        milvus_data = {
                            "photo_id": photo_id,
                            "embedding": embedding,
                            "face_index": face_idx
                        }
                        await milvus_queue.put(milvus_data)
                        total_faces_embedded_count += 1 # Считаем успешные эмбеддинги
                        embedding_extracted = True
                        logger.debug(f"[Embedder-{worker_id}] Эмбеддинг для лица {face_idx} ID {photo_id} добавлен в milvus_queue.")

                if progress_callback:
                    await progress_callback(
                        status="embedding_extracted_face" if embedding_extracted else "error_embedding_face",
                        photo_id=photo_id, url=url, image_base64=image_base64_cb, # Передаем сохраненный base64
                        face_index_processed=face_idx,
                        error_message=error_on_embedding_msg,
                        embedding_duration_ms=embedding_duration_ms,
                        timestamp_msk=current_time_msk,
                        photo_date=photo_date
                    )

                embed_queue.task_done()
                logger.debug(f"[Embedder-{worker_id}] Завершил обработку лица {face_idx} ID {photo_id}")

            except asyncio.CancelledError:
                logger.info(f"[Embedder-{worker_id}] Задача отменена.")
                break
            except Exception as e:
                logger.exception(f"[Embedder-{worker_id}] Неперехваченная ошибка: {e}")
                if 'item' in locals() and item is not None:
                    embed_queue.task_done()
                error_counter['embedding'] += 1


    async def milvus_insertion_worker():
        nonlocal inserted_counter, error_counter
        logger.info("[MilvusInserter] Запущен.")
        batch_data = []
        active_embedders = config.EMBEDDING_WORKERS # Ожидаемое количество сигналов None

        while active_embedders > 0:
            insert_triggered = False
            try:
                # Ждем новый элемент или пока не придет сигнал None
                item = await milvus_queue.get()

                if item is None:
                    active_embedders -= 1
                    milvus_queue.task_done()
                    logger.info(f"[MilvusInserter] Получен сигнал None от Embedder. Осталось: {active_embedders}")
                    # Если это был последний embedder, нужно принудительно вставить остатки батча
                    if active_embedders == 0 and batch_data:
                         insert_triggered = True
                         logger.info(f"[MilvusInserter] Последний Embedder завершился. Вставка остатков ({len(batch_data)} записей)...")
                    # Если не последний или батч пуст, просто продолжаем цикл
                    if not insert_triggered:
                         continue

                # Если пришел не None, добавляем в батч
                if item is not None:
                    batch_data.append(item)
                    milvus_queue.task_done()
                    logger.debug(f"[MilvusInserter] Добавлен эмбеддинг в батч (ID: {item.get('photo_id')}, FaceIdx: {item.get('face_index')}). Батч: {len(batch_data)}.")
                    if len(batch_data) >= config.MILVUS_INSERT_BATCH_SIZE:
                        insert_triggered = True
                        logger.info(f"[MilvusInserter] Достигнут размер батча {config.MILVUS_INSERT_BATCH_SIZE}. Вставка...")

                # Выполняем вставку, если нужно (размер батча или последний embedder)
                if insert_triggered and batch_data:
                    try:
                        insert_start_time = time.monotonic()
                        insert_result = await loop.run_in_executor(None, milvus_collection.insert, batch_data)
                        insert_duration_ms = (time.monotonic() - insert_start_time) * 1000
                        inserted_count = len(insert_result.primary_keys)
                        inserted_counter += inserted_count
                        logger.info(f"[MilvusInserter] Успешно вставлено {inserted_count} эмбеддингов.")
                        logger.debug(f"[BENCHMARK] Milvus_collection.insert ({len(batch_data)} записей) заняла {insert_duration_ms:.2f} мс.")
                        batch_data.clear()
                    except MilvusException as e_milvus:
                         logger.error(f"[MilvusInserter] Ошибка Milvus при вставке батча: {e_milvus}")
                         error_counter['milvus'] += len(batch_data)
                         batch_data.clear()
                    except Exception as e_insert:
                        logger.exception(f"[MilvusInserter] Непредвиденная ошибка при вставке батча: {e_insert}")
                        error_counter['milvus'] += len(batch_data)
                        batch_data.clear()

            except asyncio.CancelledError:
                logger.info("[MilvusInserter] Задача отменена.")
                # Попытка вставить остатки перед выходом? Зависит от требований.
                # Пока просто выходим.
                break
            except Exception as e:
                logger.exception(f"[MilvusInserter] Неперехваченная ошибка в цикле: {e}")
                if 'item' in locals() and item is not None: # Если ошибка при обработке item'а
                     milvus_queue.task_done() # Нужно пометить, чтобы join() не завис
                error_counter['milvus'] += len(batch_data) # Считаем весь текущий батч потерянным
                batch_data.clear() # Очищаем батч

        # --- Цикл завершен ---
        logger.info("[MilvusInserter] Завершен.")
        pipeline_finished_future.set_result(True) # Сигнализируем о завершении


    # --- Логика Оркестровки ---
    logger.info("Запуск worker'ов пайплайна...")
    session = None
    # Переменные для статистики, которые раньше были глобальными для функции
    # processed_images_download_count = 0 # Уже инициализирован выше
    # total_faces_extracted_count = 0   # Уже инициализирован выше
    # total_faces_embedded_count = 0    # Уже инициализирован выше
    # total_no_face_markers_created_final_count = 0 # Уже инициализирован выше
    
    try:
        # TODO: Рассмотреть создание сессии с настройками коннектора из config
        session = aiohttp.ClientSession()

        # Создаем задачи для worker'ов
        for i in range(config.DOWNLOAD_WORKERS):
            task = asyncio.create_task(downloader_worker(i + 1, download_queue, process_queue, session, progress_callback))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        for i in range(config.EXTRACTION_WORKERS):
            task = asyncio.create_task(face_extraction_worker(i + 1))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        for i in range(config.EMBEDDING_WORKERS):
            task = asyncio.create_task(embedding_worker(i + 1))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        inserter_task = asyncio.create_task(milvus_insertion_worker())
        active_tasks.add(inserter_task)
        inserter_task.add_done_callback(active_tasks.discard)

        logger.info(f"Запущено: {config.DOWNLOAD_WORKERS} downloader'ов, {config.EXTRACTION_WORKERS} extractor'ов, {config.EMBEDDING_WORKERS} embedder'ов, 1 milvus_inserter.")

        # Заполнение начальной очереди
        logger.info(f"Добавление {len(image_data_tuples)} URL в очередь скачивания...")
        for item_tuple in image_data_tuples:
            await download_queue.put(item_tuple)
        logger.info("Все URL добавлены в очередь скачивания.")

        # Сигнал о завершении для downloader'ов
        for _ in range(config.DOWNLOAD_WORKERS):
            await download_queue.put(None)
        logger.info(f"Отправлены сигналы завершения ({config.DOWNLOAD_WORKERS}) в download_queue.")

        # Ожидание завершения обработки всех очередей
        logger.info("Ожидание завершения обработки всех очередей...")
        await download_queue.join()
        logger.info("Download queue обработана.")
        await process_queue.join()
        logger.info("Process queue обработана.")
        await embed_queue.join()
        logger.info("Embed queue обработана.")
        await milvus_queue.join()
        logger.info("Milvus queue обработана.")

        # Ожидание завершения задачи Milvus Inserter через Future
        logger.info("Ожидание завершения Milvus Inserter...")
        await pipeline_finished_future
        logger.info("Future завершения пайплайна получен.")

    except asyncio.CancelledError:
        logger.warning("Основная задача process_and_store_faces_async отменена.")
        # Отменяем все еще активные worker'ы
        # Собираем статистику из worker'ов перед выходом, если это возможно/необходимо
        # (сложно при отмене, лучше полагаться на то, что worker'ы логируют свои финальные счетчики или обновляют общие)
        active_worker_tasks = [t for t in active_tasks if not t.done()] # Копируем перед итерацией
        for task in active_worker_tasks:
            task.cancel()
        # Даем время на отмену и собираем результаты/исключения
        # await asyncio.sleep(1) # Небольшая задержка, чтобы задачи успели отмениться
        if active_worker_tasks: # Проверяем, есть ли что ожидать
             await asyncio.gather(*active_worker_tasks, return_exceptions=True)
        raise

    except Exception as e_orchestrator:
        logger.exception(f"Критическая ошибка в оркестраторе пайплайна: {e_orchestrator}")
        active_worker_tasks = [t for t in active_tasks if not t.done()] # Копируем перед итерацией
        for task in active_worker_tasks:
            task.cancel()
        # await asyncio.sleep(1)
        if active_worker_tasks:
            await asyncio.gather(*active_worker_tasks, return_exceptions=True)
        raise

    finally:
        # --- Сбор итоговой статистики от каждого worker'а --- 
        # Этот подход со сбором результатов worker'ов более сложен с текущей структурой (worker'ы - не классы)
        # Проще, если каждый worker обновляет общие счетчики (nonlocal), что уже частично делается.
        # Для processed_no_face_markers_count, он должен быть nonlocal в embedding_worker и агрегироваться здесь.
        # Предположим, что nonlocal счетчики будут агрегированы.
        # Чтобы получить точный total_no_face_markers_created_final_count, нужно было бы, чтобы
        # embedding_worker возвращал свои локальные счетчики или обновлял nonlocal агрегатор.
        # Текущий processed_no_face_markers_count - это локальный для каждого инстанса embedding_worker.
        # Для корректного подсчета, `total_no_face_markers_created_final_count` нужно будет
        # либо собирать из всех embedding_workers (если бы они возвращали значения), 
        # либо сделать `total_no_face_markers_created_final_count` nonlocal и обновлять его в `embedding_worker`.
        # Сделаем total_no_face_markers_created_final_count nonlocal и будем обновлять из embedding_worker.
        # Это требует добавления `nonlocal total_no_face_markers_created_final_count` в embedding_worker.

        if session and not session.closed:
            await session.close()
            logger.info("Сессия aiohttp закрыта.")

        # --- Финальный Flush Milvus ---
        logger.info("Финальный Flush данных Milvus...")
        try:
            flush_start_time = time.monotonic()
            # Используем run_in_executor для блокирующего flush
            await loop.run_in_executor(None, milvus_collection.flush)
            flush_duration_ms = (time.monotonic() - flush_start_time) * 1000
            logger.info(f"Данные для коллекции '{config.MILVUS_COLLECTION_NAME}' сброшены на диск.")
            logger.debug(f"[BENCHMARK] Финальный Milvus_collection.flush() заняла {flush_duration_ms:.2f} мс.")
        except Exception as e_flush:
            logger.error(f"Ошибка при финальном вызове milvus_collection.flush(): {e_flush}")
            error_counter['milvus'] += 1 # Считаем это ошибкой Milvus
            if progress_callback:
                 await progress_callback(status="error_critical", error_message=f"Milvus final flush error: {e_flush}", timestamp_msk=datetime.datetime.now(moscow_tz).isoformat())

        overall_end_time = time.monotonic()
        total_duration_sec = overall_end_time - overall_start_time
        # Используем счетчики, обновленные worker'ами
        processed_images_attempted = processed_counter - skipped_counter
        avg_throughput_photos_sec = (processed_counter / total_duration_sec) if total_duration_sec > 0 else 0

        # Логируем финальную статистику
        logger.info(f"--- Завершение пакетной обработки (Pipeline) --- ")
        logger.info(f"Всего изображений было в источнике: {total_images_to_process_initially}")
        logger.info(f"Прочитано и поставлено в очередь: {processed_counter}")
        logger.info(f"Пропущено из-за существования в Milvus: {skipped_counter}")
        # Используем счетчики, которые инкрементировались в worker'ах
        logger.info(f"Успешно загружено/декодировано: {processed_images_download_count}") # Этот счетчик нужно добавить
        logger.info(f"Всего лиц извлечено (прошли confidence): {total_faces_extracted_count}")
        logger.info(f"Всего эмбеддингов успешно получено (для реальных лиц): {total_faces_embedded_count}")
        logger.info(f"Всего маркеров 'нет лиц' создано и отправлено в Milvus: {total_no_face_markers_created_final_count}")
        logger.info(f"Всего лиц успешно вставлено в Milvus: {inserted_counter}")
        logger.info(f"Ошибки скачивания: {error_counter['download']}")
        logger.info(f"Ошибки детекции лиц: {error_counter['processing']}")
        logger.info(f"Ошибки извлечения эмбеддингов: {error_counter['embedding']}")
        logger.info(f"Ошибки вставки Milvus: {error_counter['milvus']}")
        logger.info(f"Другие/логические ошибки: {error_counter['logic']}") # Этот счетчик не обновлялся, нужно добавить если нужно
        logger.info(f"Общее время выполнения: {total_duration_sec:.2f} сек.")
        logger.info(f"Средняя пропускная способность: {avg_throughput_photos_sec:.2f} URL/сек.")

        if progress_callback:
            final_time_msk = datetime.datetime.now(moscow_tz).isoformat()
            await progress_callback(
                status="finished",
                processed_count=processed_counter,
                total_count=total_images_to_process_initially,
                skipped_count=skipped_counter,
                processed_images_dl_count=processed_images_download_count,
                faces_extracted_count=total_faces_extracted_count,
                embeddings_extracted_count=total_faces_embedded_count, # Реальные эмбеддинги
                no_face_markers_count=total_no_face_markers_created_final_count, # Маркеры
                milvus_inserted_total_count=inserted_counter, # Общее число записей в Milvus (включая маркеры)
                error_counts=error_counter,
                total_duration_sec=round(total_duration_sec, 2),
                avg_throughput_photos_sec=round(avg_throughput_photos_sec, 2),
                timestamp_msk=final_time_msk
            )
