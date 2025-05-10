import asyncio
import functools
import cv2
import numpy as np
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema
from deepface import DeepFace # DeepFace нужен для search_similar_faces...
from loguru import logger
from typing import List, Dict, Any, Optional # Добавил Optional

# Предполагается, что config.py и db_utils.py находятся в том же каталоге или доступны
try:
    from . import config
    # get_urls_for_photo_ids нужна для search_similar_faces_in_milvus_by_bytes
    from .db_utils import get_urls_for_photo_ids 
except ImportError:
    import config
    from db_utils import get_urls_for_photo_ids


def init_milvus_connection() -> Collection:
    """Устанавливает соединение с Milvus и возвращает объект коллекции."""
    try:
        if config.USE_MILVUS_LITE:
            logger.info(f"Используется Milvus Lite. Путь к файлу данных: {config.MILVUS_LITE_DATA_PATH}")
            connections.connect(alias="default", uri=config.MILVUS_LITE_DATA_PATH)
            logger.info(f"Успешное подключение к Milvus Lite: {config.MILVUS_LITE_DATA_PATH}")
        else:
            logger.info(f"Используется стандартный сервер Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
            connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
            logger.info(f"Успешное подключение к Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")

        if not utility.has_collection(config.MILVUS_COLLECTION_NAME):
            logger.info(f"Коллекция '{config.MILVUS_COLLECTION_NAME}' не найдена. Создание новой коллекции...")
            field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
            field_embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIMENSION)
            field_photo_id = FieldSchema(name="photo_id", dtype=DataType.INT64)
            field_face_index = FieldSchema(name="face_index", dtype=DataType.INT32)

            schema = CollectionSchema(
                fields=[field_id, field_embedding, field_photo_id, field_face_index],
                description="Коллекция эмбеддингов лиц с фотографий VK,关联 photo_id",
                enable_dynamic_field=False
            )
            collection = Collection(config.MILVUS_COLLECTION_NAME, schema=schema)
            logger.info(f"Коллекция '{config.MILVUS_COLLECTION_NAME}' успешно создана.")

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Индекс L2 для поля 'embedding' в коллекции '{config.MILVUS_COLLECTION_NAME}' создан.")
        else:
            logger.info(f"Коллекция '{config.MILVUS_COLLECTION_NAME}' уже существует.")
            collection = Collection(config.MILVUS_COLLECTION_NAME)
        
        collection.load()
        logger.info(f"Коллекция '{config.MILVUS_COLLECTION_NAME}' загружена в память.")
        return collection

    except Exception as e:
        logger.error(f"Ошибка подключения или настройки Milvus: {e}")
        raise

def search_similar_faces_in_milvus(
    milvus_collection: Collection, 
    query_image_path: str, 
    top_k: int = config.SEARCH_TOP_K, 
    search_threshold: float = config.SEARCH_THRESHOLD_L2
) -> List[Dict[str, Any]]:
    """Ищет похожие лица в Milvus для лиц, найденных на query_image_path с помощью DeepFace."""
    logger.info(f"Поиск похожих лиц для изображения: {query_image_path}")
    # loop = asyncio.get_event_loop() # Не используется здесь напрямую для DeepFace, т.к. функция синхронная

    try:
        query_image_np = cv2.imread(query_image_path)
        if query_image_np is None:
            logger.error(f"Не удалось загрузить изображение для поиска: {query_image_path}")
            return []
    except Exception as e:
        logger.error(f"Ошибка чтения файла изображения {query_image_path}: {e}")
        return []

    try:
        # DeepFace.extract_faces блокирующая
        detected_query_faces = DeepFace.extract_faces(
            img_path=query_image_np, 
            detector_backend=config.DEEPFACE_DETECTOR_BACKEND, 
            align=True,
            enforce_detection=False 
        )
    except Exception as e:
        if "Face could not be detected" in str(e) or "No face detected" in str(e):
            logger.info(f"Лица не найдены (или ошибка детекции) на query-изображении: {query_image_path}. Сообщение: {e}")
        else:
            logger.error(f"Ошибка при детекции лиц на query-изображении {query_image_path}: {e}")
        return []

    if not detected_query_faces:
        logger.info(f"Лица не найдены на query-изображении: {query_image_path}")
        return []

    logger.info(f"Найдено {len(detected_query_faces)} лиц(а) на query-изображении.")
    
    all_search_results = []

    for i, face_data in enumerate(detected_query_faces):
        confidence = face_data.get('confidence', 0)
        if confidence < config.MIN_DET_SCORE:
            logger.debug(f"Query-лицо {i} пропущено из-за низкого confidence: {confidence:.2f}")
            continue
        
        try:
            embedding_obj_list = DeepFace.represent(
                img_path=face_data['face'], 
                model_name=config.DEEPFACE_MODEL_NAME, 
                enforce_detection=False,
                align=True, 
                detector_backend=config.DEEPFACE_DETECTOR_BACKEND
            )

            if not embedding_obj_list or not embedding_obj_list[0].get("embedding"):
                logger.warning(f"Не удалось извлечь эмбеддинг для query-лица {i} на {query_image_path}")
                continue
            
            query_embedding = embedding_obj_list[0]["embedding"]
        except Exception as e:
            logger.error(f"Ошибка при извлечении эмбеддинга для query-лица {i}: {e}")
            continue

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": config.MILVUS_NPROBE}, 
        }

        logger.debug(f"Поиск для лица {i} с query-изображения...")
        try:
            results = milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding", 
                param=search_params, 
                limit=top_k, 
                expr="face_index >= 0",
                output_fields=['photo_id', 'face_index'] 
            )
            
            hits = results[0]
            logger.info(f"Найдено {len(hits)} потенциальных совпадений для лица {i} query-изображения (до фильтрации по порогу).")
            
            current_face_matches = []
            for hit in hits:
                distance = hit.distance
                logger.debug(f"    [DEBUG CHECK L2] Raw hit.distance: {distance} (type: {type(distance)}), Search Threshold L2: {search_threshold} (type: {type(search_threshold)})")
                threshold_check_passed = distance <= search_threshold
                logger.debug(f"    [DEBUG CHECK L2] Condition ({distance} <= {search_threshold}) is {threshold_check_passed}")
                
                if threshold_check_passed:
                    logger.debug(f"  Совпадение ПРОШЛО ПОРОГ L2 для query_face {i}: ID={hit.id}, L2 Distance={distance:.4f}, photo_id={hit.entity.get('photo_id')}, FaceIdx={hit.entity.get('face_index')}")
                    current_face_matches.append({
                        "query_face_index": i,
                        "match_photo_id": hit.entity.get('photo_id'),
                        "match_face_index": hit.entity.get('face_index'),
                        "distance": round(distance, 4)
                    })
                else:
                    logger.debug(f"  Совпадение ОТБРОШЕНО ПОРОГОМ L2 для query_face {i}: ID={hit.id}, L2 Distance={distance:.4f}, Threshold={search_threshold}")
            all_search_results.extend(current_face_matches)
        except Exception as e:
            logger.error(f"Ошибка при поиске в Milvus для лица {i}: {e}")
            
    logger.info(f"Всего найдено {len(all_search_results)} совпадений, удовлетворяющих порогу L2 <= {search_threshold}.")
    return all_search_results


async def search_similar_faces_in_milvus_by_bytes(
    milvus_collection: Collection, 
    query_image_bytes: bytes, 
    top_k: int = config.SEARCH_TOP_K, 
    search_threshold: float = config.SEARCH_THRESHOLD_L2
) -> List[Dict[str, Any]]:
    """Ищет похожие лица в Milvus для лиц, найденных на query_image (переданном как байты)."""
    logger.info(f"Поиск похожих лиц для изображения (переданного как байты). Размер: {len(query_image_bytes)} байт.")
    loop = asyncio.get_event_loop() 
    # all_search_results = [] # Эта переменная не используется, результаты собираются в processed_results

    try:
        image_array = np.frombuffer(query_image_bytes, np.uint8)
        query_image_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if query_image_np is None:
            logger.error(f"Не удалось декодировать query-изображение из байтов.")
            return [] 
    except Exception as e:
        logger.error(f"Ошибка декодирования query-изображения: {e}")
        return []

    # functools.partial используется для передачи дополнительных аргументов в функции,
    # которые будут выполняться в loop.run_in_executor.
    # DeepFace функции блокирующие, поэтому их нужно запускать в executor'е.
    
    _partial_extract_faces = functools.partial( # Забыл импортировать functools
        DeepFace.extract_faces,
        # img_path=query_image_np, # передается как аргумент executor'у
        detector_backend=config.DEEPFACE_DETECTOR_BACKEND,
        enforce_detection=False,
        align=True
    )

    _partial_represent_one_face = functools.partial(
        DeepFace.represent,
        # img_path=face_np, # передается как аргумент executor'у
        model_name=config.DEEPFACE_MODEL_NAME,
        enforce_detection=False,
        detector_backend=config.DEEPFACE_DETECTOR_BACKEND,
        align=True
    )

    try:
        logger.debug("Запуск детекции лиц на query-изображении...")
        detected_query_faces = await loop.run_in_executor(None, _partial_extract_faces, query_image_np)
        logger.debug(f"Детекция лиц завершена. Найдено: {len(detected_query_faces) if detected_query_faces else 0}")

    except Exception as e:
        if "Face could not be detected" in str(e) or "No face detected" in str(e):
            logger.info(f"Лица не найдены (или ошибка детекции) на query-изображении (из байтов). Сообщение: {e}")
        else:
            logger.error(f"Ошибка при детекции лиц на query-изображении (из байтов): {e}")
        return []

    if not detected_query_faces:
        logger.info(f"Лица не найдены на query-изображении (из байтов).")
        return []

    logger.info(f"Найдено {len(detected_query_faces)} лиц(а) на query-изображении (из байтов). Запуск поиска...")
    
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": config.MILVUS_NPROBE},
    }

    milvus_search_tasks = []
    query_face_embeddings_map = {} 
    # Сохраняем информацию о query-лицах, для которых будем делать поиск
    # (индекс лица, numpy массив лица)
    faces_to_process_for_embedding = []

    for i, face_data in enumerate(detected_query_faces):
        confidence = face_data.get('confidence', 0)
        if confidence < config.MIN_DET_SCORE: 
            logger.debug(f"Query-лицо {i} пропущено из-за низкого confidence: {confidence:.2f}")
            continue
        faces_to_process_for_embedding.append({'original_index': i, 'face_np': face_data['face']})

    if not faces_to_process_for_embedding:
        logger.info("Нет query-лиц для извлечения эмбеддингов (прошли confidence).")
        return []

    # Асинхронное извлечение эмбеддингов для всех подходящих лиц
    embedding_tasks = []
    for face_info in faces_to_process_for_embedding:
        embedding_tasks.append(
            loop.run_in_executor(None, _partial_represent_one_face, face_info['face_np'])
        )
    
    try:
        embedding_results_list = await asyncio.gather(*embedding_tasks, return_exceptions=True)
    except Exception as e_gather_embed:
        logger.error(f"Критическая ошибка при ожидании результатов извлечения эмбеддингов: {e_gather_embed}")
        return [] # Если gather упал, дальше идти нет смысла

    # Теперь связываем результаты эмбеддингов с исходными лицами и запускаем поиск в Milvus
    for idx, emb_result_or_exc in enumerate(embedding_results_list):
        original_face_info = faces_to_process_for_embedding[idx]
        query_face_original_idx = original_face_info['original_index']

        if isinstance(emb_result_or_exc, Exception):
            logger.error(f"Ошибка при извлечении эмбеддинга для query-лица {query_face_original_idx}: {emb_result_or_exc}")
            continue # к следующему лицу

        embedding_obj_list = emb_result_or_exc
        if not embedding_obj_list or not isinstance(embedding_obj_list, list) or not embedding_obj_list[0].get("embedding"):
            logger.warning(f"Не удалось извлечь эмбеддинг для query-лица {query_face_original_idx}. Результат: {embedding_obj_list}")
            continue
        
        query_embedding = embedding_obj_list[0]["embedding"]
        query_face_embeddings_map[query_face_original_idx] = query_embedding
        logger.debug(f"Эмбеддинг для query-лица {query_face_original_idx} получен.")

        _partial_milvus_search = functools.partial(
            milvus_collection.search,
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr="face_index >= 0",
            output_fields=['photo_id', 'face_index']
        )
        # Сохраняем оригинальный индекс query-лица вместе с задачей
        milvus_search_tasks.append(
            (query_face_original_idx, loop.run_in_executor(None, _partial_milvus_search))
        )

    if not milvus_search_tasks:
        logger.info("Нет эмбеддингов для выполнения поиска в Milvus.")
        return []

    # Сбор результатов поиска из Milvus
    # asyncio.gather ожидает корутины, а у нас loop.run_in_executor возвращает Future.
    # Чтобы собрать результаты, нужно await для каждого future.
    
    raw_search_results_with_indices = []
    search_coroutines = [task_tuple[1] for task_tuple in milvus_search_tasks] # Только корутины/future
    
    try:
        gathered_milvus_results = await asyncio.gather(*search_coroutines, return_exceptions=True)
    except Exception as e_gather_search:
        logger.error(f"Критическая ошибка при ожидании результатов поиска Milvus: {e_gather_search}")
        return []

    for i, result_or_exc in enumerate(gathered_milvus_results):
        original_query_idx = milvus_search_tasks[i][0] # Получаем исходный query_face_index
        if isinstance(result_or_exc, Exception):
            logger.error(f"Ошибка при поиске в Milvus для query-лица {original_query_idx}: {result_or_exc}")
            continue
        raw_search_results_with_indices.append({'query_face_index': original_query_idx, 'hits_data': result_or_exc})

    processed_results = []
    matched_photo_ids = set()

    for res_item in raw_search_results_with_indices:
        query_face_idx = res_item['query_face_index']
        result_set = res_item['hits_data'] # это results от milvus_collection.search
        
        if not result_set or not result_set[0]: 
            logger.info(f"Нет результатов поиска Milvus для query-лица {query_face_idx}.")
            continue
            
        hits = result_set[0] # result_set[0] это Hits
        logger.info(f"Найдено {len(hits)} потенциальных совпадений для query-лица {query_face_idx} (до фильтрации по порогу L2).")
            
        for hit in hits:
            distance = hit.distance
            logger.debug(f"    [DEBUG CHECK L2] Raw hit.distance: {distance} (type: {type(distance)}), Search Threshold L2: {search_threshold} (type: {type(search_threshold)})")
            threshold_check_passed = distance <= search_threshold
            logger.debug(f"    [DEBUG CHECK L2] Condition ({distance} <= {search_threshold}) is {threshold_check_passed}")
            
            if threshold_check_passed:
                match_photo_id = hit.entity.get('photo_id')
                match_face_index = hit.entity.get('face_index')
                logger.debug(f"  Совпадение ПРОШЛО ПОРОГ L2 для query_face {query_face_idx}: ID={hit.id}, L2 Distance={distance:.4f}, photo_id={match_photo_id}, FaceIdx={match_face_index}")
                processed_results.append({
                    "query_face_index": query_face_idx,
                    "match_photo_id": match_photo_id,
                    "match_face_index": match_face_index,
                    "distance": round(distance, 4)
                })
                if match_photo_id: # Убедимся, что match_photo_id не None
                    matched_photo_ids.add(match_photo_id)
            else:
                logger.debug(f"  Совпадение ОТБРОШЕНО ПОРОГОМ L2 для query_face {query_face_idx}: ID={hit.id}, L2 Distance={distance:.4f}, Threshold={search_threshold}")

    urls_and_dates_map = {}
    if matched_photo_ids:
        logger.info(f"Получение URL и дат для {len(matched_photo_ids)} найденных photo_id...")
        # Передаем db_path из config
        urls_and_dates_map = get_urls_for_photo_ids(
            photo_ids=list(matched_photo_ids),
            db_path=config.URL_FILE_PATH 
        )
        logger.info(f"URL и даты получены.")
    
    for res in processed_results:
        photo_info = urls_and_dates_map.get(res["match_photo_id"])
        if photo_info:
            res["match_url"] = photo_info.get("url")
            res["match_date"] = photo_info.get("date")
        else:
            res["match_url"] = None
            res["match_date"] = None

    logger.info(f"Поиск завершен. Всего найдено {len(processed_results)} совпадений, удовлетворяющих порогу L2 <= {search_threshold}.")
    return processed_results
