from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
)
from loguru import logger
import os

# --- Константы Milvus ---
MILVUS_ALIAS = "default"
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_lite.db") 
COLLECTION_NAME = "faces"
# FACE_ID_FIELD = "face_id" # Удалено
EMBEDDING_FIELD = "embedding"
PERSON_ID_FIELD = "person_id"   # Теперь INT64
PHOTO_ID_FIELD = "photo_id"
FACE_INDEX_FIELD = "face_index" 
PRIMARY_KEY_FIELD = "pk" 
DEFAULT_PERSON_ID = 0             # Изменено на 0 (INT64)
EMBEDDING_DIM = 512 
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "IP" 
NLIST_PARAM = 128 
NPROBE_PARAM = 16
DEFAULT_SEARCH_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 300

# Настройка логгера (если еще не настроен глобально)
# logger.add(sys.stderr, level="INFO") 
# logger.add("milvus_service.log", level="DEBUG", rotation="5 MB")

# --- Функции Milvus --- 

def setup_milvus() -> MilvusClient | None:
    """Подключается к Milvus, создает и загружает коллекцию/индекс.
    Возвращает MilvusClient или None в случае ошибки."""
    try:
        client = MilvusClient(uri=MILVUS_URI)
        logger.info(f"Успешное подключение к Milvus Lite: {MILVUS_URI}")
        if not connections.has_connection(MILVUS_ALIAS):
             connections.connect(alias=MILVUS_ALIAS, uri=MILVUS_URI)
             logger.info(f"Низкоуровневое соединение '{MILVUS_ALIAS}' установлено.")
    except Exception as e:
        logger.error(f"Ошибка подключения к Milvus Lite: {e}")
        return None

    try:
        has_collection = client.has_collection(collection_name=COLLECTION_NAME)
        collection_low_level = None 

        if not has_collection:
            logger.info(f"Коллекция '{COLLECTION_NAME}' не найдена. Создание...")
            fields = [
                # FieldSchema(name=FACE_ID_FIELD, dtype=DataType.VARCHAR, max_length=255), # Удалено
                FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name=PERSON_ID_FIELD, dtype=DataType.INT64),
                FieldSchema(name=PHOTO_ID_FIELD, dtype=DataType.INT64),
                FieldSchema(name=FACE_INDEX_FIELD, dtype=DataType.INT8),
                FieldSchema(name=PRIMARY_KEY_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True)
            ]
            schema = CollectionSchema(fields=fields)
            client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
            logger.info(f"Коллекция '{COLLECTION_NAME}' успешно создана.")

            collection_low_level = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
            logger.info(f"Создание индекса '{INDEX_TYPE}' для поля '{EMBEDDING_FIELD}'...")
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name=EMBEDDING_FIELD,
                index_name="vector_index",
                index_type=INDEX_TYPE,
                metric_type=METRIC_TYPE,
                params={"nlist": NLIST_PARAM}
            )
            client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
            has_index_check = collection_low_level.has_index(index_name="vector_index")
            logger.info(f"Проверка создания индекса 'vector_index': {has_index_check}")
            if not has_index_check: logger.warning(f"Не удалось подтвердить создание индекса 'vector_index'")

        else:
            logger.info(f"Коллекция '{COLLECTION_NAME}' уже существует.")
            collection_low_level = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
            has_index_check = collection_low_level.has_index(index_name="vector_index")
            logger.info(f"Проверка индекса 'vector_index': {has_index_check}")
            if not has_index_check:
                logger.warning(f"Индекс 'vector_index' не найден. Попытка создать...")
                try:
                    index_params = client.prepare_index_params()
                    index_params.add_index(
                        field_name=EMBEDDING_FIELD, index_name="vector_index",
                        index_type=INDEX_TYPE, metric_type=METRIC_TYPE,
                        params={"nlist": NLIST_PARAM}
                    )
                    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
                    has_index_check = collection_low_level.has_index(index_name="vector_index")
                    logger.info(f"Повторная проверка индекса 'vector_index': {has_index_check}")
                except Exception as index_e:
                    logger.error(f"Ошибка при создании индекса 'vector_index' для существующей коллекции: {index_e}")

        if collection_low_level:
            load_state = client.get_load_state(collection_name=COLLECTION_NAME)
            logger.info(f"Состояние загрузки '{COLLECTION_NAME}': {load_state}")
            if load_state != "loaded":
                logger.info(f"Загрузка коллекции '{COLLECTION_NAME}' в память...")
                collection_low_level.load()
                logger.info(f"Коллекция '{COLLECTION_NAME}' загружена.")
            else:
                 logger.info(f"Коллекция '{COLLECTION_NAME}' уже была загружена.")
        else:
            logger.error("Не удалось получить объект Collection для загрузки.")
            return None 

        return client 

    except Exception as e:
        logger.error(f"Ошибка при настройке коллекции/индекса Milvus: {e}", exc_info=True)
        return None

def get_cluster_members(client: MilvusClient, person_ids: set[int]) -> dict:
    """Получает всех членов заданных кластеров (PersonID)."""
    cluster_members = {} 
    if not person_ids:
        return cluster_members

    logger.info(f"Получение всех членов для PersonIDs: {person_ids}")
    pids_list_str = ", ".join([str(pid) for pid in person_ids])
    person_id_filter = f'{PERSON_ID_FIELD} in [{pids_list_str}]'
    logger.debug(f"Фильтр для запроса членов кластера: {person_id_filter}")

    try:
        total_members_count = client.query(collection_name=COLLECTION_NAME, filter=person_id_filter, output_fields=["count(*)"])[0]["count(*)"]
        logger.debug(f"Ожидаемое количество членов кластера: {total_members_count}")
        
        all_members_results = []
        if total_members_count > 0:
            limit = min(total_members_count, 16383) 
            if total_members_count > limit:
                 logger.warning(f"Количество членов кластера ({total_members_count}) превышает лимит запроса ({limit}). Будут получены не все.")

            all_members_results = client.query(
                 collection_name=COLLECTION_NAME,
                 filter=person_id_filter,
                 output_fields=[PRIMARY_KEY_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD, PERSON_ID_FIELD],
                 limit=limit 
            )
        
        logger.info(f"Извлечено {len(all_members_results)} записей членов кластера.")
        for member_entity in all_members_results:
            pid = member_entity[PERSON_ID_FIELD]
            member_data = {
                "pk": member_entity[PRIMARY_KEY_FIELD],
                "photo_id": member_entity[PHOTO_ID_FIELD],
                "face_index": member_entity[FACE_INDEX_FIELD],
                "person_id": pid
            }
            cluster_members.setdefault(pid, []).append(member_data)
            
    except Exception as query_e:
         logger.error(f"Ошибка при запросе членов кластера: {query_e}")

    return cluster_members

# Добавим функции-обертки для основных операций

def insert_data(client: MilvusClient, data_list: list[dict]) -> list | None:
    """Вставляет данные в коллекцию Milvus.
    data_list: список словарей, каждый словарь представляет одну запись.
    Возвращает список PK или None в случае ошибки.
    """
    if not data_list:
        logger.warning("Нет данных для вставки в Milvus.")
        return []
    
    try:
        # Проверим, что все записи содержат ожидаемые поля и корректные типы
        required_fields = {EMBEDDING_FIELD, PERSON_ID_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD}
        for item_idx, item in enumerate(data_list):
            if not required_fields.issubset(item.keys()):
                logger.error(f"Запись {item_idx} для вставки не содержит все необходимые поля ({required_fields}). Проверьте ключи: {item.keys()}. Запись: {item}")
                return None

            # Type checks for integer fields
            for field_name in [PERSON_ID_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD]:
                value = item.get(field_name)
                if not isinstance(value, int):
                    logger.error(f"Запись {item_idx}: Поле '{field_name}' (значение: {value}, тип: {type(value)}) должно быть int. Запись: {item}")
                    return None
        
        logger.debug(f"Попытка вставки {len(data_list)} записей через client.insert...")
        res = client.insert(collection_name=COLLECTION_NAME, data=data_list)
        
        insert_count_val = None
        primary_keys_val = None

        # Пытаемся получить insert_count
        if hasattr(res, 'insert_count'):
            insert_count_val = res.insert_count
        elif isinstance(res, dict) and 'insert_count' in res:
            insert_count_val = res['insert_count']

        # Пытаемся получить primary_keys или ids
        if hasattr(res, 'primary_keys'):
            primary_keys_val = res.primary_keys
        elif isinstance(res, dict) and 'ids' in res:
            primary_keys_val = res['ids'] # Ключ 'ids' для OmitZeroDict/dict
        elif isinstance(res, dict) and 'primary_keys' in res: # Запасной вариант для dict
            primary_keys_val = res['primary_keys']

        if insert_count_val is not None and primary_keys_val is not None:
            logger.info(f"{insert_count_val} записей успешно вставлено. PKs: {primary_keys_val}")
            return primary_keys_val
        else:
            # Это случай, когда client.insert() вернул что-то неожиданное
            error_message = (
                f"Результат client.insert() не содержит ожидаемые атрибуты/ключи ('insert_count', 'primary_keys'/'ids'). "
                f"Тип результата: {type(res)}, Содержимое (до 500 символов): {str(res)[:500]}"
            )
            logger.error(error_message)
            # Попытка извлечь детали ошибки, если это объект ошибки Milvus
            if hasattr(res, 'code') and hasattr(res, 'message'):
                 logger.error(f"Возможно, это объект ошибки Milvus: code={getattr(res, 'code', 'N/A')}, message='{getattr(res, 'message', 'N/A')}'")
            elif isinstance(res, dict): # OmitZeroDict может быть словарем
                 logger.error(f"Содержимое словаря результата: {res}")

            # Возвращаем None, так как вставка не была подтверждена как успешная
            return None

    except AttributeError as ae:
        # Этот блок теперь менее вероятен, если предыдущая проверка hasattr ловит проблему,
        # но оставлен для непредвиденных случаев AttributeError.
        logger.error(f"AttributeError при доступе к результатам вставки: {ae}. "
                     f"Это может означать, что 'res' не является ожидаемым объектом MutationResult. "
                     f"Тип res: {type(res)}, Содержимое res (если доступно): {str(getattr(res, '__dict__', res))[:500]}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Ошибка при вставке данных в Milvus или обработке результата: {e}", exc_info=True)
        # Попытка логировать 'res', если оно было присвоено до исключения
        try:
            if 'res' in locals():
                logger.error(f"Состояние 'res' на момент исключения: Тип: {type(res)}, Содержимое: {str(res)[:500]}")
        except Exception as log_res_e:
            logger.error(f"Не удалось залогировать 'res' при обработке исключения: {log_res_e}")
        return None

def search_data(client: MilvusClient, query_vectors: list[list[float]], top_k: int) -> list[list[dict]]:
    """Выполняет поиск ближайших соседей в Milvus.
    Возвращает список списков результатов (по одному для каждого вектора запроса).
    Каждый результат - словарь с 'distance' и 'entity'.
    """
    if not query_vectors:
        logger.warning("Нет векторов для поиска в Milvus.")
        return []
        
    try:
        search_params = {"nprobe": NPROBE_PARAM}
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=query_vectors,
            anns_field=EMBEDDING_FIELD,
            metric_type=METRIC_TYPE,
            params=search_params,
            limit=top_k,
            output_fields=[PRIMARY_KEY_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD, PERSON_ID_FIELD]
        )
        return results
    except Exception as e:
        logger.error(f"Ошибка при поиске данных в Milvus: {e}", exc_info=True)
        return [] 

def query_data(client: MilvusClient, filter_expression: str = "", output_fields: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Выполняет запрос к Milvus по фильтру.
    Возвращает список словарей с результатами или пустой список.
    """
    if output_fields is None:
        # Обновляем поля по умолчанию
        output_fields = [PRIMARY_KEY_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD, PERSON_ID_FIELD, EMBEDDING_FIELD] 
        
    try:
        if limit is None:
             count_res = client.query(collection_name=COLLECTION_NAME, filter=filter_expression, output_fields=["count(*)"])
             total_count = count_res[0]["count(*)"]
             limit = min(total_count, 16383) 
             logger.debug(f"Запрос без лимита, установлено limit={limit} (total={total_count})")

        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expression,
            output_fields=output_fields,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Ошибка при выполнении запроса Milvus (filter='{filter_expression}'): {e}", exc_info=True)
        return []

def upsert_data(client: MilvusClient, data_list: list[dict], batch_size: int = 500) -> bool:
    """Обновляет или вставляет данные в коллекцию Milvus.
    data_list: список словарей, каждый словарь представляет одну запись (должен содержать PK).
    batch_size: размер пакета для операции upsert.
    Возвращает True в случае успеха (хотя бы один батч), False в случае ошибки.
    """
    if not data_list:
        logger.warning("Нет данных для upsert в Milvus.")
        return True # Считаем успехом, так как нечего было делать
    
    logger.info(f"Подготовлено {len(data_list)} записей для Upsert в Milvus.")
    success = False
    
    try:
        # Убедимся, что соединение для Collection API есть
        if not connections.has_connection(MILVUS_ALIAS):
             connections.connect(alias=MILVUS_ALIAS, uri=MILVUS_URI)
             logger.info(f"Низкоуровневое соединение '{MILVUS_ALIAS}' установлено для Upsert.")
        
        # Получаем полную информацию для каждой записи, включая embedding
        complete_data_list = []
        for item in data_list:
            if PRIMARY_KEY_FIELD not in item:
                logger.error(f"Upsert: запись не содержит обязательное поле '{PRIMARY_KEY_FIELD}': {item}")
                return False
            
            pk_value = item[PRIMARY_KEY_FIELD]
            
            # Если embedding отсутствует, запрашиваем все данные для этой записи
            if EMBEDDING_FIELD not in item:
                try:
                    pk_filter = f"{PRIMARY_KEY_FIELD} == {pk_value}"
                    original_record = client.query(
                        collection_name=COLLECTION_NAME,
                        filter=pk_filter,
                        output_fields=[EMBEDDING_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD],
                        limit=1
                    )
                    
                    if not original_record:
                        logger.error(f"Не найдена запись с {PRIMARY_KEY_FIELD}={pk_value} для получения embedding")
                        continue
                        
                    # Добавляем все необходимые поля из оригинальной записи
                    complete_item = item.copy()  # Сохраняем указанные поля (например, PERSON_ID_FIELD)
                    
                    # Добавляем embedding и другие обязательные поля
                    complete_item[EMBEDDING_FIELD] = original_record[0][EMBEDDING_FIELD]
                    
                    # Добавляем остальные поля, если их нет в item
                    if PHOTO_ID_FIELD not in complete_item and PHOTO_ID_FIELD in original_record[0]:
                        complete_item[PHOTO_ID_FIELD] = original_record[0][PHOTO_ID_FIELD]
                    
                    if FACE_INDEX_FIELD not in complete_item and FACE_INDEX_FIELD in original_record[0]:
                        complete_item[FACE_INDEX_FIELD] = original_record[0][FACE_INDEX_FIELD]
                    
                    complete_data_list.append(complete_item)
                except Exception as e:
                    logger.error(f"Ошибка при получении данных для записи {pk_value}: {e}")
                    continue
            else:
                # Если embedding уже есть, просто используем запись как есть
                complete_data_list.append(item)
        
        if not complete_data_list:
            logger.error("Нет данных для upsert после получения полной информации")
            return False
        
        logger.info(f"Подготовлено {len(complete_data_list)} полных записей для Upsert")     
             
        collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
        for i in range(0, len(complete_data_list), batch_size):
            batch = complete_data_list[i:i + batch_size]
            num_batches = (len(complete_data_list) + batch_size - 1) // batch_size
            current_batch_num = i // batch_size + 1
            logger.debug(f"Upsert батча {current_batch_num} / {num_batches} (размер {len(batch)})...)")
            for item in batch:
                if PERSON_ID_FIELD in item and not isinstance(item[PERSON_ID_FIELD], int):
                    logger.error(f"Upsert: Поле '{PERSON_ID_FIELD}' ({item[PERSON_ID_FIELD]}) должно быть int. Запись: {item}")
                    # return False # Строгая проверка
            res = collection.upsert(data=batch) # Upsert ожидает, что в batch есть PK
            success = True 

        collection.flush()
        logger.debug("Выполнен flush коллекции после Upsert.")
        logger.info(f"Upsert {len(complete_data_list)} записей завершен.")
        return success
    except Exception as e:
        logger.error(f"Ошибка при выполнении Upsert в Milvus: {e}", exc_info=True)
        return False
