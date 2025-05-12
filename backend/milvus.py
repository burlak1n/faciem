from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
)
from loguru import logger

# --- Константы Milvus ---
MILVUS_ALIAS = "default"
MILVUS_URI = "./milvus_lite.db" 
COLLECTION_NAME = "faces"
FACE_ID_FIELD = "face_id"
EMBEDDING_FIELD = "embedding"
PERSON_ID_FIELD = "person_id"
PRIMARY_KEY_FIELD = "pk"
DEFAULT_PERSON_ID = "-1"
EMBEDDING_DIM = 512 
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "IP"
NLIST_PARAM = 128 
NPROBE_PARAM = 16

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
                FieldSchema(name=FACE_ID_FIELD, dtype=DataType.VARCHAR, is_primary=False, max_length=255),
                FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name=PERSON_ID_FIELD, dtype=DataType.VARCHAR, max_length=255, default_value=DEFAULT_PERSON_ID),
                FieldSchema(name=PRIMARY_KEY_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True)
            ]
            schema = CollectionSchema(fields=fields, description="База данных эмбеддингов лиц")
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

def get_cluster_members(client: MilvusClient, person_ids: set[str]) -> dict:
    """Получает всех членов заданных кластеров (PersonID)."""
    cluster_members = {} 
    if not person_ids:
        return cluster_members

    logger.info(f"Получение всех членов для PersonIDs: {person_ids}")
    pids_list_str = ", ".join([f'\"{pid}\"' for pid in person_ids])
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
                 output_fields=[PRIMARY_KEY_FIELD, FACE_ID_FIELD, PERSON_ID_FIELD],
                 limit=limit 
            )
        
        logger.info(f"Извлечено {len(all_members_results)} записей членов кластера.")
        for member_entity in all_members_results:
            pid = member_entity[PERSON_ID_FIELD]
            member_data = {
                "pk": member_entity[PRIMARY_KEY_FIELD],
                "face_id": member_entity[FACE_ID_FIELD],
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
        collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
        
        num_entities = len(data_list)
        face_ids = [item[FACE_ID_FIELD] for item in data_list]
        embeddings = [item[EMBEDDING_FIELD] for item in data_list]
        person_ids = [item.get(PERSON_ID_FIELD, DEFAULT_PERSON_ID) for item in data_list]
        
        data_for_collection_insert = [face_ids, embeddings, person_ids]
        
        logger.debug(f"Попытка вставки {len(data_list)} записей через collection.insert...")
        res = collection.insert(data=data_for_collection_insert)
        logger.info(f"{len(data_list)} записей успешно вставлено. PKs: {res.primary_keys}")
        
        collection.flush()
        logger.debug("Выполнен flush коллекции.")
        
        return res.primary_keys
    except Exception as e:
        logger.error(f"Ошибка при вставке данных в Milvus: {e}", exc_info=True)
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
            output_fields=[PRIMARY_KEY_FIELD, FACE_ID_FIELD, PERSON_ID_FIELD]
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
        output_fields = [PRIMARY_KEY_FIELD, FACE_ID_FIELD, PERSON_ID_FIELD]
        
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
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            num_batches = (len(data_list) + batch_size - 1) // batch_size
            current_batch_num = i // batch_size + 1
            logger.debug(f"Upsert батча {current_batch_num} / {num_batches} (размер {len(batch)})...)")
            # Используем низкоуровневый Collection для upsert, т.к. MilvusClient его не имеет (на момент написания)
            # Если в будущем появится client.upsert(), можно перейти на него.
            collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
            res = collection.upsert(data=batch)
            # Проверка результата upsert может быть специфична для версии Milvus
            # logger.debug(f"Результат Upsert батча {current_batch_num}: {res}") 
            success = True # Хотя бы один батч прошел

        # Флашим после всех батчей
        collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
        collection.flush()
        logger.debug("Выполнен flush коллекции после Upsert.")
        logger.info(f"Upsert {len(data_list)} записей завершен.")
        return success
    except Exception as e:
        logger.error(f"Ошибка при выполнении Upsert в Milvus: {e}", exc_info=True)
        return False
