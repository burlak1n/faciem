import os
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
)
from deepface import DeepFace
from loguru import logger
import glob
import argparse # Добавляем argparse

# --- Настройка Логгера ---
logger.add("face_database.log", rotation="5 MB")

# --- Убедимся, что константы доступны перед функциями ---
# (Повторное определение для надежности, хотя должно быть доступно из модуля)
COLLECTION_NAME = "faces"
EMBEDDING_FIELD = "embedding"
FACE_ID_FIELD = "face_id"
FACE_ID_FIELD = "face_id"
PERSON_ID_FIELD = "person_id"
PRIMARY_KEY_FIELD = "pk"
DEFAULT_PERSON_ID = "-1"
METRIC_TYPE = "IP"
SIMILARITY_THRESHOLD = 0.54
DEEPFACE_MODEL = "Facenet512"
DEEPFACE_DETECTOR = "retinaface"
EMBEDDING_DIM = 512

# --- Константы ---
MILVUS_ALIAS = "default"
MILVUS_URI = "./milvus_lite.db" # Файл для хранения данных Milvus Lite
INDEX_TYPE = "IVF_FLAT" # Алгоритм индексации для быстрого поиска (HNSW не поддерживается в Lite)

# --- Функции Milvus ---

def connect_to_milvus():
    """Устанавливает соединение с Milvus Lite."""
    try:
        # Используем MilvusClient для упрощения
        client = MilvusClient(uri=MILVUS_URI)
        logger.info(f"Успешное подключение к Milvus Lite: {MILVUS_URI}")
        # Для некоторых операций может понадобиться низкоуровневое API
        if not connections.has_connection(MILVUS_ALIAS):
             connections.connect(alias=MILVUS_ALIAS, uri=MILVUS_URI)
             logger.info(f"Низкоуровневое соединение '{MILVUS_ALIAS}' установлено.")
        return client
    except Exception as e:
        logger.error(f"Ошибка подключения к Milvus Lite: {e}")
        raise

def create_milvus_collection(client: MilvusClient):
    """Создает коллекцию Milvus, если она не существует."""
    try:
        has_collection = client.has_collection(collection_name=COLLECTION_NAME)
        # collection_low_level = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS) # Убираем отсюда

        if not has_collection:
            logger.info(f"Коллекция '{COLLECTION_NAME}' не найдена. Создание...")
            # Определяем поля
            fields = [
                # Milvus Lite автоматически создает id поле (int64, auto_id=True),
                # но мы добавим свое строковое ID для удобства
                FieldSchema(name=FACE_ID_FIELD, dtype=DataType.VARCHAR, is_primary=False, max_length=255),
                FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                # Возвращаем person_id (VARCHAR, default "-1") перед pk
                FieldSchema(name="person_id", dtype=DataType.VARCHAR, max_length=255, default_value="-1"),
                # pk теперь последний
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
            ]
            schema = CollectionSchema(fields=fields, description="База данных эмбеддингов лиц с ID личности (pk последний)")
            client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
            logger.info(f"Коллекция '{COLLECTION_NAME}' успешно создана.")

            # Получаем объект коллекции ПОСЛЕ ее создания
            collection_low_level = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)

            # Создаем индекс для векторного поля
            logger.info(f"Создание индекса '{INDEX_TYPE}' для поля '{EMBEDDING_FIELD}'...")
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name=EMBEDDING_FIELD,
                index_name="vector_index", # Задаем имя индекса
                index_type=INDEX_TYPE,
                metric_type=METRIC_TYPE,
                params={"nlist": 128} # Параметры для IVF_FLAT
            )
            client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
            # Проверяем, создался ли индекс по имени
            has_index_check = collection_low_level.has_index(index_name="vector_index")
            logger.info(f"Индекс 'vector_index' для поля '{EMBEDDING_FIELD}' в новой '{COLLECTION_NAME}' успешно создан? {has_index_check}")
            if not has_index_check:
                 logger.warning(f"Не удалось подтвердить создание индекса 'vector_index' для новой '{COLLECTION_NAME}'")

        else:
            logger.info(f"Коллекция '{COLLECTION_NAME}' уже существует.")
            # Получаем объект коллекции, т.к. она уже существует
            collection_low_level = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)

            # Проверяем индекс по имени, если коллекция уже существовала
            has_index_check = collection_low_level.has_index(index_name="vector_index")
            logger.info(f"Проверка индекса 'vector_index' для существующей коллекции '{COLLECTION_NAME}': Индекс есть? {has_index_check}")
            if not has_index_check:
                logger.warning(f"Индекс 'vector_index' для '{COLLECTION_NAME}' не найден. Попытка создать...")
                try:
                    index_params = client.prepare_index_params()
                    index_params.add_index(
                        field_name=EMBEDDING_FIELD,
                        index_name="vector_index", # Задаем имя индекса
                        index_type=INDEX_TYPE,
                        metric_type=METRIC_TYPE,
                        params={"nlist": 128}
                    )
                    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
                    has_index_check = collection_low_level.has_index(index_name="vector_index")
                    logger.info(f"Повторная проверка после создания индекса 'vector_index': Индекс есть? {has_index_check}")
                except Exception as index_e:
                    logger.error(f"Ошибка при попытке создать индекс 'vector_index' для существующей коллекции: {index_e}")

        # Проверяем и загружаем коллекцию в память для поиска (после проверки/создания индекса)
        # Убедимся, что collection_low_level определен в обоих ветках if/else
        if 'collection_low_level' not in locals():
             logger.error(f"Не удалось получить объект Collection для '{COLLECTION_NAME}'. Прерывание.")
             raise Exception(f"Не удалось получить объект Collection для {COLLECTION_NAME}")

        load_state = client.get_load_state(collection_name=COLLECTION_NAME)
        logger.info(f"Текущее состояние загрузки коллекции '{COLLECTION_NAME}': {load_state}")
        if load_state != "loaded":
            logger.info(f"Попытка загрузки коллекции '{COLLECTION_NAME}' в память...")
            try:
                collection_low_level.load()
                logger.info(f"Коллекция '{COLLECTION_NAME}' успешно загружена в память.")
            except Exception as load_e:
                logger.error(f"!!! Ошибка непосредственно при вызове collection.load(): {load_e}")
                # Перевыбрасываем ошибку, чтобы остановить выполнение, если load не удался
                raise load_e
        else:
            logger.info(f"Коллекция '{COLLECTION_NAME}' уже была загружена.")

    except Exception as e:
        logger.error(f"Ошибка при создании/проверке/загрузке коллекции Milvus: {e}")


# --- Функции DeepFace и Базы Данных ---

def extract_embedding(image_path: str) -> list[list[float]]:
    """Извлекает все эмбеддинги лиц из изображения с помощью DeepFace."""
    embeddings = []
    try:
        # enforce_detection=False, чтобы не падать, если лиц нет, но мы проверим ниже
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False, # Не падать, если лиц нет
            align=True
        )
        if not embedding_objs:
            logger.warning(f"Лица не найдены на изображении: {image_path}")
            return [] # Возвращаем пустой список

        logger.debug(f"Найдено {len(embedding_objs)} лиц на {image_path}")
        for i, obj in enumerate(embedding_objs):
            embedding = obj['embedding']
            # Нормализация L2 (важно для метрики IP/Cosine Similarity)
            embedding_np = np.array(embedding)
            norm = np.linalg.norm(embedding_np)
            if norm == 0:
               logger.warning(f"Нулевой вектор эмбеддинга для лица #{i} на {image_path}. Пропуск.")
               continue # Пропускаем этот эмбеддинг

            normalized_embedding = embedding_np / norm
            embeddings.append(normalized_embedding.tolist()) # Добавляем как список для Milvus

        return embeddings

    except Exception as e:
        # Обработка других возможных ошибок DeepFace
        logger.error(f"Ошибка при извлечении эмбеддингов из {image_path}: {e}")
        return []


def add_face_to_db(client: MilvusClient, image_path: str, base_face_id: str | None = None):
    """Извлекает все эмбеддинги с изображения и добавляет их в Milvus."""
    if base_face_id is None:
        base_face_id = os.path.basename(image_path) # Используем имя файла как базовый ID

    embeddings = extract_embedding(image_path)
    if not embeddings:
        logger.info(f"Не найдено лиц для добавления из {image_path}.")
        return # Ошибка или лица не найдены

    # !!!!! ВРЕМЕННЫЙ ТЕСТ: ПРОПУСК ВСТАВКИ В MILVUS !!!!!
    # logger.warning(f"ВРЕМЕННЫЙ ТЕСТ: Пропуск фактической вставки в Milvus для {image_path}")
    # return 
    # !!!!! КОНЕЦ ВРЕМЕННОГО ТЕСТА !!!!!

    try:
        data_to_insert_dicts = []
        for i, embedding in enumerate(embeddings):
            face_id = f"{base_face_id}_{i}" # Генерируем уникальный ID для каждого лица
            data_to_insert_dicts.append(
                {FACE_ID_FIELD: face_id, EMBEDDING_FIELD: embedding}
            )
            logger.debug(f"Подготовлено к добавлению (dict): ID={face_id}, Embedding[0:5]={embedding[:5]}...")

        if not data_to_insert_dicts:
             logger.warning(f"Нет валидных эмбеддингов для добавления из {image_path}")
             return

        # Преобразуем данные для Collection.insert()
        # Ожидается: список списков по полям [ [face_id1,...], [emb1,...], [person_id1,...] ]
        # Порядок полей должен соответствовать схеме: FACE_ID_FIELD, EMBEDDING_FIELD, person_id (pk последний и auto_id)
        num_entities = len(data_to_insert_dicts)
        face_ids_list = [item[FACE_ID_FIELD] for item in data_to_insert_dicts]
        embeddings_list = [item[EMBEDDING_FIELD] for item in data_to_insert_dicts]
        # Передаем значения по умолчанию для person_id, так как поле есть в схеме
        person_ids_dummy_list = ["-1"] * num_entities 
        # data_for_collection_insert = [face_ids_list, embeddings_list] # Убрали pk, добавили person_id
        data_for_collection_insert = [face_ids_list, embeddings_list, person_ids_dummy_list]

        logger.debug(f"Попытка вызова collection.insert() для {len(data_to_insert_dicts)} лиц из '{base_face_id}' (низкоуровневый API)...")
        collection_low_level = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
        # Используем низкоуровневый insert
        res = collection_low_level.insert(data=data_for_collection_insert)
        logger.debug(f"Вызов collection.insert() завершен. Результат: {res}") # Посмотрим на результат
        # Убираем попытку логирования PK, т.к. res не имеет атрибута primary_keys в Lite
        # Логирование PK теперь в res.primary_keys (если используется Collection.insert)
        logger.info(f"{len(data_to_insert_dicts)} лиц(а) из '{base_face_id}' успешно добавлены в '{COLLECTION_NAME}'. PKs: {res.primary_keys}")

        # Рекомендуется периодически вызывать flush для гарантии записи на диск в Milvus Lite
        # collection_low_level уже получен выше
        logger.debug(f"Попытка вызова collection.flush() для '{COLLECTION_NAME}'...")
        collection_low_level.flush()
        logger.debug(f"Вызов collection.flush() завершен.")
    except Exception as e:
        logger.error(f"Ошибка при добавлении лиц из '{base_face_id}' в Milvus: {e}")


def search_face_in_db(client: MilvusClient, query_image_path: str, top_k: int = 5) -> dict:
    """
    Ищет в базе лица, похожие на *все* лица с изображения запроса.
    Возвращает словарь с:
    - 'direct_hits': список словарей похожих лиц выше порога (pk, id, person_id, similarity, query_face_idx)
    - 'cluster_members': словарь {person_id: [список словарей всех членов кластера (pk, face_id, person_id)]}
    """
    query_embeddings = extract_embedding(query_image_path)
    if not query_embeddings:
        logger.error(f"Не удалось извлечь эмбеддинги из файла запроса: {query_image_path}")
        return {"direct_hits": [], "cluster_members": {}} # Возвращаем пустую структуру

    logger.info(f"Найдено {len(query_embeddings)} лиц(а) на изображении запроса {query_image_path}. Поиск {top_k} ближайших для каждого...")

    direct_hits = []
    found_person_ids = set() # Уникальные ID найденных личностей
    unique_direct_hit_pks = set() # Чтобы избежать дубликатов одного и того же лица из базы в direct_hits

    try:
        search_params_inner = {"nprobe": 16} 
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=query_embeddings,
            anns_field=EMBEDDING_FIELD,
            metric_type=METRIC_TYPE,
            params=search_params_inner,
            limit=top_k,
            output_fields=[PRIMARY_KEY_FIELD, FACE_ID_FIELD, PERSON_ID_FIELD] # Включаем PK
        )

        for i, hits in enumerate(results):
             logger.debug(f"Результаты для лица #{i} из запроса:")
             if not hits: continue

             for hit in hits:
                 similarity = hit['distance']
                 db_pk = hit['entity'][PRIMARY_KEY_FIELD] # Используем PK для уникальности
                 face_id = hit['entity'][FACE_ID_FIELD]
                 person_id = hit['entity'][PERSON_ID_FIELD]
                 logger.debug(f"  - Кандидат: PK: {db_pk}, ID: {face_id}, PersonID: {person_id}, Схожесть: {similarity:.4f}")

                 if similarity >= SIMILARITY_THRESHOLD:
                     if db_pk not in unique_direct_hit_pks:
                         direct_hits.append({
                             "pk": db_pk, # Добавляем PK
                             "id": face_id,
                             "person_id": person_id,
                             "similarity": similarity,
                             "query_face_idx": i
                         })
                         unique_direct_hit_pks.add(db_pk)
                         logger.info(f"    -> Добавлено в direct_hits (найдено по лицу запроса #{i}) PK: {db_pk}")
                         if person_id != DEFAULT_PERSON_ID:
                             found_person_ids.add(person_id)
                     else:
                          logger.debug(f"    -> Пропущено (уже в direct_hits по PK {db_pk})")
                 else:
                      logger.debug(f"    (Схожесть ниже порога {SIMILARITY_THRESHOLD})")

        # Сортируем прямые попадания по схожести
        direct_hits.sort(key=lambda x: x['similarity'], reverse=True)

        # Теперь получаем всех членов кластеров для найденных person_id
        cluster_members = {} # {person_id: [member1_dict, member2_dict, ...]}
        if found_person_ids:
            logger.info(f"Найдены прямые совпадения для PersonIDs: {found_person_ids}. Получение всех членов этих кластеров...")
            # Формируем фильтр для Milvus: person_id in ["id1", "id2", ...]
            # Важно: кавычки вокруг строковых ID
            pids_list_str = ", ".join([f'\"{pid}\"' for pid in found_person_ids])
            person_id_filter = f'{PERSON_ID_FIELD} in [{pids_list_str}]'
            logger.debug(f"Фильтр для запроса членов кластера: {person_id_filter}")

            try:
                # Запрашиваем всех членов кластера
                # Узнаем общее количество, чтобы получить все
                total_members_count = client.query(collection_name=COLLECTION_NAME, filter=person_id_filter, output_fields=["count(*)"])[0]["count(*)"]
                logger.debug(f"Ожидаемое количество членов кластера: {total_members_count}")
                
                all_members_results = []
                if total_members_count > 0:
                    # Получаем все записи батчами или одним запросом, если их немного
                    # Предполагаем < 16384 записей (лимит Milvus) для простоты
                    all_members_results = client.query(
                         collection_name=COLLECTION_NAME,
                         filter=person_id_filter,
                         output_fields=[PRIMARY_KEY_FIELD, FACE_ID_FIELD, PERSON_ID_FIELD],
                         limit=min(total_members_count, 16383) # Ограничиваем на всякий случай
                    )
                    # В реальном приложении здесь нужна итерация с offset, если total_members_count > 16383
                
                logger.info(f"Извлечено {len(all_members_results)} записей членов кластера.")

                # Группируем по person_id
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

        logger.info(f"\nИтого найдено {len(direct_hits)} прямых совпадений выше порога для {query_image_path}.")
        return {"direct_hits": direct_hits, "cluster_members": cluster_members}

    except Exception as e:
        logger.error(f"Ошибка при поиске лиц в Milvus для {query_image_path}: {e}", exc_info=True)
        return {"direct_hits": [], "cluster_members": {}} # Возвращаем пустую структуру

# --- Основной блок ---
if __name__ == "__main__":
    # --- Обработка аргументов командной строки ---
    parser = argparse.ArgumentParser(description="Управление базой данных лиц Milvus.")
    parser.add_argument("--add", action="store_true", help="Добавить лица из папки data/database_faces.")
    parser.add_argument("--search", action="store_true", help="Искать лица из папки data/query_face.")
    args = parser.parse_args()

    # Поведение по умолчанию: если ни один флаг не указан, выполняем поиск
    if not args.add and not args.search:
        logger.info("Не указаны флаги --add или --search. Выполняется только поиск по умолчанию.")
        args.search = True
    # --- Конец обработки аргументов ---

    logger.info("Запуск скрипта управления базой данных лиц...")

    milvus_client = connect_to_milvus()
    create_milvus_collection(milvus_client)

    # --- Блок добавления лиц (выполняется только с флагом --add) ---
    if args.add:
        logger.info("\n=== ЭТАП ДОБАВЛЕНИЯ ЛИЦ ===")
        # Убедитесь, что папка data/database_faces существует и содержит изображения лиц
        database_image_dir = "data/database_faces"
        if os.path.exists(database_image_dir):
            logger.info(f"Добавление лиц из папки: {database_image_dir}")
            image_files = glob.glob(os.path.join(database_image_dir, "*.[jJ][pP][gG]")) + \
                          glob.glob(os.path.join(database_image_dir, "*.[jJ][pP][eE][gG]")) + \
                          glob.glob(os.path.join(database_image_dir, "*.[pP][nN][gG]"))

            # Упрощенный вариант: просто добавляем все найденные лица
            # Проверку на существующие ID убрали, т.к. face_id теперь включает индекс _0, _1 и т.д.
            # В реальном приложении может понадобиться более сложная логика для обновления/пропуска.
            added_files_count = 0
            total_faces_added = 0
            for img_path in image_files:
                 logger.info(f"Обработка файла для добавления: {img_path}")
                 # add_face_to_db теперь сама извлекает и добавляет все лица
                 # Мы можем захотеть узнать, сколько лиц было добавлено из файла
                 # (add_face_to_db логирует это, но не возвращает)
                 add_face_to_db(milvus_client, img_path)
                 added_files_count += 1
                 # Примечание: мы не знаем точно, сколько лиц добавлено без парсинга логов или модификации add_face_to_db

            logger.info(f"Обработано {added_files_count} файлов для добавления.")
        else:
            logger.warning(f"Папка для добавления лиц не найдена: {database_image_dir}. Пропуск добавления.")
            logger.warning("Создайте папку 'data/database_faces' и поместите туда изображения для базы.")
    # --- Конец блока добавления лиц ---

    # --- Блок поиска лиц (выполняется только с флагом --search или по умолчанию) ---
    if args.search:
        logger.info("\n=== ЭТАП ПОИСКА ЛИЦ ===")
        # Ищем все изображения в папке data/query_face/
        query_image_dir = "data/query_face"
        if os.path.exists(query_image_dir):
            logger.info(f"\n--- Поиск лиц для всех изображений в: {query_image_dir} ---")
            query_files = glob.glob(os.path.join(query_image_dir, "*.[jJ][pP][gG]")) + \
                          glob.glob(os.path.join(query_image_dir, "*.[jJ][pP][eE][gG]")) + \
                          glob.glob(os.path.join(query_image_dir, "*.[pP][nN][gG]"))

            if not query_files:
                logger.warning(f"Не найдено файлов изображений в {query_image_dir}.")
            else:
                # Словарь для агрегации результатов по PersonID
                # Структура: { person_id: {"direct_hits": [], "all_member_face_ids": set()} }
                final_person_details = {}
                total_queries_processed = 0

                for query_image in query_files:
                    logger.info(f"\n--- Обработка файла запроса: {query_image} ---")
                    search_output = search_face_in_db(milvus_client, query_image, top_k=10)
                    total_queries_processed += 1

                    direct_hits_for_query = search_output.get("direct_hits", [])
                    cluster_members_from_query = search_output.get("cluster_members", {})

                    if not direct_hits_for_query:
                        logger.info(f"  Для {query_image} не найдено прямых совпадений выше порога.")
                    
                    # Обрабатываем прямые совпадения
                    for hit in direct_hits_for_query:
                        pid = hit["person_id"]
                        if pid == DEFAULT_PERSON_ID:
                            logger.debug(f"  Прямое совпадение {hit['id']} (PK: {hit['pk']}) не имеет PersonID. Пропуск.")
                            continue
                        
                        final_person_details.setdefault(pid, {"direct_hits": [], "all_member_face_ids": set()})
                        final_person_details[pid]["direct_hits"].append({
                            "found_face_id": hit["id"],
                            "pk": hit["pk"],
                            "similarity": hit["similarity"],
                            "query_file": os.path.basename(query_image),
                            "query_face_idx": hit["query_face_idx"]
                        })

                    # Обрабатываем всех членов кластера, затронутых этим запросом
                    for pid, members_list in cluster_members_from_query.items():
                        if pid == DEFAULT_PERSON_ID: continue
                        final_person_details.setdefault(pid, {"direct_hits": [], "all_member_face_ids": set()})
                        for member in members_list:
                            final_person_details[pid]["all_member_face_ids"].add(member["face_id"])

                # --- НОВЫЙ ВЫВОД: Детализация по Person ID ---
                logger.info(f"\n--- Итоговый отчет по найденным личностям ({total_queries_processed} запросов обработано) ---")

                if not final_person_details:
                    logger.info("Не найдено ни одной идентифицированной личности (person_id != '-1') по результатам поиска.")
                else:
                    try:
                        sorted_person_ids = sorted(final_person_details.keys(), key=int)
                    except ValueError:
                        sorted_person_ids = sorted(final_person_details.keys())

                    logger.info(f"Найдено {len(final_person_details)} уникальных личностей:")
                    for person_id_key in sorted_person_ids:
                        data = final_person_details[person_id_key]
                        logger.info(f"\n=== PersonID: {person_id_key} ===")
                        
                        if data["direct_hits"]:
                            logger.info("  Прямые совпадения (выше порога):")
                            sorted_direct_hits = sorted(data["direct_hits"], key=lambda x: (x["query_file"], -x["similarity"]))
                            for hit_info in sorted_direct_hits:
                                # Убираем суффикс _N из face_id перед выводом
                                parts = hit_info['found_face_id'].rsplit('_', 1)
                                base_face_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else hit_info['found_face_id']
                                logger.info(f"    - Найдено в базе: {base_face_id} (PK: {hit_info['pk']}) / Схожесть: {hit_info['similarity']:.4f} / Запрос: {hit_info['query_file']} (лицо #{hit_info['query_face_idx']})")
                        else:
                            logger.info("  Прямых совпадений (выше порога) для этого PersonID не найдено (возможно, найден через кластер другого лица).")

                        if data["all_member_face_ids"]:
                            logger.info("  Все известные лица этого человека в базе (полный кластер):")
                            # Используем set для уникальности базовых имен
                            unique_base_member_ids = set()
                            for member_face_id in sorted(list(data["all_member_face_ids"])):
                                # Убираем суффикс _N из face_id перед добавлением в set и выводом
                                parts = member_face_id.rsplit('_', 1)
                                base_member_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else member_face_id
                                unique_base_member_ids.add(base_member_id)
                            # Выводим уникальные базовые имена
                            for base_id in sorted(list(unique_base_member_ids)):
                                logger.info(f"    - {base_id}")
                        else:
                             logger.warning(f"  Не найдено информации о членах кластера для PersonID: {person_id_key} (это странно, если были прямые совпадения)")
        else:
             logger.warning(f"Папка для поиска не найдена: {query_image_dir}. Пропуск поиска.")
             logger.warning(f"Создайте папку '{query_image_dir}' и поместите туда изображения для поиска.")
    # --- Конец блока поиска лиц ---

    # --- Отключение ---
    # Низкоуровневое соединение закрывается автоматически при выходе, но можно и явно
    # if connections.has_connection(MILVUS_ALIAS):
    #     connections.disconnect(MILVUS_ALIAS)
    #     logger.info(f"Низкоуровневое соединение '{MILVUS_ALIAS}' закрыто.")
    # MilvusClient не требует явного disconnect

    logger.info("Скрипт завершил работу.") 