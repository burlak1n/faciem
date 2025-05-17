import os
import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from deepface import DeepFace
from loguru import logger
import sys
from typing import List, Dict, Any, Optional

# Импортируем все необходимые функции и константы из модуля milvus
from milvus import (
    setup_milvus, 
    insert_data,
    search_data,
    query_data,
    upsert_data,
    get_cluster_members,
    MilvusClient, # Импортируем для аннотации типа
    COLLECTION_NAME,
    EMBEDDING_FIELD,
    PERSON_ID_FIELD, # Тип изменен на INT64
    DEFAULT_PERSON_ID, # Значение изменено на 0
    PRIMARY_KEY_FIELD,
    PHOTO_ID_FIELD,
    FACE_INDEX_FIELD,
    DEFAULT_SEARCH_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD
)
# Настройка логгера (можно вынести в отдельный модуль логгирования)
LOG_FILE = "face_manager.log"
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE, rotation="5 MB", level="DEBUG")

class FaceManager:
    """
    Класс для управления базой данных лиц с использованием DeepFace и Milvus.
    """
    def __init__(self, model_name: str = "Facenet512", detector_backend: str = "retinaface"):
        """
        Инициализирует менеджер лиц.

        Args:
            model_name (str): Название модели DeepFace для извлечения эмбеддингов.
            detector_backend (str): Название детектора лиц DeepFace.
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.client: MilvusClient | None = setup_milvus() # setup_milvus использует URI из milvus.py

        if not self.client:
            logger.error("Не удалось инициализировать Milvus клиент. FaceManager не может работать.")
            # В реальном приложении здесь лучше выбросить исключение
            # raise ConnectionError("Failed to connect to Milvus via setup_milvus")
        else:
             logger.info(f"FaceManager инициализирован. Model: {self.model_name}, Detector: {self.detector_backend}, Milvus OK.")

    def _extract_embedding(self, image_path_or_url: str) -> List[Dict[str, Any]]:
        """
        Извлекает эмбеддинги для ВСЕХ лиц на изображении.
        Возвращает список словарей, каждый содержит 'embedding' и 'facial_area'.
        Если лица не найдены или произошла ошибка, возвращает пустой список.
        """
        try:
            # enforce_detection=False позволяет обрабатывать изображения без лиц (вернет пустой список)
            # align=True используется по умолчанию и рекомендуется
            embedding_objs = DeepFace.represent(
                img_path=image_path_or_url,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False, 
                align=True
            )
            DeepFace.verify
            # DeepFace.represent возвращает список словарей
            if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                # Убедимся, что каждый объект содержит нужный ключ 'embedding'
                valid_embeddings = [obj for obj in embedding_objs if isinstance(obj, dict) and 'embedding' in obj]
                if len(valid_embeddings) != len(embedding_objs):
                    logger.warning(f"Некоторые объекты, возвращенные DeepFace для {image_path_or_url}, не содержали ключ 'embedding'.")
                return valid_embeddings
            else:
                logger.info(f"Лица не найдены или не удалось извлечь эмбеддинги для: {image_path_or_url}")
                return []
        except Exception as e:
            logger.error(f"Ошибка при извлечении эмбеддинга для {image_path_or_url}: {e}", exc_info=True)
            return []

    def add_face(self, image_path_or_url: str, photo_id: int) -> Optional[int]:
        """
        Обнаруживает все лица на изображении, извлекает эмбеддинги и добавляет их в Milvus.
        Сохраняет photo_id и face_index для каждой записи.

        Args:
            image_path_or_url: Путь к локальному файлу или URL изображения.
            photo_id: ID фотографии из внешней системы (например, vk_faces.db).
                      Теперь это обязательный параметр.

        Returns:
            Количество успешно добавленных лиц, или None в случае ошибки Milvus.
        """
        embedding_objs = self._extract_embedding(image_path_or_url)
        if not embedding_objs:
            return 0 # Лица не найдены или ошибка извлечения, 0 добавлено

        data_to_insert = []
        added_count = 0
        
        # photo_id теперь всегда предоставляется и является int
        # photo_id_value = photo_id # Просто используем photo_id напрямую

        for i, obj in enumerate(embedding_objs):
            embedding = obj.get('embedding')
            if embedding is None:
                logger.warning(f"Пропущен объект {i} для {image_path_or_url} из-за отсутствия 'embedding'")
                continue
                
            data_to_insert.append({
                EMBEDDING_FIELD: embedding,
                PERSON_ID_FIELD: DEFAULT_PERSON_ID, # Теперь это 0 (int)
                PHOTO_ID_FIELD: photo_id, # Используем напрямую photo_id
                FACE_INDEX_FIELD: i 
            })
            added_count += 1

        if not data_to_insert:
            logger.warning(f"Нет данных для вставки в Milvus для {image_path_or_url}")
            return 0

        logger.debug(f"Подготовлено {len(data_to_insert)} записей для вставки в Milvus.")
        # Вызываем функцию вставки из milvus.py
        insert_result = insert_data(self.client, data_to_insert) # Передаем client и список словарей

        if insert_result is not None:
            actual_inserted_count = len(insert_result) # insert_result это список PK
            logger.info(f"Успешно добавлено {actual_inserted_count} лиц для {image_path_or_url} с photo_id={photo_id}, person_id={DEFAULT_PERSON_ID}")
            return actual_inserted_count 
        else:
            logger.error(f"Ошибка при вставке данных в Milvus для {image_path_or_url}")
            return None

    def search_face(self, query_image_path_or_url: str, top_k: int = DEFAULT_SEARCH_TOP_K, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Dict[str, Any]:
        """
        Ищет похожие лица в базе Milvus по изображению-запросу.

        Args:
            query_image_path_or_url: Локальный путь или URL изображения для поиска.
            top_k: Количество ближайших соседей для поиска.
            similarity_threshold: Порог схожести (косинусное расстояние) для отбора совпадений.

        Returns:
            Словарь с результатами:
            {
                "direct_hits": list[dict], # Список прямых совпадений выше порога
                "cluster_members": dict[str, list[dict]] # Словарь {person_id: [члены кластера]} для найденных PersonID
            }
            Возвращает пустые списки/словари в случае ошибки или отсутствия совпадений.
        """
        if not self.client:
            logger.error("Milvus client не инициализирован в search_face.")
            return {"direct_hits": [], "cluster_members": {}}

        query_embedding_objs = self._extract_embedding(query_image_path_or_url)
        if not query_embedding_objs:
            # Логгируем, если _extract_embedding не вернул объекты
            error_source_log = query_image_path_or_url if len(query_image_path_or_url) < 100 else query_image_path_or_url[:100] + "..."
            logger.info(f"Не удалось извлечь эмбеддинги из запроса (или нет лиц): '{error_source_log}'")
            return {"direct_hits": [], "cluster_members": {}}

        # Извлекаем непосредственно векторы для передачи в search_data
        actual_query_vectors = [obj['embedding'] for obj in query_embedding_objs]
        # _extract_embedding гарантирует, что каждый obj содержит ключ 'embedding'
        
        logger.info(f"Найдено {len(actual_query_vectors)} лиц в запросе. Поиск top {top_k}...")

        direct_hits = []
        found_person_ids = set()
        unique_direct_hit_pks = set() # Чтобы не дублировать результаты по PK

        try:
            # Передаем actual_query_vectors (список векторов) в search_data
            search_results = search_data(self.client, actual_query_vectors, top_k)

            for i, hits_list in enumerate(search_results):
                if not hits_list: continue
                for hit in hits_list:
                    similarity = hit.get('distance', 0.0)
                    entity = hit.get('entity')
                    if not entity: continue
                    
                    db_pk = entity.get(PRIMARY_KEY_FIELD)
                    if db_pk is None: continue 

                    if similarity >= similarity_threshold and db_pk not in unique_direct_hit_pks:
                        photo_id = entity.get(PHOTO_ID_FIELD, -1)
                        face_index = entity.get(FACE_INDEX_FIELD, -1)
                        person_id = entity.get(PERSON_ID_FIELD, DEFAULT_PERSON_ID) # Будет int (0 по умолчанию)
                        
                        direct_hits.append({
                            "pk": db_pk,
                            "photo_id": photo_id, 
                            "face_index": face_index, 
                            "person_id": person_id, # Теперь int
                            "similarity": similarity,
                            "query_face_idx": i 
                        })
                        unique_direct_hit_pks.add(db_pk)
                        logger.info(f"  -> Прямое совпадение: PK:{db_pk}, PhotoID:{photo_id}, FaceIdx:{face_index}, PersonID:{person_id}, Sim:{similarity:.4f} (лицо запроса #{i}) ")
                        if person_id != DEFAULT_PERSON_ID: # Сравниваем с 0
                            found_person_ids.add(person_id) # person_id теперь int
                    elif db_pk in unique_direct_hit_pks:
                         logger.debug(f"    Пропуск (уже найдено): PK:{db_pk}")

            direct_hits.sort(key=lambda x: x['similarity'], reverse=True)

            cluster_members_data = get_cluster_members(self.client, found_person_ids)

            logger.info(f"Итого найдено {len(direct_hits)} прямых совпадений (выше порога {similarity_threshold}) для запроса.")
            return {"direct_hits": direct_hits, "cluster_members": cluster_members_data}

        except Exception as e:
            logger.error(f"Ошибка при обработке результатов поиска: {e}", exc_info=True)
            return {"direct_hits": [], "cluster_members": {}}
            
    # --- Методы Кластеризации (перенесены из cluster_faces.py) ---

    def _get_all_data_for_clustering(self) -> list:
        """Извлекает все данные для полной перекластеризации."""
        if not self.client: return []
        logger.info(f"Полная кластеризация: извлечение всех данных...")
        all_entities_data = query_data(
            self.client, filter_expression="", 
            output_fields=[PRIMARY_KEY_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD, EMBEDDING_FIELD], limit=None
        )
        logger.info(f"Извлечено {len(all_entities_data)} записей.")
        return all_entities_data

    def _get_unclustered_faces(self) -> list:
        """Извлекает лица с person_id = DEFAULT_PERSON_ID."""
        if not self.client: return []
        logger.info(f"Инкрементальная кластеризация: извлечение некластеризованных лиц (PersonID={DEFAULT_PERSON_ID})...")
        unclustered_filter = f"{PERSON_ID_FIELD} == {DEFAULT_PERSON_ID}" # Сравнение с числом
        unclustered_faces_data = query_data(
            self.client, filter_expression=unclustered_filter,
            output_fields=[PRIMARY_KEY_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD, EMBEDDING_FIELD], limit=None
        )
        logger.info(f"Извлечено {len(unclustered_faces_data)} некластеризованных записей.")
        return unclustered_faces_data

    def _get_existing_cluster_centroids(self) -> dict:
        """Вычисляет центроиды существующих кластеров."""
        if not self.client: return {}
        logger.info(f"Инкрементальная кластеризация: извлечение данных существующих кластеров (PersonID!={DEFAULT_PERSON_ID})...")
        clustered_filter = f"{PERSON_ID_FIELD} != {DEFAULT_PERSON_ID}" # Сравнение с числом
        existing_clusters_raw = query_data(
            self.client, filter_expression=clustered_filter,
            output_fields=[PERSON_ID_FIELD, EMBEDDING_FIELD], limit=None
        )
        
        existing_centroids = {}
        if existing_clusters_raw:
            # ... (логика расчета центроидов осталась та же) ...
            unique_pids = set(d[PERSON_ID_FIELD] for d in existing_clusters_raw)
            logger.info(f"Расчет центроидов для {len(unique_pids)} существующих PersonID...")
            grouped_embeddings = {}
            for item in existing_clusters_raw:
                pid = item[PERSON_ID_FIELD]
                emb = item[EMBEDDING_FIELD]
                grouped_embeddings.setdefault(pid, []).append(emb)
            
            calculated_count = 0
            for pid, embs_list in grouped_embeddings.items():
                if embs_list:
                    try:
                        existing_centroids[pid] = np.mean(np.array(embs_list), axis=0)
                        calculated_count += 1
                    except Exception as e:
                         logger.warning(f"Ошибка расчета центроида для PersonID {pid}: {e}")
            logger.info(f"Рассчитано {calculated_count} центроидов.")
            
        return existing_centroids

    def _get_next_person_id(self, existing_person_ids: set[int]) -> int: # existing_person_ids теперь set[int]
        next_person_numeric_id = 1
        if existing_person_ids:
            # Поскольку DEFAULT_PERSON_ID (0) может быть в existing_person_ids если мы его не отфильтровали ранее,
            # или если кто-то вручную так установил, убедимся, что мы ищем максимум среди > 0.
            # Однако, обычно existing_person_ids приходят из _get_existing_cluster_centroids, где уже отфильтрованы != DEFAULT_PERSON_ID.
            max_existing_id = 0 
            for pid_int in existing_person_ids:
                if pid_int > max_existing_id: # Игнорируем DEFAULT_PERSON_ID (0) и отрицательные, если вдруг появятся
                    max_existing_id = pid_int
            next_person_numeric_id = max_existing_id + 1
        logger.info(f"Новые PersonID для кластеров начнутся с {next_person_numeric_id}.")
        return next_person_numeric_id

    def _perform_dbscan_and_assign_ids(self, entities_to_cluster: list, cluster_eps: float, 
                                        min_cluster_size: int, existing_person_ids: set[int],
                                        hdbscan_min_samples: Optional[int] = None) -> list:
        """Выполняет HDBSCAN и назначает PersonID."""
        if not entities_to_cluster: return []

        pks = [e[PRIMARY_KEY_FIELD] for e in entities_to_cluster]
        embeddings_list = [e[EMBEDDING_FIELD] for e in entities_to_cluster]
        embeddings_np = np.array(embeddings_list)

        hdbscan_params = {
            "min_cluster_size": min_cluster_size,
            # "metric": cosine_distances # <--- Старый вариант
        }
        if hdbscan_min_samples is not None:
            hdbscan_params["min_samples"] = hdbscan_min_samples

        # Обертка для cosine_distances, чтобы она принимала два 1D вектора и возвращала скаляр
        def custom_cosine_metric(u, v):
            # Преобразуем 1D векторы в 2D массивы (одна строка, N столбцов)
            u_2d = u.reshape(1, -1)
            v_2d = v.reshape(1, -1)
            # cosine_distances возвращает матрицу [[расстояние]], извлекаем скаляр
            return cosine_distances(u_2d, v_2d)[0][0]

        hdbscan_params["metric"] = custom_cosine_metric

        logger.info(f"Выполнение HDBSCAN (params={hdbscan_params}) для {len(embeddings_np)} векторов...")
        # HDBSCAN с precomputed метрикой ожидает квадратную матрицу расстояний.
        # Если мы передаем callable, он должен принимать два 1D-массива и возвращать float.
        # Поэтому используем embeddings_np напрямую, а HDBSCAN будет вызывать custom_cosine_metric для пар.
        
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        labels = clusterer.fit_predict(embeddings_np) # Передаем исходные эмбеддинги

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = np.sum(labels == -1)
        logger.info(f"Кластеризация завершена. Найдено новых кластеров: {num_clusters}, шумовых точек: {num_noise}")

        upsert_data_list = [] # Переименовал, чтобы не конфликтовать с импортом upsert_data
        person_id_map = {} 
        next_person_numeric_id = self._get_next_person_id(existing_person_ids)

        for i, label in enumerate(labels):
            person_id_to_upsert = DEFAULT_PERSON_ID # Теперь это 0
            if label != -1: 
                if label not in person_id_map:
                    person_id_map[label] = next_person_numeric_id
                    next_person_numeric_id += 1
                person_id_to_upsert = person_id_map[label] # Это уже int
            
            upsert_data_list.append({
                PRIMARY_KEY_FIELD: pks[i],
                PERSON_ID_FIELD: person_id_to_upsert 
            })
        return upsert_data_list

    def _assign_faces_to_existing_clusters(self, unclustered_faces: list, existing_centroids: dict, 
                                            cluster_eps: float) -> tuple[list, list]:
        """Пытается присвоить лица существующим кластерам."""
        # ... (логика присвоения по близости к центроидам осталась та же) ...
        assigned_data_for_upsert = []
        remaining_faces_for_dbscan = []
        processed_pks = set()

        if not existing_centroids or not unclustered_faces:
            return [], unclustered_faces 

        logger.info(f"Инкрементальный режим: попытка присвоить {len(unclustered_faces)} лиц к {len(existing_centroids)} кластерам (eps={cluster_eps})...")
        
        for face_data in unclustered_faces:
            pk = face_data[PRIMARY_KEY_FIELD]
            if pk in processed_pks: continue

            current_embedding_np = np.array(face_data[EMBEDDING_FIELD]).reshape(1, -1)
            min_dist = float('inf')
            assigned_pid = None

            for pid, centroid_np in existing_centroids.items():
                try:
                     if centroid_np.ndim == 1: centroid_np_reshaped = centroid_np.reshape(1, -1)
                     else: centroid_np_reshaped = centroid_np
                     dist = cosine_distances(current_embedding_np, centroid_np_reshaped)[0][0]
                     if dist < min_dist:
                        min_dist = dist
                        assigned_pid = pid
                except ValueError as e:
                     logger.warning(f"Ошибка сравнения PK {pk} с центроидом PersonID {pid}: {e}.")
                     continue 

            if assigned_pid is not None and min_dist < cluster_eps:
                logger.debug(f"Лицо PK {pk} присвоено PersonID {assigned_pid} (расстояние: {min_dist:.4f})")
                assigned_data_for_upsert.append({
                    PRIMARY_KEY_FIELD: pk,
                    PERSON_ID_FIELD: assigned_pid # assigned_pid - это int
                })
                processed_pks.add(pk)
            else:
                remaining_faces_for_dbscan.append(face_data)
                processed_pks.add(pk) 

        logger.info(f"{len(assigned_data_for_upsert)} лиц присвоено существующим. {len(remaining_faces_for_dbscan)} осталось для DBSCAN.")
        return assigned_data_for_upsert, remaining_faces_for_dbscan

    def cluster_faces(self, mode: str, cluster_eps: float, min_cluster_size: int, hdbscan_min_samples: Optional[int] = None):
        """
        Выполняет кластеризацию лиц в базе данных Milvus.

        Args:
            mode: Режим кластеризации ('full' или 'incremental').
            cluster_eps: Порог для инкрементального присвоения к существующим кластерам.
            min_cluster_size: Минимальный размер кластера для HDBSCAN.
            hdbscan_min_samples: Параметр min_samples для HDBSCAN (влияет на обработку шума).
        """
        if not self.client:
            logger.error("Milvus client не инициализирован в cluster_faces.")
            return

        logger.info(f"Запуск кластеризации лиц: режим={mode.upper()}, eps (для присвоения)={cluster_eps}, min_cluster_size={min_cluster_size}, hdbscan_min_samples={hdbscan_min_samples}")
        
        all_upsert_data = [] 

        try:
            if mode == "full":
                all_entities_data = self._get_all_data_for_clustering()
                if not all_entities_data:
                    logger.info("Нет данных для полной кластеризации.")
                    return
                all_upsert_data = self._perform_dbscan_and_assign_ids(
                    all_entities_data, cluster_eps, min_cluster_size, set(), hdbscan_min_samples=hdbscan_min_samples
                )

            elif mode == "incremental":
                unclustered_faces = self._get_unclustered_faces()
                existing_centroids = self._get_existing_cluster_centroids()
                
                assigned_data, remaining_for_dbscan = self._assign_faces_to_existing_clusters(
                    unclustered_faces, existing_centroids, cluster_eps
                )
                all_upsert_data.extend(assigned_data)

                if remaining_for_dbscan:
                    logger.info(f"Инкрементальный режим: кластеризация {len(remaining_for_dbscan)} оставшихся лиц с HDBSCAN...")
                    existing_pid_strings = set(existing_centroids.keys())
                    newly_clustered_data = self._perform_dbscan_and_assign_ids(
                        remaining_for_dbscan, cluster_eps, min_cluster_size, existing_pid_strings, hdbscan_min_samples=hdbscan_min_samples
                    )
                    all_upsert_data.extend(newly_clustered_data)
                else:
                     logger.info("Инкрементальный режим: не осталось лиц для DBSCAN.")

            else:
                logger.error(f"Неизвестный режим кластеризации: {mode}")
                return

            # Обновление данных в Milvus
            if all_upsert_data:
                success = upsert_data(self.client, all_upsert_data) # upsert_data из модуля milvus
                if success:
                    logger.info("Обновление PersonID в Milvus успешно завершено.")
                else:
                     logger.error("Ошибка во время обновления PersonID в Milvus.")
            else:
                logger.info("Нет данных для обновления PersonID.")

        except Exception as e:
            logger.error(f"Общая ошибка в процессе кластеризации (режим {mode}): {e}", exc_info=True)

        finally:
            logger.info(f"Процесс кластеризации (режим {mode}) завершен.") 