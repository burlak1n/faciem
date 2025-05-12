import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from deepface import DeepFace
from loguru import logger
import sys

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
    FACE_ID_FIELD,
    EMBEDDING_FIELD,
    PERSON_ID_FIELD,
    DEFAULT_PERSON_ID,
    PRIMARY_KEY_FIELD
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

    def _extract_embedding(self, image_path_or_url: str) -> list[list[float]]:
        """
        Извлекает и нормализует все эмбеддинги лиц из изображения. (Приватный метод)

        Args:
            image_path_or_url: Локальный путь или URL изображения.

        Returns:
            Список нормализованных эмбеддингов или пустой список.
        """
        if not self.client:
            logger.error("Milvus client не инициализирован в _extract_embedding.")
            return []
            
        embeddings = []
        try:
            # Используем параметры из self
            embedding_objs = DeepFace.represent(
                img_path=image_path_or_url,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            if not embedding_objs:
                logger.warning(f"Лица не найдены на: {image_path_or_url}")
                return []

            logger.debug(f"Найдено {len(embedding_objs)} лиц на {image_path_or_url}")
            for i, obj in enumerate(embedding_objs):
                embedding = obj.get('embedding')
                if not embedding:
                     logger.warning(f"Отсутствует 'embedding' для лица #{i} на {image_path_or_url}. Пропуск.")
                     continue
                
                embedding_np = np.array(embedding)
                norm = np.linalg.norm(embedding_np)
                if norm == 0:
                   logger.warning(f"Нулевой вектор эмбеддинга для лица #{i} на {image_path_or_url}. Пропуск.")
                   continue
                normalized_embedding = embedding_np / norm
                embeddings.append(normalized_embedding.tolist())
            return embeddings
        except Exception as e:
            # Логируем URL/путь для отладки
            error_source = image_path_or_url if len(image_path_or_url) < 100 else image_path_or_url[:100] + "..."
            logger.error(f"Ошибка при извлечении эмбеддингов из '{error_source}': {e}", exc_info=True)
            return []

    def add_face(self, image_path_or_url: str, base_face_id: str | None = None) -> list | None:
        """
        Извлекает эмбеддинги из изображения и добавляет их в базу Milvus.

        Args:
            image_path_or_url: Локальный путь или URL изображения.
            base_face_id: Базовый ID для лица (обычно имя файла). Если None, используется имя файла из пути.

        Returns:
            Список PK добавленных записей или None в случае ошибки Milvus.
        """
        if not self.client:
            logger.error("Milvus client не инициализирован в add_face.")
            return None

        if base_face_id is None:
            if '/' in image_path_or_url or '\\' in image_path_or_url: # Проверяем, похоже ли на путь
                base_face_id = os.path.basename(image_path_or_url)
            else: # Иначе считаем, что это URL или просто ID
                 base_face_id = "image" # Нужен какой-то ID по умолчанию

        embeddings = self._extract_embedding(image_path_or_url)
        if not embeddings:
            logger.info(f"Не найдено валидных лиц для добавления из {base_face_id}.")
            return [] # Возвращаем пустой список, т.к. ошибки не было, просто нечего добавлять

        data_to_insert = []
        for i, embedding in enumerate(embeddings):
            face_id = f"{base_face_id}_{i}"
            data_to_insert.append({
                FACE_ID_FIELD: face_id,
                EMBEDDING_FIELD: embedding,
                # PERSON_ID_FIELD будет установлен по умолчанию в insert_data (через collection.insert)
            })
            logger.debug(f"Подготовлено к добавлению: ID={face_id}")

        if data_to_insert:
            # Используем функцию insert_data из модуля milvus
            pks = insert_data(self.client, data_to_insert) # insert_data сама логирует успех/ошибку
            return pks
        else:
            logger.warning(f"Нет данных для вставки для {base_face_id}.")
            return []

    def search_face(self, query_image_path_or_url: str, top_k: int, similarity_threshold: float) -> dict:
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

        query_embeddings = self._extract_embedding(query_image_path_or_url)
        if not query_embeddings:
            error_source = query_image_path_or_url if len(query_image_path_or_url) < 100 else query_image_path_or_url[:100] + "..."
            logger.error(f"Не удалось извлечь эмбеддинги из запроса: '{error_source}'")
            return {"direct_hits": [], "cluster_members": {}}

        logger.info(f"Найдено {len(query_embeddings)} лиц в запросе. Поиск top {top_k}...")

        direct_hits = []
        found_person_ids = set()
        unique_direct_hit_pks = set() # Чтобы не дублировать результаты по PK

        try:
            # Используем search_data из модуля milvus
            search_results = search_data(self.client, query_embeddings, top_k)

            for i, hits_list in enumerate(search_results):
                if not hits_list: continue
                for hit in hits_list:
                    similarity = hit.get('distance', 0.0)
                    entity = hit.get('entity')
                    if not entity: continue
                    
                    db_pk = entity.get(PRIMARY_KEY_FIELD)
                    if db_pk is None: continue # Пропускаем, если нет PK

                    # Отбираем только те, что выше порога и еще не были добавлены
                    if similarity >= similarity_threshold and db_pk not in unique_direct_hit_pks:
                        face_id = entity.get(FACE_ID_FIELD, "N/A")
                        person_id = entity.get(PERSON_ID_FIELD, DEFAULT_PERSON_ID)
                        
                        direct_hits.append({
                            "pk": db_pk,
                            "face_id": face_id, # Сохраняем полный face_id (с индексом)
                            "person_id": person_id,
                            "similarity": similarity,
                            "query_face_idx": i # Индекс лица в запросе, которое дало это совпадение
                        })
                        unique_direct_hit_pks.add(db_pk)
                        logger.info(f"  -> Прямое совпадение: PK:{db_pk}, ID:{face_id}, PersonID:{person_id}, Sim:{similarity:.4f} (лицо запроса #{i})")
                        # Собираем ID кластеров, для которых нужны все члены
                        if person_id != DEFAULT_PERSON_ID:
                            found_person_ids.add(person_id)
                    elif db_pk in unique_direct_hit_pks:
                         logger.debug(f"    Пропуск (уже найдено): PK:{db_pk}")
                    # else: # Логирование не прошедших порог можно убрать для чистоты
                    #     logger.debug(f"    Ниже порога ({similarity:.4f} < {similarity_threshold}): PK:{db_pk}")

            direct_hits.sort(key=lambda x: x['similarity'], reverse=True)

            # Используем get_cluster_members из модуля milvus
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
            output_fields=[PRIMARY_KEY_FIELD, FACE_ID_FIELD, EMBEDDING_FIELD], limit=None
        )
        logger.info(f"Извлечено {len(all_entities_data)} записей.")
        return all_entities_data

    def _get_unclustered_faces(self) -> list:
        """Извлекает лица с person_id = DEFAULT_PERSON_ID."""
        if not self.client: return []
        logger.info(f"Инкрементальная кластеризация: извлечение некластеризованных лиц...")
        unclustered_filter = f"{PERSON_ID_FIELD} == '{DEFAULT_PERSON_ID}'"
        unclustered_faces_data = query_data(
            self.client, filter_expression=unclustered_filter,
            output_fields=[PRIMARY_KEY_FIELD, FACE_ID_FIELD, EMBEDDING_FIELD], limit=None
        )
        logger.info(f"Извлечено {len(unclustered_faces_data)} некластеризованных записей.")
        return unclustered_faces_data

    def _get_existing_cluster_centroids(self) -> dict:
        """Вычисляет центроиды существующих кластеров."""
        if not self.client: return {}
        logger.info(f"Инкрементальная кластеризация: извлечение данных существующих кластеров...")
        clustered_filter = f"{PERSON_ID_FIELD} != '{DEFAULT_PERSON_ID}'"
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

    def _get_next_person_id(self, existing_person_ids: set[str]) -> int:
        """Определяет следующий числовой ID для нового кластера."""
        # ... (логика осталась та же) ...
        next_person_numeric_id = 1
        if existing_person_ids:
            max_existing_id = 0
            for pid_str in existing_person_ids:
                try:
                    pid_int = int(pid_str)
                    if pid_int >= max_existing_id:
                        max_existing_id = pid_int
                except ValueError:
                    logger.warning(f"Не удалось преобразовать PersonID '{pid_str}' в число.")
            next_person_numeric_id = max_existing_id + 1
        logger.info(f"Новые PersonID для кластеров начнутся с {next_person_numeric_id}.")
        return next_person_numeric_id

    def _perform_dbscan_and_assign_ids(self, entities_to_cluster: list, cluster_eps: float, 
                                        cluster_min_samples: int, existing_person_ids: set[str]) -> list:
        """Выполняет DBSCAN и назначает PersonID."""
        # ... (логика DBSCAN и присвоения ID осталась та же) ...
        if not entities_to_cluster: return []

        pks = [e[PRIMARY_KEY_FIELD] for e in entities_to_cluster]
        face_ids = [e[FACE_ID_FIELD] for e in entities_to_cluster]
        embeddings_list = [e[EMBEDDING_FIELD] for e in entities_to_cluster]
        embeddings_np = np.array(embeddings_list)

        logger.info(f"Выполнение DBSCAN (eps={cluster_eps}, min_samples={cluster_min_samples}) для {len(embeddings_np)} векторов...")
        dbscan = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples, metric='cosine', n_jobs=-1)
        labels = dbscan.fit_predict(embeddings_np)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = np.sum(labels == -1)
        logger.info(f"Кластеризация завершена. Найдено новых кластеров: {num_clusters}, шумовых точек: {num_noise}")

        upsert_data = []
        person_id_map = {} 
        next_person_numeric_id = self._get_next_person_id(existing_person_ids)

        for i, label in enumerate(labels):
            person_id_to_upsert = DEFAULT_PERSON_ID
            if label != -1: 
                if label not in person_id_map:
                    person_id_map[label] = next_person_numeric_id
                    next_person_numeric_id += 1
                person_id_numeric = person_id_map[label]
                person_id_to_upsert = str(person_id_numeric) 
            
            upsert_data.append({
                PRIMARY_KEY_FIELD: pks[i],
                FACE_ID_FIELD: face_ids[i], 
                EMBEDDING_FIELD: embeddings_list[i], 
                PERSON_ID_FIELD: person_id_to_upsert 
            })
        return upsert_data

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
                    PRIMARY_KEY_FIELD: pk, FACE_ID_FIELD: face_data[FACE_ID_FIELD],
                    EMBEDDING_FIELD: face_data[EMBEDDING_FIELD], PERSON_ID_FIELD: assigned_pid
                })
                processed_pks.add(pk)
            else:
                remaining_faces_for_dbscan.append(face_data)
                processed_pks.add(pk) 

        logger.info(f"{len(assigned_data_for_upsert)} лиц присвоено существующим. {len(remaining_faces_for_dbscan)} осталось для DBSCAN.")
        return assigned_data_for_upsert, remaining_faces_for_dbscan

    def cluster_faces(self, mode: str, cluster_eps: float, cluster_min_samples: int):
        """
        Выполняет кластеризацию лиц в базе данных Milvus.

        Args:
            mode: Режим кластеризации ('full' или 'incremental').
            cluster_eps: Параметр eps для DBSCAN и порог для инкрементального присвоения.
            cluster_min_samples: Параметр min_samples для DBSCAN.
        """
        if not self.client:
            logger.error("Milvus client не инициализирован в cluster_faces.")
            return

        logger.info(f"Запуск кластеризации лиц: режим={mode.upper()}, eps={cluster_eps}, min_samples={cluster_min_samples}")
        
        all_upsert_data = [] 

        try:
            if mode == "full":
                all_entities_data = self._get_all_data_for_clustering()
                if not all_entities_data:
                    logger.info("Нет данных для полной кластеризации.")
                    return
                all_upsert_data = self._perform_dbscan_and_assign_ids(
                    all_entities_data, cluster_eps, cluster_min_samples, set()
                )

            elif mode == "incremental":
                unclustered_faces = self._get_unclustered_faces()
                existing_centroids = self._get_existing_cluster_centroids()
                
                assigned_data, remaining_for_dbscan = self._assign_faces_to_existing_clusters(
                    unclustered_faces, existing_centroids, cluster_eps
                )
                all_upsert_data.extend(assigned_data)

                if remaining_for_dbscan:
                    logger.info(f"Инкрементальный режим: кластеризация {len(remaining_for_dbscan)} оставшихся лиц...")
                    existing_pid_strings = set(existing_centroids.keys())
                    newly_clustered_data = self._perform_dbscan_and_assign_ids(
                        remaining_for_dbscan, cluster_eps, cluster_min_samples, existing_pid_strings
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