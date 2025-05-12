import os
import numpy as np
# Убраны ненужные импорты Milvus и DeepFace напрямую
from loguru import logger
import glob
import argparse
import sys 

# Импортируем FaceManager и используемые константы
from face_manager import FaceManager 
# Импортируем только константы, нужные для этого скрипта (пути, параметры по умолчанию)
from milvus import DEFAULT_PERSON_ID # Нужен для проверки в выводе

# --- Константы Скрипта (параметры по умолчанию для argparse) ---
DATABASE_IMAGE_DIR = "data/database_faces" 
QUERY_IMAGE_DIR = "data/query_face"       
LOG_FILE = "face_database.log" # Лог самого скрипта
DEFAULT_SEARCH_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# --- Настройка Логгера --- 
logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add(LOG_FILE, level="DEBUG", rotation="5 MB") 

# --- Вспомогательные функции (остается find_image_files) --- 
def find_image_files(directory: str) -> list[str]:
    """Находит все файлы JPG, JPEG, PNG в указанной директории."""
    patterns = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
    files = []
    if not os.path.exists(directory):
        logger.warning(f"Директория не найдена: {directory}")
        return []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    logger.debug(f"Найдено {len(files)} файлов изображений в {directory}")
    return files

# --- Функции для режимов работы (Теперь используют FaceManager) --- 

def run_add_mode(manager: FaceManager, image_dir: str):
    """Выполняет добавление лиц из папки, используя FaceManager."""
    logger.info(f"\n=== РЕЖИМ ДОБАВЛЕНИЯ ЛИЦ ИЗ {image_dir} ===")
    image_files = find_image_files(image_dir)
    if not image_files:
        logger.warning(f"Не найдено изображений в {image_dir}. Пропуск добавления.")
        logger.warning(f"Создайте папку '{image_dir}' и поместите туда изображения.")
        return

    added_files_count = 0
    processed_files_count = 0
    for img_path in image_files:
         logger.info(f"Обработка файла для добавления: {img_path}")
         # Используем метод менеджера
         pks = manager.add_face(img_path) 
         processed_files_count += 1
         # Проверяем, были ли реально добавлены лица (pks не None и не пустой)
         if pks: 
              added_files_count += 1 
              logger.info(f"Добавлены лица из {os.path.basename(img_path)}. PKs: {pks}")
         # Если pks пустой, значит лица не найдены или эмбеддинги нулевые (логируется внутри add_face)
         elif pks is None:
             logger.error(f"Произошла ошибка Milvus при добавлении лиц из {os.path.basename(img_path)}.")

    logger.info(f"Завершено добавление. Обработано файлов: {processed_files_count}. Файлов с добавленными лицами: {added_files_count}")

def format_and_print_search_results(search_output: dict):
    """Форматирует и выводит результаты поиска, полученные от FaceManager.search_face."""
    direct_hits = search_output.get("direct_hits", [])
    cluster_members = search_output.get("cluster_members", {})
    
    if not direct_hits:
        logger.info("Прямых совпадений (выше порога) не найдено.")
        return
        
    # Группируем результаты по PersonID
    final_person_details = {} 
    for hit in direct_hits:
        pid = hit["person_id"]
        if pid == DEFAULT_PERSON_ID: # Не показываем детали для некластеризованных прямых хитов
             # Можно добавить логирование, если нужно видеть и некластеризованные
             # logger.info(f"  -> Прямое совпадение (некластеризованное): {hit[\'face_id\']} (Sim: {hit[\'similarity\']:.4f}) - пропускаем в итоговом отчете")
             continue
             
        details = final_person_details.setdefault(pid, {"direct_hits": [], "all_member_face_ids": set()})
        # Добавляем информацию о прямом совпадении
        details["direct_hits"].append({
            "found_face_id": hit["face_id"], # Полный ID с суффиксом
            "pk": hit["pk"],
            "similarity": hit["similarity"],
            # "query_file": os.path.basename(query_image), # Источник запроса уже не передаем сюда
            "query_face_idx": hit["query_face_idx"]
        })

    # Добавляем информацию о всех членах кластера
    for pid, members_list in cluster_members.items():
        if pid == DEFAULT_PERSON_ID: continue # Не должно быть, но на всякий случай
        details = final_person_details.setdefault(pid, {"direct_hits": [], "all_member_face_ids": set()})
        for member in members_list:
             details["all_member_face_ids"].add(member["face_id"]) # Полный ID с суффиксом
             
    # Вывод отчета
    logger.info(f"\n--- Итоговый отчет по найденным личностям --- ")
    if not final_person_details:
        logger.info("Не найдено ни одной идентифицированной личности (кластеризованной).")
        return
        
    logger.info(f"Найдено {len(final_person_details)} уникальных кластеризованных личностей:")
    for person_id, details in final_person_details.items():
        logger.info(f"  Личность ID: {person_id}")
        
        # Прямые совпадения для этой личности
        if details["direct_hits"]:
            logger.info(f"    Прямые совпадения ({len(details['direct_hits'])}):")
            # Сортируем прямые хиты по схожести
            sorted_hits = sorted(details["direct_hits"], key=lambda x: x['similarity'], reverse=True)
            for hit in sorted_hits:
                # Удаляем суффикс _число из face_id для вывода
                base_hit_face_id = "_".join(hit['found_face_id'].split('_')[:-1]) if '_' in hit['found_face_id'] else hit['found_face_id']
                logger.info(f"      - {base_hit_face_id} (PK: {hit['pk']}, Схожесть: {hit['similarity']:.4f}, лицо запроса #{hit['query_face_idx']})")
        else:
            # Это может случиться, если личность найдена только через cluster_members (что маловероятно при текущей логике)
             logger.info("    Прямых совпадений для этой личности не найдено (были ниже порога или некластеризованные)." )

        # Все члены кластера
        if details["all_member_face_ids"]:
            # Удаляем суффиксы _число из face_id для вывода и сортируем уникальные
            cleaned_member_ids = sorted(list(set(
                "_".join(fid.split('_')[:-1]) if '_' in fid else fid 
                for fid in details["all_member_face_ids"]
            )))
            logger.info(f"    Все известные фото этой личности ({len(cleaned_member_ids)}):")
            for face_id_cleaned in cleaned_member_ids:
                logger.info(f"      - {face_id_cleaned}")
        else:
             # Этого не должно происходить, если есть direct_hits с person_id
             logger.warning(f"    Не найдены члены кластера для PersonID {person_id}.") 
        logger.info("-" * 20)

def run_search_mode(manager: FaceManager, query_source: str, top_k: int, threshold: float):
    """Выполняет поиск лиц из файла, URL или директории и выводит отчет."""
    logger.info(f"\n=== РЕЖИМ ПОИСКА ЛИЦ ИЗ {query_source} ===")
    
    query_targets = []
    is_url = query_source.startswith(('http://', 'https://'))
    
    if os.path.isdir(query_source):
        logger.info(f"Источник запроса - директория. Поиск файлов изображений...")
        query_targets = find_image_files(query_source)
        if not query_targets:
            logger.warning(f"Не найдено изображений в директории запроса: {query_source}")
            return
    elif is_url:
        logger.info(f"Источник запроса - URL.")
        query_targets.append(query_source)
    elif os.path.isfile(query_source):
        logger.info(f"Источник запроса - файл.")
        query_targets.append(query_source)
    else:
        logger.error(f"Источник запроса не найден или не является файлом/директорией/URL: {query_source}")
        return

    total_queries_processed = 0
    for target in query_targets:
        logger.info(f"--- Обработка запроса: {os.path.basename(target) if not target.startswith(('http')) else target} ---")
        # Используем метод менеджера для поиска
        search_output = manager.search_face(target, top_k, threshold)
        
        # Форматируем и выводим результат для ЭТОГО запроса
        format_and_print_search_results(search_output)
        total_queries_processed += 1
        
    logger.info(f"Завершено обработка поиска. Всего обработано запросов: {total_queries_processed}")

# --- Основной блок --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Управление базой данных лиц Milvus с помощью FaceManager.")
    parser.add_argument("--add", metavar='DIR', type=str, help=f"Добавить все лица из указанной директории (например, {DATABASE_IMAGE_DIR}).")
    parser.add_argument("--query", metavar='PATH_OR_URL', type=str, help="Искать лицо по указанному пути к файлу или URL.")
    # Добавляем опциональные параметры для поиска
    parser.add_argument("--top_k", type=int, default=DEFAULT_SEARCH_TOP_K, help=f"Количество ближайших соседей для поиска (по умолчанию: {DEFAULT_SEARCH_TOP_K}).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help=f"Порог схожести для совпадений (по умолчанию: {DEFAULT_SIMILARITY_THRESHOLD}).")
    # Можно добавить параметры для __init__ FaceManager, если нужно менять модель/детектор
    # parser.add_argument("--model", type=str, default="Facenet512", help="Модель DeepFace.")
    # parser.add_argument("--detector", type=str, default="retinaface", help="Детектор DeepFace.")

    args = parser.parse_args()

    # Проверяем, что указан хотя бы один режим
    if not args.add and not args.query:
        parser.print_help()
        logger.warning("Необходимо указать режим работы: --add DIR или --query PATH_OR_URL")
        sys.exit(1)
        
    # Проверяем, что режимы не указаны одновременно (хотя можно и разрешить)
    if args.add and args.query:
         logger.warning("Указаны оба режима --add и --query. Будет выполнен только --add.")
         args.query = None # Приоритет у добавления

    logger.info("Инициализация FaceManager...")
    # Можно передать параметры модели/детектора из args, если они добавлены
    # face_manager = FaceManager(model_name=args.model, detector_backend=args.detector)
    face_manager = FaceManager()
    
    # Проверяем, успешно ли инициализирован менеджер (подключился ли к Milvus)
    if not face_manager.client:
         logger.error("Не удалось инициализировать FaceManager (проблема с Milvus). Завершение работы.")
         sys.exit(1)

    # Выполняем запрошенное действие
    if args.add:
        run_add_mode(face_manager, args.add)
    elif args.query:
         run_search_mode(face_manager, args.query, args.top_k, args.threshold)

    logger.info("Скрипт face_database.py завершил работу.") 