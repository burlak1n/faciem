import asyncio
from pymilvus import connections, utility, Collection
import sys
import os
import sqlite3 # <--- Добавлено для работы с SQLite

# Добавляем путь к корневой директории проекта, чтобы импортировать config и утилиты
# Это может потребовать корректировки в зависимости от того, откуда запускается скрипт
# Определяем путь к корневой директории проекта относительно текущего файла
# Предполагается, что скрипт находится в корневой директории проекта
# или в поддиректории, тогда нужно будет скорректировать путь.
# Для простоты, если скрипт в корне, sys.path.append('.') должно быть достаточно.
# Если скрипт в папке scripts/, а config.py в корне, то sys.path.append('..')
# Будем считать, что скрипт запускается из корня, где лежит config.py
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Добавляет директорию скрипта
# Если config.py в корне проекта, а скрипт, например, в папке utils:
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # Подняться на один уровень
# sys.path.append(project_root)


try:
    import config
    # Предполагаем, что эти переменные есть в config.py или milvus_utils.py
    # Если они в milvus_utils, то from milvus_utils import ...
    MILVUS_COLLECTION_NAME = getattr(config, 'MILVUS_COLLECTION_NAME', 'vk_faces')
    MILVUS_ALIAS = getattr(config, 'MILVUS_ALIAS', 'default')
    MILVUS_HOST = getattr(config, 'MILVUS_HOST', 'localhost')
    MILVUS_PORT = getattr(config, 'MILVUS_PORT', '19530')
    # Добавляем путь к файлу SQLite из конфига
    SQLITE_DB_PATH = getattr(config, 'URL_FILE_PATH', None)

except ImportError as e:
    print(f"Ошибка импорта конфигурационного файла (config.py): {e}")
    print("Убедитесь, что config.py существует и доступен в PYTHONPATH, или скрипт находится в корне проекта.")
    sys.exit(1)
except AttributeError as e:
    print(f"Ошибка: одна из необходимых констант Milvus (MILVUS_COLLECTION_NAME, MILVUS_ALIAS, MILVUS_HOST, MILVUS_PORT) не найдена в config.py: {e}")
    sys.exit(1)

# --- Функция для получения ID из SQLite ---
def get_ids_from_sqlite(db_path: str) -> set:
    """
    Извлекает все ID из таблицы 'photos' в указанной SQLite базе данных.
    """
    if not db_path or not os.path.exists(db_path):
        print(f"Ошибка: Путь к базе данных SQLite не указан или файл не найден: {db_path}")
        return set()
        
    all_sqlite_ids = set()
    try:
        print(f"\nПодключение к SQLite базе данных: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("Выполнение запроса: SELECT id FROM photos")
        cursor.execute("SELECT id FROM photos")
        count = 0
        while True:
            rows = cursor.fetchmany(10000) # Читаем порциями
            if not rows:
                break
            all_sqlite_ids.update(row[0] for row in rows)
            count += len(rows)
            print(f"Загружено {count} ID из SQLite...", end='\r')
        print(f"\nВсего загружено {len(all_sqlite_ids)} уникальных ID из SQLite.")
        conn.close()
    except sqlite3.Error as e:
        print(f"Ошибка при работе с SQLite: {e}")
        if 'conn' in locals() and conn:
            conn.close()
        return set() # Возвращаем пустой сет в случае ошибки
    except Exception as e:
        print(f"Непредвиденная ошибка при чтении из SQLite: {e}")
        if 'conn' in locals() and conn:
            conn.close()
        return set()
        
    return all_sqlite_ids

# --- Основная функция для получения данных из Milvus ---
async def get_all_marked_data(collection_name: str, batch_size: int = 1000) -> dict:
    """
    Извлекает все photo_id и face_index из указанной коллекции Milvus постранично.
    Возвращает словарь {photo_id: [face_indices]}
    """
    all_milvus_photo_data = {} # Словарь для хранения {photo_id: [face_indices]}
    try:
        connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=str(MILVUS_PORT)) # Порт должен быть строкой
        print(f"Успешное подключение к Milvus: {MILVUS_HOST}:{MILVUS_PORT} (alias: {MILVUS_ALIAS})")
    except Exception as e:
        print(f"Не удалось подключиться к Milvus: {e}")
        # Возвращаем пустой словарь при ошибке подключения
        return all_milvus_photo_data 

    if not utility.has_collection(collection_name, using=MILVUS_ALIAS):
        print(f"Коллекция '{collection_name}' не найдена.")
        # Возвращаем пустой словарь
        return all_milvus_photo_data 

    collection = Collection(collection_name, using=MILVUS_ALIAS)
    print(f"Информация о коллекции '{collection_name}':")
    print(f"  Схема: {collection.schema}")
    print(f"  Описание: {collection.description}")
    print(f"  Количество записей (приблизительно): {collection.num_entities}")
    
    try:
        print(f"Загрузка коллекции '{collection_name}' в память...")
        collection.load()
        print(f"Коллекция '{collection_name}' успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке коллекции '{collection_name}': {e}")
        # Возвращаем данные, которые успели собрать, если такие есть
        return all_milvus_photo_data 


    offset = 0
    total_entities_processed = 0

    # Проверяем, есть ли поле face_index в схеме
    has_face_index_field = any(field.name == 'face_index' for field in collection.schema.fields)
    output_fields = ['photo_id'] # photo_id должен быть первичным ключом и существовать
    
    # Проверяем, что photo_id действительно является частью полей, которые можно запросить
    # (Обычно первичный ключ неявно включается, но для output_fields лучше указать явно, если он не вектор)
    if not any(field.name == 'photo_id' for field in collection.schema.fields):
        print(f"Ошибка: Поле 'photo_id' не найдено в схеме коллекции '{collection_name}'. Невозможно продолжить.")
        return

    if has_face_index_field:
        output_fields.append('face_index')
    else:
        print("Предупреждение: поле 'face_index' не найдено в схеме. Будут извлечены только 'photo_id'.")


    print(f"\nНачинаем извлечение данных из коллекции '{collection_name}' с размером батча {batch_size}...")
    print(f"Будут запрошены поля: {output_fields}")
    
    pk_field_name = collection.schema.primary_field.name
    print(f"Имя первичного ключа: {pk_field_name}")


    while True:
        try:
            # Выражение expr="" для выборки всех записей
            # Для корректной выборки всех записей, если photo_id - первичный ключ,
            # можно использовать выражение типа "pk_field_name like '%'" если photo_id строка,
            # или "pk_field_name > 0" если photo_id число.
            # Но лучше всего просто expr="" или expr="true" (если поддерживается)
            # Или использовать специфичное для вашего photo_id выражение, если знаете его тип.
            # Для универсальности, если photo_id строка: expr=f"{pk_field_name} != \"\""
            # Если photo_id число: expr=f"{pk_field_name} >= 0" (или < 0 если могут быть отрицательные)
            # Самый простой и часто работающий вариант - просто пустой expr или "true"
            
            # Попробуем более надежное выражение для выборки всех:
            # Для строкового ID: photo_id like "%"
            # Для числового ID: photo_id >= 0 (предполагая что ID не отрицательные)
            # Уточним, что `expr` для "всех записей" зависит от типа первичного ключа.
            # Если photo_id - это VARCHAR, то `photo_id like "%"` подойдет.
            # Если INT64, то `photo_id >= 0` (или другое в зависимости от диапазона).
            # Самый безопасный способ запросить все - использовать `pk_field_name is not null` (если PK не может быть null)
            # или просто пустой `expr`, Milvus обычно понимает это как "все".
            # Для большей надежности, можно проверить тип PK и составить выражение.
            # Пока оставим expr="", обычно это работает.
            
            query_expr = "" # По умолчанию для "всех записей"
            # Если вы знаете тип photo_id и хотите быть более явными:
            # if collection.schema.primary_field.dtype == DataType.VARCHAR:
            #    query_expr = f"{pk_field_name} like \"%\""
            # elif collection.schema.primary_field.dtype == DataType.INT64:
            #    query_expr = f"{pk_field_name} >= 0" # или другое подходящее условие

            results = collection.query(
                expr=query_expr,
                output_fields=output_fields,
                limit=batch_size,
                offset=offset,
                consistency_level="Strong" # Для получения самых свежих данных
            )
        except Exception as e:
            print(f"Ошибка при выполнении запроса query (offset={offset}, limit={batch_size}): {e}")
            break

        if not results:
            print("Больше нет данных для извлечения.")
            break

        for res_item in results: # res_item это dict
            # Имя первичного ключа мы знаем из pk_field_name, используем его
            current_photo_id = res_item.get('photo_id')
            
            if current_photo_id is None:
                print(f"Предупреждение: получен результат без значения для поля 'photo_id'. Запись: {res_item}")
                continue

            face_index_val = res_item.get('face_index', None) # None если поля face_index нет или оно не было запрошено

            # Попытка привести ID из Milvus к int для сравнения с SQLite ID
            try:
                milvus_id_as_int = int(current_photo_id)
            except (ValueError, TypeError):
                 # Если photo_id не приводится к int, пропускаем (или логируем как предупреждение)
                 # Это важно, если у вас могут быть строковые ID не числового вида
                 print(f"Предупреждение: photo_id '{current_photo_id}' из Milvus не может быть преобразован в int. Пропуск.")
                 continue

            if milvus_id_as_int not in all_milvus_photo_data:
                all_milvus_photo_data[milvus_id_as_int] = []
            
            if face_index_val is not None:
                 all_milvus_photo_data[milvus_id_as_int].append(face_index_val)
            # Если face_index_val is None (поле не существует или не было данных для этой записи), 
            # photo_id все равно будет добавлен в all_milvus_photo_data с пустым списком face_indices,
            # что означает, что фото "размечено" (есть запись), но без информации о конкретных лицах отсюда.

        num_retrieved = len(results)
        total_entities_processed += num_retrieved
        print(f"Извлечено {num_retrieved} сущностей (всего обработано: {total_entities_processed}). Текущий offset: {offset}")

        if num_retrieved < batch_size:
            print("Извлечено меньше сущностей, чем размер батча, вероятно, это конец данных.")
            break
        
        offset += batch_size
        # await asyncio.sleep(0.01) # Очень короткая пауза, если нужно

    print(f"\n--- Итог ---")
    print(f"Всего уникальных photo_id (размеченных фотографий) найдено: {len(all_milvus_photo_data)}")
    
    # Сохранение в CSV (раскомментируйте, если нужно)
    # import csv
    # csv_file_path = 'marked_photos_info.csv'
    # try:
    #     with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         if has_face_index_field:
    #             writer.writerow(['photo_id', 'face_indices'])
    #             for pid, findices in all_milvus_photo_data.items():
    #                 writer.writerow([pid, sorted(list(set(findices)))]) # Сортируем и убираем дубликаты face_index
    #         else:
    #             writer.writerow(['photo_id'])
    #             for pid in all_milvus_photo_data.keys():
    #                 writer.writerow([pid])
    #     print(f"Данные сохранены в файл: {csv_file_path}")
    # except IOError as e:
    #     print(f"Ошибка при сохранении CSV файла: {e}")


    print("\nПример данных (первые 20 размеченных фотографий и их face_index, если есть):")
    count = 0
    for pid, findices in all_milvus_photo_data.items():
        if findices: # Если список face_indices не пуст
            print(f"Photo ID: {pid}, Face Indices: {sorted(list(set(findices)))}")
        else: # Если список face_indices пуст (поле могло отсутствовать или для этого ID не было face_index)
            print(f"Photo ID: {pid} (размечен, информация о face_index отсутствует или поле не найдено в схеме)")
        count += 1
        if count >= 20:
            break
            
    try:
        collection.release()
        print(f"Коллекция '{collection_name}' выгружена из памяти.")
    except Exception as e:
        print(f"Ошибка при выгрузке коллекции '{collection_name}': {e}")

    # Возвращаем собранные данные
    return all_milvus_photo_data 

async def main():
    collection_to_query = MILVUS_COLLECTION_NAME
    sqlite_db = SQLITE_DB_PATH
    
    if not collection_to_query:
        print("Имя коллекции Milvus (MILVUS_COLLECTION_NAME) не определено в config.py.")
        return
        
    if not sqlite_db:
        print("Путь к базе данных SQLite (URL_FILE_PATH) не определен в config.py.")
        return

    print(f"Запрос информации для коллекции: {collection_to_query}")
    # Сначала получаем данные из Milvus
    milvus_data = await get_all_marked_data(collection_to_query, batch_size=500) # Размер батча можно настроить
    
    if not milvus_data:
        print("Не удалось получить данные из Milvus или коллекция пуста.")
        return
        
    # Получаем ID из SQLite
    sqlite_ids = get_ids_from_sqlite(sqlite_db)
    
    if not sqlite_ids:
        print("Не удалось получить ID из SQLite или таблица пуста.")
        return
        
    # Находим пересечение (ID, которые есть и в Milvus, и в SQLite)
    # Ключи milvus_data уже должны быть int после обработки в get_all_marked_data
    milvus_processed_ids = set(milvus_data.keys()) 
    common_ids = milvus_processed_ids.intersection(sqlite_ids)
    
    print(f"\n--- Итоговое пересечение --- ")
    print(f"Найдено ID в Milvus: {len(milvus_processed_ids)}")
    print(f"Найдено ID в SQLite: {len(sqlite_ids)}")
    print(f"Найдено ID, присутствующих и в Milvus, и в SQLite: {len(common_ids)}")
    
    # Сохранение в CSV (раскомментируйте и адаптируйте, если нужно)
    # import csv
    # csv_file_path = 'processed_sqlite_ids_info.csv'
    # try:
    #     with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         # Определяем, было ли найдено поле face_index ранее
    #         # (проверяем первый элемент в milvus_data, если он есть)
    #         has_face_indices = False
    #         if common_ids:
    #             first_id = next(iter(common_ids))
    #             if milvus_data.get(first_id) is not None: # Проверяем, есть ли список (даже пустой)
    #                 has_face_indices = True 
    #                 
    #         if has_face_indices:
    #             writer.writerow(['sqlite_id', 'face_indices'])
    #             for sid in sorted(list(common_ids)):
    #                 findices = milvus_data.get(sid, []) # Получаем face_indices для этого ID
    #                 writer.writerow([sid, sorted(list(set(findices)))]) 
    #         else:
    #             writer.writerow(['sqlite_id'])
    #             for sid in sorted(list(common_ids)):
    #                 writer.writerow([sid])
    #     print(f"Данные о пересечении сохранены в файл: {csv_file_path}")
    # except IOError as e:
    #     print(f"Ошибка при сохранении CSV файла: {e}")

    print("\nПример ID из SQLite, которые были найдены в Milvus (первые 20):")
    count = 0
    for sid in sorted(list(common_ids)):
        findices = milvus_data.get(sid, []) # Получаем face_indices для этого ID
        if findices:
            print(f"SQLite ID: {sid}, Milvus Face Indices: {sorted(list(set(findices)))}")
        else:
            print(f"SQLite ID: {sid} (размечен в Milvus, face_indices отсутствуют или не найдены)")
        count += 1
        if count >= 20:
            break

if __name__ == "__main__":
    # Для запуска из командной строки: python get_marked_photos_info.py
    asyncio.run(main()) 