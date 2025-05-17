import sys
from pymilvus import MilvusClient
from loguru import logger

# --- Константы (должны совпадать с другими скриптами) ---
MILVUS_URI = "./milvus_lite.db"
COLLECTION_NAME = "faces"
# FACE_ID_FIELD = "face_id"  # Это поле удалено из схемы
PERSON_ID_FIELD = "person_id"
PRIMARY_KEY_FIELD = "pk" 
PHOTO_ID_FIELD = "photo_id"
FACE_INDEX_FIELD = "face_index"

# --- Настройка Логгера (вывод в stderr) ---
logger.remove() # Убираем стандартный обработчик
logger.add(sys.stderr, level="INFO") 

def inspect_collection():
    """Подключается к Milvus и выводит содержимое коллекции (без эмбеддингов)."""
    logger.info(f"Попытка подключения к Milvus: {MILVUS_URI}")
    try:
        client = MilvusClient(uri=MILVUS_URI)
        logger.info(f"Успешное подключение.")
        
        # Проверяем, существует ли коллекция
        if not client.has_collection(collection_name=COLLECTION_NAME):
             logger.warning(f"Коллекция '{COLLECTION_NAME}' не найдена.")
             return

        # Запрашиваем все записи, извлекая только нужные поля
        output_fields = [PRIMARY_KEY_FIELD, PHOTO_ID_FIELD, FACE_INDEX_FIELD, PERSON_ID_FIELD]
        logger.info(f"Запрос всех записей из '{COLLECTION_NAME}' с полями: {output_fields}...")
        
        # Сначала узнаем общее количество записей
        total_count = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["count(*)"])[0]["count(*)"]
        logger.info(f"Всего записей в коллекции: {total_count}")

        if total_count == 0:
            logger.info("Коллекция пуста.")
            return

        # Извлекаем все данные (если их не слишком много)
        # В реальном приложении здесь может потребоваться пагинация (limit/offset)
        try:
            results = client.query(
                 collection_name=COLLECTION_NAME,
                 filter="",
                 output_fields=output_fields,
                 limit=total_count # Получаем все записи
            )
            logger.info(f"Извлечено {len(results)} записей. Вывод:")
            
            # Выводим заголовок
            print("-" * 80)
            print(f"{PRIMARY_KEY_FIELD:<10} | {PHOTO_ID_FIELD:<10} | {FACE_INDEX_FIELD:<10} | {PERSON_ID_FIELD:<15}")
            print("-" * 80)
            
            # Выводим данные
            for entity in results:
                 pk_val = entity.get(PRIMARY_KEY_FIELD, 'N/A')
                 photo_id_val = entity.get(PHOTO_ID_FIELD, 'N/A')
                 face_index_val = entity.get(FACE_INDEX_FIELD, 'N/A')
                 person_id_val = entity.get(PERSON_ID_FIELD, 'N/A')
                 print(f"{str(pk_val):<10} | {str(photo_id_val):<10} | {str(face_index_val):<10} | {str(person_id_val):<15}")
            
            print("-" * 80)

        except Exception as query_e:
            logger.error(f"Ошибка при извлечении данных: {query_e}")

    except Exception as e:
        logger.error(f"Ошибка при подключении или выполнении запроса: {e}")

if __name__ == "__main__":
    inspect_collection() 