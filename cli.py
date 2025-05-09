import argparse
import asyncio
import sys
# --- Импорт Loguru --- 
from loguru import logger

# --- Импорт конфигурации ---
# Предполагаем, что config.py находится в том же каталоге или доступен через PYTHONPATH
try:
    import config # Если запускается из корня, где лежит config.py
    # Либо, если cli.py будет в подпапке, то from .. import config
except ImportError:
    # Попытка импорта как если бы cli.py и milvus_vk_pipeline.py находились рядом
    # Это может быть не идеальным решением для всех структур проекта
    try:
        from . import config as app_config # Используем псевдоним, чтобы не конфликтовать с импортом config в milvus_vk_pipeline
        config = app_config
    except ImportError:
        logger.error("Не удалось импортировать файл конфигурации (config.py). Убедитесь, что он доступен.")
        sys.exit(1)


# --- Импорт основного пайплайна ---
# Предполагается, что milvus_vk_pipeline.py находится в том же каталоге, что и cli.py
# или доступен через PYTHONPATH.
try:
    import milvus_vk_pipeline
except ImportError as e:
    logger.error(f"Не удалось импортировать milvus_vk_pipeline: {e}. Убедитесь, что файл находится в правильном месте.")
    sys.exit(1)

# --- Настройка Loguru (можно вынести в config.py или инициализировать здесь) ---
# Если Loguru уже настроен в milvus_vk_pipeline при его импорте, повторная настройка может не понадобиться
# или может потребовать удаления предыдущих обработчиков, если это необходимо.
# Для простоты, предполагаем, что настройки Loguru из config.py, используемые в milvus_vk_pipeline, достаточны.
# Если нет, то:
# logger.remove() 
# logger.add(
#     sys.stderr, 
#     level=config.LOGURU_LEVEL, 
#     format=config.LOGURU_FORMAT
# )


async def main_async_wrapper_cli(args: argparse.Namespace):
    """
    Асинхронная обертка для вызова функций индексации из milvus_vk_pipeline.
    """
    logger.info(f"Асинхронная инициализация для режима: {args.mode} (вызов из CLI)")
    
    milvus_vk_pipeline.init_deepface_model() 

    try:
        milvus_collection = milvus_vk_pipeline.init_milvus_connection()
    except Exception as e:
        logger.critical(f"Критическая ошибка на этапе инициализации Milvus (CLI): {e}. Выход.")
        return

    logger.info(f"Чтение URL-адресов из {args.input_path} (тип: {args.source_type}) для CLI")
    image_data_tuples_generator = milvus_vk_pipeline.read_urls_from_file(
        args.input_path, 
        source_type=args.source_type, 
        db_table=args.db_table, 
        db_column=args.db_column,
        db_id_column=args.db_id_column
    )
    image_data_tuples = list(image_data_tuples_generator) 
    
    if not image_data_tuples:
        logger.warning("Нет данных (ID, URL) для обработки в режиме индексации (CLI). Завершение работы.")
        return
        
    try:
        # Для CLI progress_callback обычно None
        await milvus_vk_pipeline.process_and_store_faces_async(
            milvus_collection, 
            image_data_tuples, 
            progress_callback=None, # В CLI режиме колбэк не используется
            skip_existing=args.skip_existing # Передаем флаг
        ) 
    except Exception as e:
        logger.error(f"Произошла ошибка во время процесса асинхронной индексации (CLI): {e}")


def main_cli():
    """
    Основная функция для интерфейса командной строки.
    """
    parser = argparse.ArgumentParser(description="CLI для пайплайна индексации или поиска фотографий VK в Milvus.")
    parser.add_argument("--mode", type=str, default="index", choices=["index", "search"], 
                        help="Режим работы: 'index' для индексации новых фото, 'search' для поиска.")
    parser.add_argument("--image_path", type=str, help="Путь к изображению для поиска (только для режима 'search').")
    parser.add_argument("--input_path", type=str, default=config.URL_FILE_PATH, 
                        help=f"Путь к входному файлу (текстовый файл с URL или SQLite БД). По умолчанию: {config.URL_FILE_PATH}")
    parser.add_argument("--source_type", type=str, default="txt", choices=["txt", "sqlite"],
                        help="Тип источника URL-адресов: 'txt' или 'sqlite'. По умолчанию: 'txt'.")
    parser.add_argument("--db_table", type=str, default="photos",
                        help="Имя таблицы в SQLite БД (если source_type='sqlite'). По умолчанию: 'photos'.")
    parser.add_argument("--db_column", type=str, default="url",
                        help="Имя колонки с URL в SQLite БД (если source_type='sqlite'). По умолчанию: 'url'.")
    parser.add_argument("--db_id_column", type=str, default="id",
                        help="Имя колонки с ID в SQLite БД (если source_type='sqlite'). По умолчанию: 'id'.")
    parser.add_argument("--top_k", type=int, default=config.SEARCH_TOP_K,
                        help=f"Количество лучших совпадений для поиска (режим 'search'). По умолчанию: {config.SEARCH_TOP_K}")
    parser.add_argument("--threshold", type=float, default=config.SEARCH_THRESHOLD_IP,
                        help=f"Порог сходства для поиска (режим 'search'). По умолчанию: {config.SEARCH_THRESHOLD_IP}")
    parser.add_argument("--skip_existing", action='store_true', # Новый аргумент
                        help="Пропускать индексацию фото, если photo_id уже есть в Milvus (режим 'index').")


    args = parser.parse_args()

    logger.info(f"Запуск CLI в режиме: {args.mode}")
    
    if args.mode == "index":
        try:
            asyncio.run(main_async_wrapper_cli(args))
        except Exception as e:
            logger.critical(f"Критическая ошибка при запуске асинхронной индексации из CLI: {e}")
    
    elif args.mode == "search":
        milvus_vk_pipeline.init_deepface_model() 

        try:
            milvus_collection = milvus_vk_pipeline.init_milvus_connection() 
        except Exception as e:
            logger.critical(f"Критическая ошибка на этапе инициализации Milvus для поиска (CLI): {e}. Выход.")
            return

        if not args.image_path:
            logger.error("Для режима 'search' необходимо указать --image_path. Выход.")
            return
        try:
            search_results = milvus_vk_pipeline.search_similar_faces_in_milvus(
                milvus_collection, 
                args.image_path,
                top_k=args.top_k, 
                search_threshold=args.threshold
            )
            if search_results:
                logger.info("Найденные совпадения (CLI):")
                for res in search_results:
                    logger.info(f"  Query Face Idx: {res['query_face_index']}, Match photo_id: {res['match_photo_id']}, Match Face Idx: {res['match_face_index']}, Similarity: {res['similarity']:.4f}")
            else:
                logger.info("Совпадений не найдено (CLI).")
        except Exception as e:
            logger.error(f"Произошла ошибка во время процесса поиска (CLI): {e}")

    logger.info("CLI пайплайн завершил работу.")

if __name__ == "__main__":
    main_cli() 