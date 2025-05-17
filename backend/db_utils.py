import sqlite3
from typing import Optional, Any, List, Dict, Generator
from loguru import logger
import os
from fastapi import HTTPException

# Предполагается, что config.py находится в том же каталоге или доступен через PYTHONPATH
try:
    from . import config
    from .download_utils import get_max_quality_url
except ImportError:
    import config
    from download_utils import get_max_quality_url


def read_urls_from_file(
    input_path: str, 
    source_type: str = "txt", 
    db_table: str = "photos", 
    db_column: str = "url", 
    db_id_column: str = "id", 
    db_date_column: str = "date"
) -> Generator[tuple[Any, str, Optional[str]], None, None]:
    """
    Читает ID, URL и дату из файла или SQLite.
    Применяет get_max_quality_url к каждому URL перед возвратом.
    Возвращает генератор кортежей (id, url, date).
    Для текстовых файлов ID будет сгенерирован, дата будет None.
    """
    if source_type == "txt":
        try:
            with open(input_path, 'r') as f:
                count = 0
                for line_idx, line in enumerate(f):
                    url = line.strip()
                    if url:
                        # Применяем get_max_quality_url здесь
                        processed_url = get_max_quality_url(url) 
                        yield (f"txt_{line_idx}", processed_url, None)
                        count += 1
                logger.info(f"Загружено {count} URL-адресов из текстового файла: {input_path} (ID будут сгенерированы, дата отсутствует)")
        except FileNotFoundError:
            logger.error(f"Файл с URL-адресами не найден: {input_path}")
            # Здесь можно либо пробросить исключение дальше, либо вернуть пустой генератор
            # return
            # raise # Если считаем это критической ошибкой
        except Exception as e:
            logger.error(f"Ошибка чтения файла {input_path}: {e}")
            # Аналогично, обработка ошибки
            # return
            # raise
    elif source_type == "sqlite":
        conn = None
        try:
            conn = sqlite3.connect(input_path)
            cursor = conn.cursor()
            query = f"SELECT {db_id_column}, {db_column}, {db_date_column} FROM {db_table}"
            logger.info(f"Выполнение SQL запроса: {query} для базы данных: {input_path}")
            cursor.execute(query)
            count = 0
            for row in cursor:
                if row and len(row) == 3 and row[0] is not None and row[1] is not None:
                    photo_id, url_val, photo_date_val = row
                    # Применяем get_max_quality_url здесь
                    processed_url = get_max_quality_url(str(url_val))
                    photo_date_str = str(photo_date_val) if photo_date_val is not None else None
                    yield (photo_id, processed_url, photo_date_str)
                    count += 1
                else:
                    logger.warning(f"Пропущена строка из SQLite из-за неполных данных (ожидалось 3 значения: id, url, date): {row}")
            logger.info(f"Загружено {count} записей (ID, URL, дата) из SQLite: {input_path} (таблица: {db_table}, колонки: {db_id_column}, {db_column}, {db_date_column})")
        except sqlite3.Error as e:
            logger.error(f"Ошибка SQLite при чтении из {input_path} (таблица: {db_table}): {e}")
            # raise # или return
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при чтении из SQLite {input_path}: {e}")
            # raise # или return
        finally:
            if conn:
                conn.close()
    else:
        logger.error(f"Неизвестный тип источника: {source_type}. Допустимые значения: 'txt', 'sqlite'.")
        # raise ValueError(f"Неизвестный тип источника: {source_type}") # или return

def get_urls_for_photo_ids(
    photo_ids: List[Any],
    db_path: str,
    db_table: str = "photos", 
    db_id_column: str = "id",
    db_url_column: str = "url",
    db_date_column: str = "date"
) -> Dict[Any, Dict[str, Optional[str]]]:
    """Запрашивает SQLite для получения URL и даты по списку photo_id."""
    if not photo_ids:
        return {}
    
    results = {pid: {"url": None, "date": None} for pid in photo_ids} 
    conn = None
    try:
        # db_path должен быть передан, например, из config.URL_FILE_PATH
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        placeholders = ',' .join('?' * len(photo_ids))
        query = f"SELECT {db_id_column}, {db_url_column}, {db_date_column} FROM {db_table} WHERE {db_id_column} IN ({placeholders})"
        
        param_values = tuple(photo_ids) # Убедимся, что это кортеж
        
        logger.debug(f"Выполнение SQL для получения URL и даты: {query} с параметрами {param_values}")
        cursor.execute(query, param_values)
        
        for row in cursor:
            if len(row) == 3:
                pid, url_val, date_val = row
                # Не применяем get_max_quality_url здесь, т.к. предполагается, что в БД уже обработанные URL или это не требуется для этой функции
                results[pid]["url"] = str(url_val) if url_val is not None else None
                results[pid]["date"] = str(date_val) if date_val is not None else None
            else:
                logger.warning(f"Получена строка с неожиданным количеством колонок: {row} при запросе URL и даты.")
                
    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при получении URL и даты для ID {photo_ids}: {e}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при получении URL и даты из SQLite: {e}")
    finally:
        if conn:
            conn.close()
            
    return results

def get_groups_list() -> List[Dict[str, str]]:
    """
    Извлекает список групп (screen_name) из SQLite.
    """
    db_path = getattr(config, 'URL_FILE_PATH', None)
    if not db_path or not os.path.exists(db_path):
        print(f"[API_ERROR /api/groups] SQLite DB path not configured or file not found: {db_path}")
        raise HTTPException(status_code=500, detail="База данных не найдена на сервере")

    groups = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Используем screen_name и как id, и как name
        cursor.execute("SELECT screen_name FROM groups ORDER BY screen_name") 
        for row in cursor:
            if row and row[0]:
                groups.append({"id": str(row[0]), "name": str(row[0])})
    except sqlite3.Error as e:
        print(f"[API_ERROR /api/groups] SQLite error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных при получении списка групп: {e}")
    except Exception as e_gen:
        print(f"[API_ERROR /api/groups] General error: {e_gen}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e_gen}")
    finally:
        if conn:
            conn.close()
    
    print(f"[API_DEBUG /api/groups] Returning {len(groups)} groups.")
    return groups

def get_albums_for_group(group_id: str) -> List[Dict[str, Any]]:
    """
    Извлекает список альбомов для указанной группы из SQLite.
    ВРЕМЕННО: Проверка статуса в Milvus удалена.
    """
    db_path = getattr(config, 'URL_FILE_PATH', None)
    if not db_path or not os.path.exists(db_path):
        print(f"[API_ERROR /api/albums/{group_id}] SQLite DB path not configured or file not found: {db_path}")
        raise HTTPException(status_code=500, detail="База данных не найдена на сервере")

    albums = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем альбомы
        cursor.execute(
            "SELECT id, title, size FROM albums WHERE group_screen_name = ? ORDER BY updated", 
            (group_id,)
        )
        
        album_rows = cursor.fetchall()
        
        for row in album_rows:
            if row and row[0] is not None and row[1] is not None:
                album_id = row[0]
                
                # --- Начало удаленного блока проверки Milvus ---
                # processed_photos_count = 0 
                # В реальной реализации здесь была бы проверка статуса в Milvus
                # для подсчета processed_photos_count
                # --- Конец удаленного блока проверки Milvus ---
                
                albums.append({
                    "id": album_id,
                    "title": str(row[1]),
                    "size": row[2] if row[2] is not None else 0,
                    # "processed_count": processed_photos_count # Временно убрано
                })
                
    except sqlite3.Error as e:
        print(f"[API_ERROR /api/albums/{group_id}] SQLite error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных при получении списка альбомов: {e}")
    except Exception as e_gen:
        print(f"[API_ERROR /api/albums/{group_id}] General error: {e_gen}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e_gen}")
    finally:
        if conn:
            conn.close()
            
    print(f"[API_DEBUG /api/albums/{group_id}] Returning {len(albums)} albums.")
    return albums
