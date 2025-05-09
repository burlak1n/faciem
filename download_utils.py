import os
import time
import urllib.parse
import asyncio
import aiohttp
import cv2
import numpy as np
from loguru import logger

# Предполагается, что config.py находится в том же каталоге или доступен через PYTHONPATH
try:
    # Попытка относительного импорта, если download_utils является частью пакета
    from . import config 
except ImportError:
    # Прямой импорт, если запускается как отдельный скрипт или config.py в PYTHONPATH
    import config

# --- Вспомогательная функция для локального пути ---
def get_local_path_for_url(photo_id: any, url: str, local_base_path: str) -> str:
    """
    Генерирует локальный путь для сохранения/загрузки файла на основе URL и ID.
    Создает директорию local_base_path, если она не существует.
    Обеспечивает уникальность и безопасность имени файла.
    """
    os.makedirs(local_base_path, exist_ok=True)
    
    filename_from_url = ""
    url_ext = ".jpg" 
    try:
        parsed_url = urllib.parse.urlparse(url)
        path_component = parsed_url.path
        if path_component:
            filename_from_url = os.path.basename(path_component)
            _, url_ext_candidate = os.path.splitext(filename_from_url)
            if url_ext_candidate:
                url_ext = url_ext_candidate.lower()
    except Exception as e:
        logger.warning(f"Не удалось полностью распарсить URL ({url}) для извлечения имени файла: {e}. Используется расширение по умолчанию '{url_ext}'.")

    safe_photo_id_str = str(photo_id)
    invalid_file_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    for char in invalid_file_chars:
        safe_photo_id_str = safe_photo_id_str.replace(char, '_')
    
    safe_photo_id_str = "_".join(filter(None, safe_photo_id_str.split('_')))
    local_filename = f"{safe_photo_id_str}{url_ext}"
    
    max_filename_len = 200 
    if len(local_filename) > max_filename_len:
        name_part, ext_part = os.path.splitext(local_filename)
        ext_len = len(ext_part)
        dot_len = 1 if ext_part and not ext_part.startswith('.') else 0
        max_name_part_len = max_filename_len - ext_len - dot_len
        
        if max_name_part_len < 1:
            safe_photo_id_prefix = safe_photo_id_str[:max(10, max_filename_len - ext_len - dot_len - 7)] # 7 for '_trunc_'
            local_filename = f"{safe_photo_id_prefix}_trunc_{url_ext.replace('.', '')}{ext_part}" # Ensure some uniqueness
            logger.warning(f"Имя файла для URL {url} (ID: {photo_id}) было сильно укорочено и изменено на '{local_filename}' из-за ограничений длины.")
        else:
            name_part = name_part[:max_name_part_len]
            local_filename = name_part + ext_part
            logger.warning(f"Имя файла для URL {url} (ID: {photo_id}) было укорочено до '{local_filename}' из-за общей длины.")

    full_path = os.path.join(local_base_path, local_filename)
    return full_path

# --- Функция для модификации URL ---
def get_max_quality_url(url_string: str) -> str:
    """Проверяет URL на наличие параметров 'as' и 'cs'.
    Если они есть, пытается заменить значение 'cs' на максимальное из 'as'.
    """
    try:
        parsed_url = urllib.parse.urlparse(url_string)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        if 'as' in query_params and 'cs' in query_params:
            as_values_str = query_params['as'][0]
            
            available_sizes = []
            for size_pair_str in as_values_str.split(','):
                if 'x' in size_pair_str:
                    try:
                        width, height = map(int, size_pair_str.split('x'))
                        available_sizes.append(((width, height), width * height))
                    except ValueError:
                        logger.warning(f"Не удалось распарсить пару размеров: {size_pair_str} в URL: {url_string}")
                        continue
            
            if not available_sizes:
                logger.debug(f"Не найдено корректных размеров в параметре 'as' для URL: {url_string}")
                return url_string

            available_sizes.sort(key=lambda x: x[1], reverse=True)
            max_size_tuple = available_sizes[0][0]
            max_quality_cs_value = f"{max_size_tuple[0]}x{max_size_tuple[1]}"
            
            query_params['cs'] = [max_quality_cs_value]
            
            new_query_string = urllib.parse.urlencode(query_params, doseq=True)
            new_url = parsed_url._replace(query=new_query_string).geturl()
            return new_url
        else:
            return url_string

    except Exception as e:
        logger.error(f"Ошибка при обработке URL для максимального качества ({url_string}): {e}")
        return url_string

async def download_image_async(session: aiohttp.ClientSession, photo_id: any, url: str) -> tuple[any, np.ndarray | None]:
    """
    Асинхронно скачивает изображение по URL или загружает из локального кэша.
    Возвращает его как OpenCV (numpy) массив вместе с photo_id.
    """
    download_start_time = time.monotonic() # Забыли импортировать time
    image_np = None 
    local_path = None 
    source = "download"

    if hasattr(config, 'DOWNLOAD_FILES_LOCALLY') and config.DOWNLOAD_FILES_LOCALLY:
        if not hasattr(config, 'LOCAL_DOWNLOAD_PATH') or not hasattr(config, 'OVERWRITE_LOCAL_FILES'):
            logger.error("Переменные LOCAL_DOWNLOAD_PATH или OVERWRITE_LOCAL_FILES не найдены в конфигурации! Локальное кэширование не будет работать для этого вызова.")
        else:
            local_path = get_local_path_for_url(photo_id, url, config.LOCAL_DOWNLOAD_PATH)
            if os.path.exists(local_path) and not config.OVERWRITE_LOCAL_FILES:
                logger.info(f"Попытка загрузки изображения из локального кэша: {local_path} (ID: {photo_id})")
                try:
                    image_np_candidate = cv2.imread(local_path) 
                    if image_np_candidate is not None:
                        logger.info(f"Изображение успешно загружено из локального кэша: {local_path} (ID: {photo_id})")
                        source = "cache"
                        duration_ms = (time.monotonic() - download_start_time) * 1000
                        logger.debug(f"[BENCHMARK] Загрузка изображения (ID: {photo_id}) из КЭША заняла {duration_ms:.2f} мс.")
                        return photo_id, image_np_candidate 
                    else:
                        logger.warning(f"Не удалось декодировать локальный файл: {local_path} (ID: {photo_id}). Файл может быть поврежден. Попытка скачивания.")
                except Exception as e:
                    logger.error(f"Ошибка чтения локального файла {local_path} (ID: {photo_id}): {e}. Попытка скачивания.")
    
    try:
        logger.debug(f"Скачивание изображения с URL: {url} (ID: {photo_id})")
        async with session.get(url, timeout=config.DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status() 
            image_data = await response.read()
            
            image_array = np.frombuffer(image_data, np.uint8)
            image_np_downloaded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image_np_downloaded is None:
                logger.warning(f"Не удалось декодировать изображение, скачанное с URL: {url} (ID: {photo_id})")
                return photo_id, None 

            logger.debug(f"Изображение успешно скачано и декодировано с URL: {url} (ID: {photo_id})")
            image_np = image_np_downloaded 

            if hasattr(config, 'DOWNLOAD_FILES_LOCALLY') and config.DOWNLOAD_FILES_LOCALLY and local_path:
                save_start_time = time.monotonic()
                try:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True) 
                    with open(local_path, 'wb') as f: 
                        f.write(image_data) 
                    logger.info(f"Скачанное изображение сохранено локально: {local_path}")
                    save_duration_ms = (time.monotonic() - save_start_time) * 1000
                    logger.debug(f"[BENCHMARK] Сохранение изображения (ID: {photo_id}) локально заняло {save_duration_ms:.2f} мс.")
                except Exception as e_save:
                    logger.error(f"Ошибка сохранения скачанного файла {local_path} (ID: {photo_id}): {e_save}")
            
            duration_ms = (time.monotonic() - download_start_time) * 1000
            logger.debug(f"[BENCHMARK] Скачивание ({source}) и декодирование изображения (ID: {photo_id}) с URL заняло {duration_ms:.2f} мс.")
            return photo_id, image_np 

    except asyncio.TimeoutError:
        duration_ms = (time.monotonic() - download_start_time) * 1000
        logger.error(f"Тайм-аут при скачивании изображения {url} (ID: {photo_id}). Затрачено: {duration_ms:.2f} мс.")
        return photo_id, None
    except aiohttp.ClientError as e_http:
        duration_ms = (time.monotonic() - download_start_time) * 1000
        logger.error(f"Ошибка HTTP при скачивании изображения {url} (ID: {photo_id}): {e_http}. Затрачено: {duration_ms:.2f} мс.")
        return photo_id, None
    except Exception as e: 
        duration_ms = (time.monotonic() - download_start_time) * 1000
        logger.error(f"Непредвиденная ошибка при скачивании или обработке изображения {url} (ID: {photo_id}): {e}. Затрачено: {duration_ms:.2f} мс.")
        return photo_id, None
