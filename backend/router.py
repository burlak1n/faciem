import os
import shutil
import uuid
from fastapi import File, UploadFile, Form, HTTPException, BackgroundTasks, APIRouter
from typing import Optional
# Импортируем настроенный логгер
from .logging_config import logger

# Импортируем наш менеджер и константы
from face_manager import FaceManager
from milvus import COLLECTION_NAME, DEFAULT_PERSON_ID, FACE_ID_FIELD, PERSON_ID_FIELD # Нужны для форматирования ответа
# Импортируем утилиты для работы с SQLite
from db_utils import get_groups_list, get_albums_for_group
# Импортируем Pydantic модели
from .schemas import (
    SearchResultItem, SearchResponse, 
    GroupItem, GroupListResponse, 
    AlbumItem, AlbumListResponse
    # AddResponse, ClusterRequest, ClusterResponse # Модели удаленных эндпоинтов
)

# --- Инициализация APIRouter --- 
rourer = APIRouter() # Создаем роутер

# --- Глобальные переменные и объекты для роутера --- 

# Создаем экземпляр менеджера (оставляем здесь)
try:
    face_manager = FaceManager()
except ConnectionError as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать FaceManager: {e}")
    face_manager = None
    
# Директория для временного хранения (оставляем здесь)
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Вспомогательные функции (оставляем здесь) --- 

def save_upload_file(upload_file: UploadFile) -> str:
    """Сохраняет загруженный файл во временную директорию и возвращает путь."""
    try:
        # Генерируем уникальное имя файла, сохраняя расширение
        ext = os.path.splitext(upload_file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.debug(f"Файл '{upload_file.filename}' временно сохранен как '{temp_file_path}'")
        return temp_file_path
    finally:
        upload_file.file.close()

# --- Вспомогательная функция для очистки временных файлов --- 

def cleanup_temp_file(path: str):
    """Удаляет временный файл."""
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Временный файл удален: {path}")
    except OSError as e:
        logger.error(f"Ошибка при удалении временного файла {path}: {e}")
        
# --- Вспомогательная функция для форматирования ответа поиска --- 

def format_search_response(manager_output: dict) -> SearchResponse:
    """
    Форматирует вывод FaceManager.search_face в формат, ожидаемый фронтендом.
    Использует photo_id и face_index из результатов Milvus.
    URL и Date пока не добавляем.
    """
    results = []
    direct_hits = manager_output.get("direct_hits", [])
    
    for hit in direct_hits:
        # Извлекаем данные напрямую из результатов поиска FaceManager
        # face_id = hit.get("face_id", "unknown_0") # Удалено
        match_photo_id = hit.get("photo_id", -1) # Получаем photo_id (INT64)
        match_face_index = hit.get("face_index", -1) # Получаем face_index (INT8)
        similarity = hit.get("similarity", 0.0)
        query_face_idx = hit.get("query_face_idx", 0)

        # Валидация (на всякий случай)
        if match_photo_id == -1 or match_face_index == -1:
             pk = hit.get("pk", "N/A")
             logger.warning(f"Получено невалидное photo_id ({match_photo_id}) или face_index ({match_face_index}) для PK {pk}. Пропуск или обработка по умолчанию.")
             # Можно пропустить эту запись или вернуть с плейсхолдерами
             # continue # Пропустить
             match_photo_id_str = str(match_photo_id) if match_photo_id != -1 else "invalid"
             match_face_index_int = match_face_index if match_face_index != -1 else -1
        else:
            match_photo_id_str = str(match_photo_id) # Приводим к строке для JSON
            match_face_index_int = match_face_index
             
        results.append(
            SearchResultItem(
                match_photo_id=match_photo_id_str, 
                match_face_index=match_face_index_int,
                query_face_index=query_face_idx,
                similarity=similarity,
                # match_url= ..., # Пока None
                # match_date= ... # Пока None
            )
        )
    return SearchResponse(results=results)

# --- Эндпоинты API (заменяем @app на @router)--- 

@rourer.get("/", tags=["Root"])
async def read_root():
    """
    Корневой эндпоинт для проверки работы API и статуса Milvus.
    """
    # Логика из бывшего /health
    milvus_status = "not_checked"
    if face_manager and face_manager.client:
        try:
             has_collection = face_manager.client.has_collection(collection_name=COLLECTION_NAME)
             milvus_status = "connected" if has_collection else "collection_not_found"
        except Exception as e:
             logger.warning(f"Ошибка проверки Milvus в /: {e}")
             milvus_status = "check_error"
        api_status = "ok"
    else:
        milvus_status = "not_checked_manager_unavailable"
        api_status = "error"
        
    return {
        "message": "Face Recognition API",
        "status": api_status,
        "milvus_status": milvus_status
    }

# Объединенный эндпоинт для поиска по файлу или URL
@rourer.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_face(
    background_tasks: BackgroundTasks,
    top_k: int = Form(default=10, description="Количество ближайших соседей"),
    similarity_threshold: float = Form(default=0.4, description="Порог схожести"),
    file: Optional[UploadFile] = File(None, description="Изображение для поиска (альтернатива image_url)"), 
    image_url: Optional[str] = Form(None, description="URL изображения для поиска (альтернатива file)")
): 
    """
    Поиск лиц по загруженному изображению ИЛИ по URL.
    
    Необходимо предоставить **либо** `file`, **либо** `image_url`.
    """
    if not face_manager or not face_manager.client:
        raise HTTPException(status_code=503, detail="Сервис временно недоступен (проблема с FaceManager или Milvus)")

    # Проверка входных данных
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Необходимо предоставить либо файл ('file'), либо URL ('image_url')")
    if file and image_url:
        raise HTTPException(status_code=400, detail="Необходимо предоставить только один источник: либо файл ('file'), либо URL ('image_url')")

    input_source: str
    temp_file_path: Optional[str] = None

    try:
        if file:
            if not file.content_type or not file.content_type.startswith("image/"):
                 raise HTTPException(status_code=400, detail="Некорректный тип файла. Ожидается изображение.")
            temp_file_path = save_upload_file(file)
            input_source = temp_file_path
            # Добавляем задачу удаления файла в фон только если файл был успешно сохранен
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
            logger.info(f"Запрос /search: file={file.filename}, top_k={top_k}, threshold={similarity_threshold}")
        elif image_url: # Проверку на None уже сделали выше
            input_source = image_url
            logger.info(f"Запрос /search: url={image_url}, top_k={top_k}, threshold={similarity_threshold}")
        else:
             # Эта ветка не должна выполниться из-за проверок выше, но для полноты
             raise HTTPException(status_code=500, detail="Внутренняя ошибка: не удалось определить источник изображения.")

        search_output = face_manager.search_face(
            query_image_path_or_url=input_source, 
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        return format_search_response(search_output)
        
    except HTTPException as http_exc: # Пробрасываем HTTP исключения
        raise http_exc
    except Exception as e:
        logger.error(f"Ошибка в /search: {e}", exc_info=True)
        # Убедимся, что временный файл удален, если ошибка произошла после его создания
        # if temp_file_path:
        #     cleanup_temp_file(temp_file_path) # cleanup_temp_file уже добавлена в background_tasks
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при поиске: {e}")

# --- Эндпоинты для работы с vk_faces.db --- 

@rourer.get("/api/groups", response_model=GroupListResponse, tags=["VK Database"])
async def api_get_groups():
    """
    Получение списка групп из базы данных SQLite (vk_faces.db).
    """
    logger.info("Запрос /api/groups")
    try:
        groups_data = get_groups_list()
        return GroupListResponse(groups=groups_data)
    except HTTPException as http_exc:
        logger.error(f"Ошибка HTTPException в /api/groups: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Неожиданная ошибка в /api/groups: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")

@rourer.get("/api/albums/{group_id}", response_model=AlbumListResponse, tags=["VK Database"])
async def api_get_albums(group_id: str):
    """
    Получение списка альбомов для указанной группы из SQLite (vk_faces.db).
    Проверка статуса обработки в Milvus временно отключена.
    """
    logger.info(f"Запрос /api/albums/{group_id}")
    try:
        albums_data = get_albums_for_group(group_id=group_id)
        # Pydantic автоматически проверит соответствие данных модели AlbumItem
        return AlbumListResponse(albums=albums_data)
    except HTTPException as http_exc:
        logger.error(f"Ошибка HTTPException в /api/albums/{group_id}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Неожиданная ошибка в /api/albums/{group_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")
 