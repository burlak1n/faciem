import asyncio
import traceback
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import datetime # Added for file modification time
from typing import List, Optional, Any, Tuple, Callable # Added Tuple and Callable
import json
import sqlite3
import signal
import sys

from pymilvus import Collection # <--- Added for direct SQLite query in web_server for now
from milvus_vk_pipeline import (
    init_deepface_model,
    init_milvus_connection,
    process_and_store_faces_async,  # Используем существующую реализацию
    search_similar_faces_in_milvus_by_bytes,
)

# Предполагается, что config.py и milvus_vk_pipeline.py находятся в том же каталоге
# или доступны через PYTHONPATH. Для удобства импорта, если они в корне проекта,
# а web_server.py в подкаталоге, может потребоваться настройка sys.path или структуры проекта.
# Для текущей структуры, где все в корне:
try:
    import config
    # Если функция get_image_data_by_album будет в db_utils, ее нужно будет импортировать
    # from db_utils import get_image_data_by_album 
    # Импортируем get_local_path_for_url для генерации ожидаемых локальных путей
    from download_utils import get_local_path_for_url
except ImportError as e:
    print(f"Ошибка импорта: {e}. Убедитесь, что config.py и download_utils.py доступны.")
    # Можно добавить обработку или выход, если критичные модули не найдены
    # sys.exit(1) 

# --- Глобальные переменные и объекты ---
app = FastAPI()

# Настройка для статических файлов (HTML, CSS, JS)
# Создадим каталог 'static' и 'templates' позже
# Для простоты, HTML будет пока инлайновым или в виде строки
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Добавляем константу для максимального количества параллельных обработок
MAX_CONCURRENT_PROCESSING = getattr(config, 'MAX_CONCURRENT_PROCESSING', 2)
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSING)

# --- Модели данных для FastAPI ---
class IndexByAlbumRequest(BaseModel):
    # group_id: Optional[str] = None # Оставим опциональным, если не всегда используется
    album_id: int
    skip_existing: bool = False

# Новая модель данных для запроса разметки фотографий
class MarkPhotosRequest(BaseModel):
    photo_ids: List[str]  # ID фотографий для разметки
    mark_as_processed: bool = True  # True - пометить как обработанные, False - как необработанные
    custom_label: Optional[str] = None  # Опциональная метка для фотографий

# --- Очередь для обновлений прогресса ---
# Это простой способ. Для более надежного решения можно использовать Redis Pub/Sub или Kafka.
progress_queue = asyncio.Queue()

# --- Колбэк для отправки прогресса ---
async def send_progress_update(
    photo_id: Any, 
    url: str, 
    image_base64: Optional[str], 
    status: str, 
    faces_count: Optional[int] = None,
    face_confidences: Optional[List[float]] = None,
    embedding_duration_ms: Optional[float] = None,
    processed_count: Optional[int] = None,
    total_count: Optional[int] = None,
    error_message: Optional[str] = None,
    face_index_processed: Optional[int] = None,
    total_duration_sec: Optional[float] = None,
    avg_throughput_photos_sec: Optional[float] = None,
    timestamp_msk: Optional[str] = None,
    photo_date: Optional[str] = None,
    skipped_count: Optional[int] = None,
    # Добавим новые поля для статистики измененного пайплайна, если потребуется
    processed_images_dl_count: Optional[int] = None,
    faces_extracted_count: Optional[int] = None,
    embeddings_extracted_count: Optional[int] = None,
    no_face_markers_count: Optional[int] = None,
    milvus_inserted_total_count: Optional[int] = None,
    error_counts: Optional[dict] = None,
):
    """Отправляет обновление в очередь."""
    update_data = {
        "photo_id": str(photo_id) if photo_id is not None else None,
        "url": url, # В новом сценарии это будет локальный путь или идентификатор
        "image_base64": image_base64, 
        "status": status,
        "faces_count": faces_count,
        "face_confidences": face_confidences, 
        "embedding_duration_ms": embedding_duration_ms,
        "processed_count": processed_count,
        "total_count": total_count,
        "error_message": error_message,
        "face_index_processed": face_index_processed,
        "total_duration_sec": total_duration_sec,
        "avg_throughput_photos_sec": avg_throughput_photos_sec,
        "timestamp_msk": timestamp_msk,
        "photo_date": photo_date,
        "skipped_count": skipped_count,
        "processed_images_dl_count": processed_images_dl_count,
        "faces_extracted_count": faces_extracted_count,
        "embeddings_extracted_count": embeddings_extracted_count,
        "no_face_markers_count": no_face_markers_count,
        "milvus_inserted_total_count": milvus_inserted_total_count,
        "error_counts": error_counts,
    }
    # Убираем None значения для чистоты JSON, если нужно (опционально)
    update_data = {k: v for k, v in update_data.items() if v is not None}
    
    # Логируем только ID для краткости
    log_preview = {"photo_id": update_data.get("photo_id"), "status": update_data.get("status"), "error": update_data.get("error_message")}
    print(f"[DEBUG web_server] send_progress_update: Добавление в очередь: {log_preview}") # Отладка
    await progress_queue.put(update_data)

# --- Вспомогательная функция для получения данных из SQLite по альбому ---
# Эту функцию ЛУЧШЕ вынести в db_utils.py для чистоты кода.
# Пока оставляю здесь для простоты демонстрации.
def get_image_data_by_album_from_sqlite(
    db_path: str, 
    album_id_filter: str,
    # Колонки в таблице photos согласно CREATE TABLE:
    id_col: str = "id",            # INTEGER
    album_id_col: str = "album_id",# INTEGER
    # owner_id_col: str = "owner_id" # INTEGER, пока не используется для фильтрации здесь
    download_url_col: str = "url", # TEXT, содержит URL для скачивания
    date_col: str = "date",        # INTEGER (unix timestamp)
    # description_col: str = "description" # TEXT, пока не используется
) -> List[Tuple[Any, str, Optional[str]]]:
    """
    Извлекает из SQLite данные о фотографиях (photo_id, original_download_url, photo_date)
    для указанного album_id. Затем генерирует ОЖИДАЕМЫЙ локальный путь 
    и возвращает (photo_id, expected_local_path, photo_date_iso) только для существующих локальных файлов.
    """
    print(f"[DEBUG get_image_data_by_album] Querying SQLite: {db_path} for album_id: {album_id_filter}")
    image_data_tuples_for_pipeline = []
    conn = None
    
    # Базовый путь, куда должны были скачиваться файлы (из конфига)
    # Этот путь используется в get_local_path_for_url
    local_download_base = getattr(config, 'LOCAL_DOWNLOAD_PATH', 'data/downloaded_images')
    if not os.path.isdir(local_download_base):
        print(f"[ERROR get_image_data_by_album] Базовая директория для скачанных файлов '{local_download_base}' не найдена. Проверьте config.LOCAL_DOWNLOAD_PATH.")
        # Можно либо создать директорию, либо выбросить ошибку, либо вернуть пустой список.
        # Для безопасности, лучше вернуть пустой список или ошибку, если путь критичен.
        # os.makedirs(local_download_base, exist_ok=True) # Если хотим создавать
        # raise HTTPException(status_code=500, detail=f"Директория {local_download_base} не найдена.")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = f"SELECT {id_col}, {download_url_col}, {date_col} FROM photos WHERE {album_id_col} = ?"
        params = [int(album_id_filter)] # album_id в БД - INTEGER
            
        print(f"[DEBUG get_image_data_by_album] Executing SQL: {query} with params: {params}")
        cursor.execute(query, tuple(params))
        
        raw_db_records = 0
        found_local_files = 0

        for row in cursor:
            raw_db_records += 1
            if len(row) == 3 and row[0] is not None and row[1] is not None:
                photo_id_val, original_url_val, photo_unix_date_val = row
                
                photo_id_str = str(photo_id_val)
                original_url_str = str(original_url_val)
                
                photo_date_iso_str: Optional[str] = None
                if photo_unix_date_val is not None:
                    try:
                        photo_date_iso_str = datetime.datetime.fromtimestamp(int(photo_unix_date_val)).isoformat()
                    except ValueError:
                        print(f"[WARNING get_image_data_by_album] Некорректное значение даты (timestamp) {photo_unix_date_val} для photo_id {photo_id_str}")
                
                # Генерируем ожидаемый локальный путь, используя ту же логику, что и при скачивании
                expected_local_file_path = get_local_path_for_url(photo_id_str, original_url_str, local_download_base)
                
                if os.path.isfile(expected_local_file_path):
                    image_data_tuples_for_pipeline.append((photo_id_str, expected_local_file_path, photo_date_iso_str))
                    found_local_files +=1
                else:
                    print(f"[INFO get_image_data_by_album] Локальный файл не найден для photo_id: {photo_id_str} по пути: {expected_local_file_path} (Оригинальный URL из БД: {original_url_str}). Пропуск.")
            else:
                print(f"[WARNING get_image_data_by_album] Пропущена строка из SQLite из-за неполных данных: {row}")

    except sqlite3.Error as e:
        print(f"[ERROR get_image_data_by_album] SQLite error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
    except ValueError as ve: # Для int(album_id_filter)
        print(f"[ERROR get_image_data_by_album] album_id должен быть числом: {ve}")
        raise HTTPException(status_code=400, detail=f"album_id должен быть числом: {album_id_filter}")
    except Exception as e_gen:
        print(f"[ERROR get_image_data_by_album] General error: {e_gen}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e_gen}")
    finally:
        if conn:
            conn.close()
            
    print(f"[DEBUG get_image_data_by_album] Найдено {raw_db_records} записей в БД для album_id {album_id_filter}. Из них {found_local_files} файлов существует локально и будет обработано.")
    return image_data_tuples_for_pipeline

# --- Эндпоинты FastAPI ---

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    # Убедимся, что index.html есть в папке templates
    if not os.path.exists(os.path.join("templates", "index.html")):
        return HTMLResponse(content="<html><body><h1>Index page</h1><p>Template 'index.html' not found in 'templates' folder.</p></body></html>", status_code=500)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/index/start_by_album")
async def start_indexing_by_album_endpoint(request_data: IndexByAlbumRequest):
    """
    Запускает процесс индексации изображений из указанного альбома.
    Использует существующую реализацию process_and_store_faces_async из pipeline_core.py
    """
    global milvus_collection
    global total_images_for_current_run
    print("[DEBUG /index/start_by_album] Endpoint started.")
    print(f"[DEBUG /index/start_by_album] Request data: {request_data}")
    
    db_file_path = getattr(config, 'URL_FILE_PATH', None)
    if not db_file_path or not os.path.exists(db_file_path):
        print(f"[ERROR /index/start_by_album] SQLite DB path not configured (URL_FILE_PATH) or file not found: {db_file_path}")
        raise HTTPException(status_code=500, detail="Путь к базе данных SQLite не настроен или файл не найден.")

    if not milvus_collection:
        print("[WARN /index/start_by_album] Milvus collection not initialized at startup. Initializing now...")
        try:
             milvus_collection = init_milvus_connection()
             print("[WARN /index/start_by_album] Milvus collection initialized successfully.")
        except Exception as e:
             print(f"[ERROR /index/start_by_album] Failed to initialize Milvus: {e}\n{traceback.format_exc()}")
             raise HTTPException(status_code=500, detail="Ошибка инициализации Milvus. Невозможно запустить индексацию.")

    try:
        print(f"[DEBUG /index/start_by_album] Reading images from SQLite for album_id='{request_data.album_id}'...")
        
        image_data_tuples = get_image_data_by_album_from_sqlite(
            db_path=db_file_path,
            album_id_filter=request_data.album_id
        )
        
        print(f"[DEBUG /index/start_by_album] Получено {len(image_data_tuples)} локальных путей к файлам из SQLite для обработки.")

        if not image_data_tuples:
            print("[DEBUG /index/start_by_album] No images found in the specified SQLite database.")
            return {"message": f"Не найдено локально существующих изображений для альбома: {request_data.album_id} (проверьте, что файлы скачаны в {getattr(config, 'LOCAL_DOWNLOAD_PATH', 'data/downloaded_images')})"}

        total_images_for_current_run = len(image_data_tuples)
        album_identifier = request_data.album_id
        print(f"[DEBUG /index/start_by_album] Sending 'started' progress update for {album_identifier}...")
        await send_progress_update(None, album_identifier, None, "started", total_count=total_images_for_current_run)
        print("[DEBUG /index/start_by_album] 'started' progress update sent.")

        print("[DEBUG /index/start_by_album] Creating background task for process_and_store_faces_async...")
        # Используем существующую реализацию из pipeline_core.py
        asyncio.create_task(
            process_and_store_faces_async(
                milvus_collection=milvus_collection,
                image_data_tuples=image_data_tuples, 
                progress_callback=send_progress_update,
                skip_existing=request_data.skip_existing
            )
        )
        print("[DEBUG /index/start_by_album] Background task created. Returning response.")
        return {"message": f"Процесс индексации для альбома {album_identifier} запущен ({total_images_for_current_run} файлов)."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Ошибка при запуске индексации из базы данных SQLite: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка при запуске индексации: {str(e)}")

@app.post("/search/upload")
async def search_by_upload_endpoint(file: UploadFile = File(...)):
    """Принимает изображение, ищет похожие лица в Milvus."""
    global milvus_collection
    print(f"[DEBUG /search/upload] Received file: {file.filename}, type: {file.content_type}")

    if not milvus_collection:
        # Это не должно происходить, если startup_event сработал
        print("[ERROR /search/upload] Milvus collection not available.")
        raise HTTPException(status_code=503, detail="Сервис Milvus недоступен (не инициализирован). Попробуйте позже.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Недопустимый тип файла. Пожалуйста, загрузите изображение.")

    try:
        image_bytes = await file.read()
        print(f"[DEBUG /search/upload] Image read into bytes ({len(image_bytes)} bytes). Starting search...")
        
        search_results = await search_similar_faces_in_milvus_by_bytes(
            milvus_collection=milvus_collection,
            query_image_bytes=image_bytes,
            # Можно передать top_k и threshold из запроса, если нужно
            # top_k=request.query_params.get('top_k', config.SEARCH_TOP_K),
            # search_threshold=request.query_params.get('threshold', config.SEARCH_THRESHOLD_IP)
        )
        
        print(f"[DEBUG /search/upload] Search finished. Found {len(search_results)} matches.")
        # Возвращаем результаты как есть (они уже содержат match_url)
        return JSONResponse(content={"results": search_results})

    except Exception as e:
        print(f"[ERROR /search/upload] Error during search: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка во время поиска: {str(e)}")

def signal_handler(sig, frame):
    print("\nПолучен сигнал завершения. Завершаем работу...")
    # Закрываем все активные соединения
    if milvus_collection:
        try:
            milvus_collection.flush()
            print("Данные Milvus сохранены")
        except Exception as e:
            print(f"Ошибка при сохранении данных Milvus: {e}")
    sys.exit(0)

# Регистрируем обработчик сигнала
signal.signal(signal.SIGINT, signal_handler)

@app.get("/index/stream_status")
async def stream_status(request: Request):
    """
    Отправляет обновления статуса индексации через Server-Sent Events.
    """
    async def event_generator():
        print("[DEBUG web_server] event_generator: Начало работы генератора SSE.")
        try:
            while True:
                if await request.is_disconnected():
                    print("[DEBUG web_server] event_generator: Клиент отключился.")
                    break
                
                try:
                    update = await progress_queue.get()
                    print(f"[DEBUG web_server] event_generator: Получено из очереди: {update.get('status')}, photo_id: {update.get('photo_id')}")
                    yield f"data: {json.dumps(update)}\n\n"
                    progress_queue.task_done()
                    
                    if update.get("status") in ["finished", "error_critical"]:
                        print("[DEBUG web_server] event_generator: Процесс завершен или критическая ошибка.")
                        break
                        
                except asyncio.CancelledError:
                    print("[DEBUG web_server] event_generator: Получен сигнал отмены.")
                    break
                except Exception as e:
                    print(f"[ERROR web_server] event_generator: Ошибка: {e}")
                    break
                    
        except asyncio.CancelledError:
            print("[DEBUG web_server] event_generator: Задача отменена.")
        finally:
            print("[DEBUG web_server] event_generator: Завершение работы генератора SSE.")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# --- Новые эндпоинты для получения групп и альбомов ---
@app.get("/api/groups")
async def get_groups_list():
    db_path = getattr(config, 'URL_FILE_PATH', None)
    if not db_path or not os.path.exists(db_path):
        print(f"[API_ERROR /api/groups] SQLite DB path not configured or file not found: {db_path}")
        raise HTTPException(status_code=500, detail="База данных не найдена на сервере")

    groups = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # В вашей схеме таблица groups имеет одну колонку screen_name, которая является PRIMARY KEY
        # Используем screen_name и как id, и как name для простоты
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
    return {"groups": groups}

@app.get("/api/albums/{group_id}")
async def get_albums_for_group(group_id: str):
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
        
        # Сохраняем все альбомы в список перед обработкой фотографий
        album_rows = cursor.fetchall()
        
        for row in album_rows:
            if row and row[0] is not None and row[1] is not None:
                album_id = row[0]
                
                # Выполняем запрос для подсчета обработанных фотографий
                if milvus_collection:
                    processed_photos_count = 0
                    try:
                        # Запрос к SQLite для получения всех ID фотографий альбома
                        photo_ids_cursor = conn.cursor()
                        photo_ids_cursor.execute("SELECT COUNT(*) FROM photos WHERE album_id = ?", (album_id,))
                        total_photos_in_album = photo_ids_cursor.fetchone()[0]
                        
                        # Проверяем статистику по альбому (можно доработать для более точных данных)
                        processed_photos_count = 0  # По умолчанию считаем, что ничего не обработано
                        
                        # Используем новый курсор для запроса фотографий
                        sample_photos_cursor = conn.cursor()
                        sample_photos_cursor.execute(
                            "SELECT id FROM photos WHERE album_id = ? LIMIT 10",
                            (album_id,)
                        )
                        
                        # Проверяем статус нескольких фотографий для оценки прогресса
                        processed_count = 0
                        total_checked = 0
                        
                        for sample_photo_row in sample_photos_cursor:
                            if sample_photo_row and sample_photo_row[0] is not None:
                                total_checked += 1
                                photo_id = sample_photo_row[0]
                                
                                try:
                                    milvus_expr = f'photo_id == {photo_id}' if isinstance(photo_id, (int, float)) else f'photo_id == "{str(photo_id).replace("\\", "\\\\").replace('"', '\\"')}"'
                                    check_res = milvus_collection.query(expr=milvus_expr, output_fields=["photo_id"], limit=1)
                                    if check_res:
                                        processed_count += 1
                                except Exception as e:
                                    print(f"[WARNING] Ошибка проверки статуса фото {photo_id} в Milvus: {e}")
                        
                        # Оцениваем долю обработанных фотографий по выборке
                        if total_checked > 0:
                            processed_ratio = processed_count / total_checked
                            processed_photos_count = int(total_photos_in_album * processed_ratio)
                        
                    except Exception as e:
                        print(f"[WARNING] Ошибка при подсчете обработанных фотографий для альбома {album_id}: {e}")
                
                albums.append({
                    "id": album_id,
                    "title": str(row[1]),
                    "size": row[2] if row[2] is not None else 0,
                    "processed_count": processed_photos_count if 'processed_photos_count' in locals() else 0
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
    return {"albums": albums}

@app.get("/api/albums/{album_id}/photos")
async def get_album_photos(
    album_id: int,
    page: int = 1,
    page_size: int = 20,
    status_filter: Optional[str] = None  # "all", "processed", "not_processed"
):
    print(f"[API_DEBUG] /api/albums/{album_id}/photos: Получен запрос с параметрами page={page}, page_size={page_size}, status_filter={status_filter}")
    
    db_path = getattr(config, 'URL_FILE_PATH', None)
    if not db_path or not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="База данных не найдена на сервере")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Определяем тип фильтрации
        if not status_filter or status_filter == "all":
            filtering_needed = False
        else:
            filtering_needed = True
            filter_for_processed = status_filter == "processed"
        
        # Базовый запрос для получения фотографий
        base_query = """
            SELECT id, url, date 
            FROM photos 
            WHERE album_id = ? 
            ORDER BY id
        """
        
        # Сначала получаем все фотографии для фильтрации по статусу
        cursor.execute(base_query, (album_id,))
        all_photos = []
        filtered_photos = []
        
        for row in cursor:
            if row and row[0] is not None:
                photo_id = row[0]
                photo_url = row[1]
                photo_date = row[2]
                
                # Проверяем статус в Milvus
                is_processed = False
                if milvus_collection:
                    try:
                        milvus_expr = f'photo_id == {photo_id}' if isinstance(photo_id, (int, float)) else f'photo_id == "{str(photo_id).replace("\\", "\\\\").replace('"', '\\"')}"'
                        check_res = milvus_collection.query(expr=milvus_expr, output_fields=["photo_id"], limit=1)
                        if check_res:
                            is_processed = True
                    except Exception as e:
                        print(f"[WARNING] Ошибка проверки статуса фото {photo_id} в Milvus: {e}")
                
                status = "processed" if is_processed else "not_processed"
                
                photo_data = {
                    "id": photo_id,
                    "url": photo_url,
                    "date": photo_date,
                    "status": status
                }
                
                all_photos.append(photo_data)
                
                # Применяем фильтр по статусу
                if not filtering_needed or (filter_for_processed == is_processed):
                    filtered_photos.append(photo_data)
        
        # Вычисляем общее количество фотографий после фильтрации
        total_count = len(filtered_photos)
        
        # Применяем пагинацию
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_photos = filtered_photos[start_idx:end_idx] if start_idx < total_count else []
        
        print(f"[API_DEBUG] /api/albums/{album_id}/photos: Всего найдено {len(all_photos)} фото, после фильтрации: {total_count}, на текущей странице: {len(paged_photos)}")
        
        return {
            "photos": paged_photos,
            "pagination": {
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size if total_count > 0 else 1
            }
        }
        
    except sqlite3.Error as e:
        print(f"[API_ERROR] /api/albums/{album_id}/photos: Ошибка SQLite: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
    except Exception as e:
        print(f"[API_ERROR] /api/albums/{album_id}/photos: Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")
    finally:
        if conn:
            conn.close()

@app.post("/api/photos/mark")
async def mark_photos(request_data: MarkPhotosRequest):
    """
    Эндпоинт для разметки фотографий.
    Если mark_as_processed=True, запускает полную обработку изображений:
    - обнаружение лиц
    - получение эмбеддингов
    - запись в Milvus
    
    Если mark_as_processed=False, удаляет записи из Milvus для выбранных фото.
    """
    global milvus_collection
    
    if not milvus_collection:
        print("[ERROR /api/photos/mark] Milvus collection не инициализирована")
        raise HTTPException(status_code=503, detail="Сервис Milvus недоступен")
    
    if not request_data.photo_ids:
        raise HTTPException(status_code=400, detail="Не указаны ID фотографий для разметки")
    
    print(f"[DEBUG /api/photos/mark] Запрос на разметку {len(request_data.photo_ids)} фотографий, mark_as_processed={request_data.mark_as_processed}")
    
    results = {
        "total": len(request_data.photo_ids),
        "successful": 0,
        "failed": 0,
        "failures": [],
        "faces_found": 0
    }
    
    db_path = getattr(config, 'URL_FILE_PATH', None)
    if not db_path or not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="База данных не найдена на сервере")
    
    try:
        # Подготавливаем данные для обработки
        image_data_tuples = []
        photos_to_delete = []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for photo_id in request_data.photo_ids:
            try:
                if request_data.mark_as_processed:
                    # Получаем информацию о файле из базы данных
                    cursor.execute("SELECT url, date FROM photos WHERE id = ?", (photo_id,))
                    row = cursor.fetchone()
                    
                    if not row:
                        raise Exception(f"Фото с ID {photo_id} не найдено в базе данных")
                    
                    photo_url, photo_date = row
                    
                    # Генерируем ожидаемый локальный путь, используя ту же логику, что и при скачивании
                    local_download_base = getattr(config, 'LOCAL_DOWNLOAD_PATH', 'data/downloaded_images')
                    expected_local_file_path = get_local_path_for_url(str(photo_id), photo_url, local_download_base)
                    
                    if not os.path.isfile(expected_local_file_path):
                        raise Exception(f"Файл не найден локально: {expected_local_file_path}")
                    
                    # Форматируем дату
                    photo_date_iso = None
                    if photo_date is not None:
                        try:
                            photo_date_iso = datetime.datetime.fromtimestamp(int(photo_date)).isoformat()
                        except ValueError:
                            print(f"[WARNING] Некорректное значение даты {photo_date} для photo_id {photo_id}")
                    
                    # Добавляем в список для обработки
                    image_data_tuples.append((str(photo_id), expected_local_file_path, photo_date_iso))
                else:
                    # Если помечаем как необработанное, добавляем в список для удаления
                    photos_to_delete.append(photo_id)
            except Exception as e:
                print(f"[ERROR /api/photos/mark] Ошибка при подготовке photo_id={photo_id}: {e}")
                results["failed"] += 1
                results["failures"].append({"photo_id": photo_id, "error": str(e)})
        
        conn.close()
        
        # Обработка фотографий
        if request_data.mark_as_processed and image_data_tuples:
            # Создаем callback для получения прогресса обработки
            progress_updates = []
            
            async def mark_progress_callback(
                photo_id: Any = None, 
                url: str = "", 
                image_base64: Optional[str] = None, 
                status: str = "", 
                faces_count: Optional[int] = None,
                **kwargs
            ):
                if status == "extraction_completed" and faces_count is not None:
                    results["faces_found"] += faces_count
                
                # Сохраняем прогресс для возможного использования
                progress_updates.append({
                    "photo_id": photo_id,
                    "status": status,
                    "faces_count": faces_count
                })
            
            # Импортируем функцию обработки
            from pipeline_core import process_and_store_faces_async
            
            # Запускаем обработку
            await process_and_store_faces_async(
                milvus_collection=milvus_collection,
                image_data_tuples=image_data_tuples, 
                progress_callback=mark_progress_callback,
                skip_existing=False,  # Всегда обрабатываем заново
                sequential_mode=True  # Используем последовательный режим для выбранных фотографий
            )
            
            # Подсчитываем успешные операции (все что не в failures)
            results["successful"] = results["total"] - results["failed"]
            
        # Удаление фотографий из Milvus (если mark_as_processed=False)
        if not request_data.mark_as_processed and photos_to_delete:
            for photo_id in photos_to_delete:
                try:
                    # Формируем выражение для поиска в Milvus
                    milvus_expr = f'photo_id == {photo_id}' if isinstance(photo_id, (int, float)) else f'photo_id == "{str(photo_id).replace("\\", "\\\\").replace('"', '\\"')}"'
                    
                    # Проверяем существование фото в Milvus
                    check_res = milvus_collection.query(expr=milvus_expr, output_fields=["photo_id"], limit=1)
                    
                    if check_res:
                        delete_result = milvus_collection.delete(milvus_expr)
                        if delete_result and delete_result.delete_count > 0:
                            results["successful"] += 1
                        else:
                            raise Exception(f"Удаление из Milvus не выполнено, результат: {delete_result}")
                    else:
                        # Если записи нет, считаем успешным
                        results["successful"] += 1
                        
                except Exception as e:
                    print(f"[ERROR /api/photos/mark] Ошибка при удалении photo_id={photo_id}: {e}")
                    results["failed"] += 1
                    results["failures"].append({"photo_id": photo_id, "error": str(e)})
        
        # Выполняем flush для сохранения изменений
        milvus_collection.flush()
        print(f"[DEBUG /api/photos/mark] Обработка завершена: {results['successful']} успешно, {results['failed']} с ошибками")
        
        return results
    
    except Exception as e:
        print(f"[ERROR /api/photos/mark] Критическая ошибка: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")

# --- Инициализация при старте приложения (если нужно) ---
@app.on_event("startup")
async def startup_event():
    global milvus_collection # Объявляем, что будем изменять глобальную переменную
    print("Запуск FastAPI приложения...")
    try:
        print("Инициализация DeepFace модели...")
        init_deepface_model() 
        print("DeepFace модель инициализирована.")
    except Exception as e:
        print(f"[WARN] Ошибка при инициализации DeepFace на старте: {e}")
        # Не критично для старта, но поиск/индексация не будут работать

    try:
        print("Инициализация соединения Milvus...")
        milvus_collection = init_milvus_connection() # <--- Инициализируем коллекцию при старте
        print("Соединение Milvus установлено и коллекция загружена.")
    except Exception as e:
        print(f"[ERROR] КРИТИЧЕСКАЯ ОШИБКА при инициализации Milvus на старте: {e}\n{traceback.format_exc()}")
        # Здесь можно решить, останавливать ли приложение или работать без Milvus
        milvus_collection = None 

# Убедимся, что переменная milvus_collection объявлена на уровне модуля до ее использования в эндпоинтах
milvus_collection: Optional[Collection] = None
total_images_for_current_run: int = 0