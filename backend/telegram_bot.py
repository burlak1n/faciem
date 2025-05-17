import asyncio
import os
import aiohttp
import aiosqlite
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.callback_data import CallbackData
from loguru import logger
import sys

# Импорт вашего FaceManager
try:
    from face_manager import FaceManager
    # Константы из вашего milvus.py, если нужны для форматирования ответа
    from milvus import DEFAULT_PERSON_ID
except ImportError:
    logger.error("Не удалось импортировать FaceManager или константы Milvus. Убедитесь, что скрипт запускается из правильной директории и PYTHONPATH настроен.")
    sys.exit(1)

# --- Конфигурация ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("Переменная окружения TELEGRAM_BOT_TOKEN не установлена.")
    sys.exit(1)

DATABASE_PATH = "vk_photos_data.db" # Путь к вашей SQLite базе данных VK
DOWNLOAD_BASE_PATH = os.path.join("data", "downloaded_vk_photos") # Базовый путь для скачивания альбомов
TEMP_PHOTO_PATH = os.path.join("data", "temp_telegram_photos") # Временное хранение фото для поиска

# Создаем директории, если их нет
os.makedirs(DOWNLOAD_BASE_PATH, exist_ok=True)
os.makedirs(TEMP_PHOTO_PATH, exist_ok=True)

# --- Настройка Логгера ---
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("telegram_bot.log", level="DEBUG", rotation="5 MB")

# --- Инициализация Aiogram и FaceManager ---
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
face_manager = FaceManager() # Инициализируем ваш FaceManager

# --- CallbackData ---
class GroupCallback(CallbackData, prefix="group"):
    group_screen_name: str

class AlbumCallback(CallbackData, prefix="album"):
    album_id: int # Оставляем только ID альбома


# --- Функции для работы с БД (vk_photos_data.db) ---
async def get_vk_groups() -> list[tuple[str]]: # Возвращает список кортежей со screen_name
    """Получает список screen_name групп из БД VK."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            # Запрашиваем только screen_name, так как это PK и единственная колонка
            cursor = await db.execute("SELECT screen_name FROM groups ORDER BY screen_name")
            groups_data = await cursor.fetchall()
            # Преобразуем список кортежей [(screen_name,), ...] в список строк [screen_name, ...]
            # Хотя для кнопок подойдет и кортеж, для единообразия можно так
            groups = [row[0] for row in groups_data] if groups_data else []
            return groups
    except aiosqlite.Error as e:
        logger.error(f"Ошибка БД при получении групп: {e}")
        return []

async def get_vk_albums_for_group(group_screen_name: str) -> list[tuple[int, str]]:
    """Получает ID и название альбомов для указанной группы."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            sql_query = "SELECT id, title FROM albums WHERE group_screen_name = ? ORDER BY title"
            cursor = await db.execute(sql_query, (group_screen_name,))
            albums = await cursor.fetchall()
            return albums if albums else []
    except aiosqlite.Error as e:
        logger.error(f"Ошибка БД при получении альбомов для группы {group_screen_name}: {e}")
        return []

async def get_album_details(album_id: int) -> tuple[str, str] | None:
    """Получает group_screen_name и title для указанного album_id."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            sql_query = "SELECT group_screen_name, title FROM albums WHERE id = ?"
            cursor = await db.execute(sql_query, (album_id,))
            details = await cursor.fetchone() 
            if details:
                return details 
            else:
                logger.warning(f"Детали для альбома ID {album_id} не найдены.")
                return None
    except aiosqlite.Error as e:
        logger.error(f"Ошибка БД при получении деталей альбома {album_id}: {e}")
        return None

async def get_photo_urls_for_album(album_id: int) -> list[tuple[int, str]]:
    """Получает ID и URL фотографий для указанного альбома."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            cursor = await db.execute("SELECT id, url FROM photos WHERE album_id = ?", (album_id,))
            photos = await cursor.fetchall()
            return photos if photos else []
    except aiosqlite.Error as e:
        logger.error(f"Ошибка БД при получении фото для альбома {album_id}: {e}")
        return []

async def update_photo_local_path(photo_id: int, local_path: str):
    """Обновляет путь к скачанному файлу в БД."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("UPDATE photos SET local_path = ? WHERE id = ?", (local_path, photo_id))
            await db.commit()
    except aiosqlite.Error as e:
        logger.error(f"Ошибка БД при обновлении local_path для фото {photo_id}: {e}")


# --- Обработчики команд ---
@dp.message(CommandStart())
@dp.message(Command("groups"))
async def send_groups_list(message: Message):
    """Отправляет список групп в виде кнопок."""
    groups_screen_names = await get_vk_groups()
    if not groups_screen_names:
        await message.answer("Группы не найдены в базе данных.")
        return

    builder = InlineKeyboardBuilder()
    for screen_name in groups_screen_names:
        builder.button(
            text=screen_name, # Используем screen_name как текст кнопки
            callback_data=GroupCallback(group_screen_name=screen_name).pack()
        )
    builder.adjust(1)
    await message.answer("Выберите группу:", reply_markup=builder.as_markup())

# --- Обработчики Callback ---
@dp.callback_query(GroupCallback.filter())
async def process_group_selection(query: CallbackQuery, callback_data: GroupCallback):
    """Обрабатывает выбор группы и показывает ее альбомы."""
    group_screen_name = callback_data.group_screen_name
    albums = await get_vk_albums_for_group(group_screen_name)

    if not albums:
        await query.message.edit_text(f"Альбомы для группы '{group_screen_name}' не найдены.")
        await query.answer()
        return

    builder = InlineKeyboardBuilder()
    for album_id, album_title_text in albums: # album_title_text для отображения в кнопке
        builder.button(
            text=album_title_text, # Отображаем полное название
            callback_data=AlbumCallback(album_id=album_id).pack() # Передаем только ID
        )
    builder.adjust(1)
    await query.message.edit_text(f"Альбомы группы '{group_screen_name}':", reply_markup=builder.as_markup())
    await query.answer()


async def download_photo(session: aiohttp.ClientSession, url: str, save_path: str) -> bool:
    """Скачивает одну фотографию."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(await response.read())
                logger.debug(f"Фото успешно скачано: {save_path}")
                return True
            else:
                logger.warning(f"Ошибка скачивания {url}: статус {response.status}")
                return False
    except Exception as e:
        logger.error(f"Исключение при скачивании {url}: {e}")
        return False

@dp.callback_query(AlbumCallback.filter())
async def process_album_selection(query: CallbackQuery, callback_data: AlbumCallback):
    """Обрабатывает выбор альбома и запускает его скачивание."""
    album_id = callback_data.album_id

    album_details = await get_album_details(album_id)
    if not album_details:
        await query.message.edit_text(f"Не удалось получить детали для альбома ID {album_id}. Загрузка отменена.")
        await query.answer()
        return
    
    group_screen_name, album_title = album_details # album_title теперь только для сообщения пользователю

    await query.message.edit_text(f"Начинаю загрузку альбома '{album_title}' (ID: {album_id}) из группы '{group_screen_name}'...")
    await query.answer()

    photos_to_download = await get_photo_urls_for_album(album_id)
    if not photos_to_download:
        await query.message.answer(f"Фотографии в альбоме '{album_title}' (ID: {album_id}) не найдены.")
        return

    # Используем group_screen_name (если он безопасен) и album_id для пути
    # Убедимся, что group_screen_name не содержит небезопасных символов (хотя обычно screen_name безопасны)
    safe_group_screen_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in group_screen_name).rstrip('_')
    if not safe_group_screen_name: # На случай, если screen_name был полностью из спецсимволов
        safe_group_screen_name = "unknown_group"

    # Имя папки альбома будет его ID
    album_folder_name = str(album_id)
    
    album_download_path = os.path.join(DOWNLOAD_BASE_PATH, safe_group_screen_name, album_folder_name)
    os.makedirs(album_download_path, exist_ok=True)

    total_photos = len(photos_to_download)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for photo_db_id, photo_url in photos_to_download:
            if not photo_url:
                logger.warning(f"Пропуск фото ID {photo_db_id} в альбоме {album_id} из-за отсутствия URL.")
                total_photos -=1 
                continue

            # Формируем имя файла: IDфото_оригинальноеимя (без параметров URL)
            original_filename = os.path.basename(photo_url.split('?')[0])
            photo_filename = f"{photo_db_id}_{original_filename}"
            
            # Добавляем расширение .jpg если его нет или оно некорректное, или если вообще нет расширения
            if not os.path.splitext(photo_filename)[1] or \
               not any(photo_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                 photo_filename += ".jpg"

            save_path = os.path.join(album_download_path, photo_filename)
            tasks.append(asyncio.ensure_future(download_photo(session, photo_url, save_path)))
        
        results = await asyncio.gather(*tasks)

    download_count = sum(1 for r in results if r) 

    if download_count > 0:
        await query.message.answer(
            f"Скачивание альбома '{album_title}' (ID: {album_id}) завершено. "
            f"Загружено {download_count} из {total_photos} фото в папку:\n`{album_download_path}`",
            parse_mode="Markdown" # Для отображения пути как кода
        )
    else:
        await query.message.answer(f"Не удалось скачать ни одного фото для альбома '{album_title}' (ID: {album_id}).")


# --- Обработчик фотографий для поиска ---
@dp.message(F.photo)
async def handle_photo_for_search(message: Message):
    """Обрабатывает полученную фотографию и выполняет поиск лиц."""
    if not face_manager.client:
        await message.answer("Ошибка: FaceManager не инициализирован (проблема с Milvus). Поиск невозможен.")
        return

    photo = message.photo[-1] 
    unique_filename = f"{message.from_user.id}_{photo.file_unique_id}.jpg"
    temp_file_path = os.path.join(TEMP_PHOTO_PATH, unique_filename)

    await message.answer("Фото получено, начинаю поиск лиц...")
    try:
        await bot.download(file=photo.file_id, destination=temp_file_path)
        logger.info(f"Фото временно сохранено: {temp_file_path}")

        search_output = face_manager.search_face(query_image_path_or_url=temp_file_path)
        
        direct_hits = search_output.get("direct_hits", [])
        cluster_members = search_output.get("cluster_members", {})
        response_parts = []

        if not direct_hits:
            response_parts.append("Прямых совпадений (выше порога) не найдено.")
        else:
            final_person_details = {}
            for hit in direct_hits:
                pid = hit["person_id"]
                if pid == DEFAULT_PERSON_ID:
                    continue 
                
                details = final_person_details.setdefault(pid, {"direct_hits": [], "all_member_faces": set()})
                face_identifier = f"photo_{hit['photo_id']}_{hit['face_index']}" 
                details["direct_hits"].append({
                    "face_identifier": face_identifier,
                    "pk": hit["pk"],
                    "similarity": hit["similarity"],
                    "photo_id": hit["photo_id"],
                    "face_index": hit["face_index"],
                    "query_face_idx": hit["query_face_idx"]
                })

            for pid_str, members_list in cluster_members.items():
                pid = int(pid_str) 
                if pid == DEFAULT_PERSON_ID: continue
                details = final_person_details.setdefault(pid, {"direct_hits": [], "all_member_faces": set()})
                for member in members_list:
                    member_face_identifier = f"photo_{member['photo_id']}_{member['face_index']}"
                    details["all_member_faces"].add(member_face_identifier)
            
            if not final_person_details:
                response_parts.append("Найдены только некластеризованные лица:")
                limited_unclustered_hits = 0
                for hit in direct_hits: 
                    if hit["person_id"] == DEFAULT_PERSON_ID:
                        response_parts.append(
                            f"  - PhotoID: {hit['photo_id']}, FaceIdx: {hit['face_index']} (PK: {hit['pk']}, Схожесть: {hit['similarity']:.2f})"
                        )
                        limited_unclustered_hits += 1
                        if limited_unclustered_hits >=5:
                            response_parts.append("...и возможно другие некластеризованные.")
                            break
            else:
                response_parts.append(f"Найдено {len(final_person_details)} уникальных кластеризованных личностей:")
                for person_id, details_data in final_person_details.items():
                    response_parts.append(f"\n*Личность ID: {person_id}*")
                    
                    if details_data["direct_hits"]:
                        response_parts.append(f"  Прямые совпадения ({len(details_data['direct_hits'])}):")
                        sorted_hits = sorted(details_data["direct_hits"], key=lambda x: x['similarity'], reverse=True)
                        for i, hit_data in enumerate(sorted_hits):
                            if i < 5:
                                response_parts.append(
                                    f"    - {hit_data['face_identifier']} (PK: {hit_data['pk']}, Схожесть: {hit_data['similarity']:.2f}, запрос #{hit_data['query_face_idx']})"
                                )
                            elif i == 5:
                                response_parts.append("      ... и другие возможные прямые совпадения.")
                    
                    if details_data["all_member_faces"]:
                        sorted_member_identifiers = sorted(list(details_data["all_member_faces"]))
                        response_parts.append(f"  Все известные фото этой личности ({len(sorted_member_identifiers)}):")
                        for i, face_id_cleaned in enumerate(sorted_member_identifiers):
                            if i < 5: 
                                response_parts.append(f"    - {face_id_cleaned}")
                            elif i == 5:
                                response_parts.append("      ... и другие фото.")
                                break 
        
        response_text = "\n".join(response_parts)
        if not response_text: 
            response_text = "Поиск не дал результатов или произошла ошибка при форматировании."
        
        if len(response_text) > 4096:
            logger.warning(f"Ответ слишком длинный ({len(response_text)} символов), будет разбит.")
            for i in range(0, len(response_text), 4096):
                await message.answer(response_text[i:i+4096], parse_mode="Markdown") 
        else:
            await message.answer(response_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Ошибка при обработке фото для поиска: {e}", exc_info=True)
        await message.answer("Произошла ошибка при поиске лиц.")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Временный файл удален: {temp_file_path}")
            except OSError as e_rm:
                logger.error(f"Не удалось удалить временный файл {temp_file_path}: {e_rm}")


async def main():
    logger.info("Запуск телеграм-бота...")
    if not face_manager.client:
        logger.error("FaceManager не смог подключиться к Milvus. Бот не может корректно работать с поиском лиц.")

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main()) 