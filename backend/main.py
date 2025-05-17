import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Убираем прямой импорт loguru
# from loguru import logger
# Импортируем настроенный логгер
from .logging_config import logger

# Импортируем сам роутер
from . import router as api_router 

def create_app() -> FastAPI:
    """Создает и настраивает экземпляр FastAPI."""
    app = FastAPI(
        title="Face Recognition API",
        description="API для добавления, поиска и кластеризации лиц с использованием DeepFace и Milvus.",
        version="0.1.0"
    )

    # Настройка CORS
    origins = [
        "*", # Разрешить все источники (НЕ для продакшена!)
        # "http://localhost:5173", 
        # "http://localhost:8000", 
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Подключаем маршруты из router.py
    # Можно добавить префикс, например, prefix="/api/v1"
    app.include_router(api_router.router)

    return app

# Создаем приложение
app = create_app()

# Запуск сервера (для локальной разработки)
if __name__ == "__main__":
    # Используем импортированный logger
    logger.info("Запуск FastAPI сервера с Uvicorn из main.py...")
    # host="0.0.0.0" делает сервер доступным извне
    # reload=True автоматически перезапускает сервер при изменениях кода
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True) 