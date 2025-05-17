import sys
from loguru import logger

LOG_FILE = "app.log" # Используем общее имя файла лога

def configure_logger():
    """Настраивает глобальный логгер Loguru."""
    logger.remove() # Удаляем стандартный обработчик
    logger.add(
        sys.stderr, 
        level="INFO", # Логируем INFO и выше в консоль
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        LOG_FILE, 
        rotation="10 MB", # Ротация лог-файла
        level="DEBUG", # Логируем DEBUG и выше в файл
        enqueue=True, # Асинхронная запись для производительности
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.info("Логгер сконфигурирован.")

# Настраиваем логгер при импорте модуля
configure_logger()

# Экспортируем настроенный экземпляр
# Другие модули будут импортировать 'logger' из этого файла
# from .logging_config import logger 