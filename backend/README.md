# Backend для Системы Обработки Фотографий с Поиском Лиц

## Описание
Backend-часть системы для автоматизированной обработки изображений с функциями поиска, идентификации и группировки лиц. Система использует современные технологии машинного обучения для распознавания лиц и векторную базу данных для эффективного поиска.

## Технологии
- **FastAPI** - современный веб-фреймворк для создания API
- **RetinaFace** - нейронная сеть для детекции лиц
- **FaceNet512** - модель для создания векторных представлений лиц
- **Milvus** - векторная база данных для эффективного поиска
- **SQLAlchemy** - ORM для работы с базами данных
- **FastStream** - асинхронная обработка задач
- **Prometheus** + **Grafana** - мониторинг и визуализация метрик
- **Loguru** - логирование

## Требования
- Python 3.12+
- CUDA (опционально, для GPU-ускорения)
- Docker и Docker Compose
- Redis (для брокера сообщений)

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/faciem.git
cd faciem/backend
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env файл
```

## Запуск
python face_database.py --add data/database_faces
### Локальный запуск
```bash
faciem\backend> python3.12 -m uvicorn web_server:app --reload --port 8000
```

### Запуск через Docker
```bash
docker-compose up -d
```

## API Endpoints

### Основные эндпоинты:
- `POST /upload/local` - загрузка фото из локальной директории
- `POST /search/upload` - поиск лиц по образцу
- `GET /clusters` - получение кластеров лиц
- `GET /metrics` - метрики системы

Подробная документация API доступна по адресу `/docs` после запуска сервера.

## Структура проекта
```
backend/
├── api/            # API endpoints
├── core/           # Основная логика
├── models/         # Модели данных
├── services/       # Сервисы (детекция лиц, эмбеддинги)
├── utils/          # Вспомогательные функции
├── config.py       # Конфигурация
├── web_server.py   # Точка входа
└── requirements.txt
```

## Разработка

### Тестирование
```bash
pytest
```

### Линтинг
```bash
flake8
black .
```

## Мониторинг
- Prometheus метрики доступны по адресу `/metrics`
- Grafana дашборды доступны по адресу `http://localhost:3000`
- Логи доступны в директории `logs/`

## Безопасность
- Все API endpoints защищены JWT-аутентификацией
- Чувствительные данные хранятся в переменных окружения
- Все внешние запросы проходят через HTTPS

## Лицензия
MIT 