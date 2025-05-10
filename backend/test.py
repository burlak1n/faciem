# Примерный код создания таблицы SQLite
import sqlite3

from config import URL_FILE_PATH


conn = sqlite3.connect(URL_FILE_PATH)
cursor = conn.cursor()

cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_id ON photos(id)")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_url ON photos(url)")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date)")

conn.commit()
conn.close()