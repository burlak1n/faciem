from deepface import DeepFace
import cv2
import numpy as np
from pprint import pprint
import os
import time # <--- Добавлен импорт

backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yolov11s',
    'yolov11n', 'yolov11m', 'yunet', 'centerface',
]
# detector = "yunet" # <-- Убираем жестко заданный детектор

# --- Интерактивный выбор детектора ---
def select_detector(available_backends):
    print("Доступные детекторы:")
    for i, backend_name in enumerate(available_backends):
        print(f"  {i + 1}: {backend_name}")

    while True:
        try:
            choice = input(f"Введите номер детектора (1-{len(available_backends)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_backends):
                selected = available_backends[choice_idx]
                print(f"Выбран детектор: {selected}\n")
                return selected
            else:
                print("Неверный номер. Попробуйте снова.")
        except ValueError:
            print("Пожалуйста, введите число.")
        except EOFError:
            print("\nОтмена выбора.")
            return None # Или можно выбрать дефолтный, если отмена нежелательна

detector = select_detector(backends)

if not detector:
    print("Детектор не выбран. Выход.")
    exit()
# --------------------------------------

# --- Интерактивный выбор align ---
def select_alignment():
    while True:
        choice = input("Включить выравнивание лиц (align=True)? (yes/no): ").lower()
        if choice in ['yes', 'y']:
            print("Выравнивание включено (align=True).\n")
            return True
        elif choice in ['no', 'n']:
            print("Выравнивание выключено (align=False).\n")
            return False
        else:
            return False
            print("Пожалуйста, введите 'yes' или 'no'.")

align = select_alignment()
# ---------------------------------

img_path1 = "data/downloaded_images/2dSzEavb9IE.jpg"
img_path2 = "data/downloaded_images/gwrTkS57h_4.jpg"

# obj = DeepFace.verify(
#   img1_path = img_path1, img2_path = img_path2, detector_backend = detector, align = align
# )
# print("\n--- Результат DeepFace.verify ---")
# pprint(obj)

# Проверка существования файлов перед использованием
if not os.path.exists(img_path1):
    print(f"ОШИБКА: Файл не найден: {os.path.abspath(img_path1)}")
    exit()
if not os.path.exists(img_path2):
    print(f"ОШИБКА: Файл не найден: {os.path.abspath(img_path2)}")
    exit()

# embedding_objs = DeepFace.represent(
#   img_path = img_path1, detector_backend = detector, align = align
# )
# print("\n--- Результат DeepFace.represent ---")
# pprint(embedding_objs)

print(f"Используем изображение 1: {os.path.abspath(img_path1)}")
print(f"Используем изображение 2: {os.path.abspath(img_path2)}")

demographies = DeepFace.analyze(
  img_path = img_path1, detector_backend = detector, align = align
)
print("\n--- Результат DeepFace.analyze ---")
pprint(demographies)

print("\nЗапуск DeepFace.extract_faces...")
start_time = time.perf_counter()
face_objs = DeepFace.extract_faces(
  img_path = img_path1, detector_backend = detector, align = align
)
end_time = time.perf_counter()
duration = end_time - start_time
print(f"DeepFace.extract_faces выполнено за: {duration:.4f} сек")

print(f"\n--- Результат DeepFace.extract_faces (Найдено лиц: {len(face_objs)}) ---")
pprint(face_objs)

# Отображение извлеченных лиц
if face_objs:
    print("\nОтображение извлеченных лиц (нажмите любую клавишу для закрытия окна)...")
    for i, face_data in enumerate(face_objs):
        face_img = face_data.get('face')
        if face_img is not None and isinstance(face_img, np.ndarray):
            
            face_img_processed = face_img.copy()
            
            # --- Преобразование типа данных ПЕРЕД cvtColor ---
            if face_img_processed.dtype != np.uint8:
                if np.issubdtype(face_img_processed.dtype, np.floating): # Проверка на любой float
                    # Если значения похожи на диапазон [0, 1]
                    if np.min(face_img_processed) >= 0.0 and np.max(face_img_processed) <= 1.0:
                        face_img_processed = (face_img_processed * 255)
                    # Обрезаем на всякий случай и конвертируем
                    face_img_processed = np.clip(face_img_processed, 0, 255).astype(np.uint8)
                else:
                    # Попытка конвертировать другие типы (если вдруг возникнут)
                    try:
                        face_img_processed = face_img_processed.astype(np.uint8)
                    except Exception as e_conv:
                         print(f"  Предупреждение: Не удалось конвертировать тип {face_img_processed.dtype} в uint8 для лица {i+1}: {e_conv}")
                         continue # Пропустить это лицо, если конвертация не удалась
            # --------------------------------------------------

            # DeepFace возвращает RGB, cv2 ожидает BGR
            # Теперь используем face_img_processed с типом uint8
            if len(face_img_processed.shape) == 3 and face_img_processed.shape[2] == 3:
                 try:
                      # Убедимся, что на вход идет uint8
                      if face_img_processed.dtype == np.uint8:
                          face_to_show = cv2.cvtColor(face_img_processed, cv2.COLOR_RGB2BGR)
                      else:
                          # Этого не должно произойти после кода выше, но для безопасности
                          print(f"  Предупреждение: Изображение лица {i+1} не uint8 ({face_img_processed.dtype}) перед cvtColor. Отображаем как есть.")
                          face_to_show = face_img_processed 
                 except cv2.error as e:
                     print(f"  Ошибка cv2.cvtColor для лица {i+1}: {e}")
                     print(f"  Параметры изображения: dtype={face_img_processed.dtype}, shape={face_img_processed.shape}")
                     face_to_show = face_img_processed # Показать как есть, если конвертация не удалась
            elif len(face_img_processed.shape) == 2 or (len(face_img_processed.shape) == 3 and face_img_processed.shape[2] == 1):
                 face_to_show = face_img_processed # Уже Ч/Б или одноканальное
            else:
                 print(f"  Неподдерживаемая форма изображения для лица {i+1}: {face_img_processed.shape}")
                 face_to_show = face_img_processed # Показать как есть
                     
            window_title = f"Face {i+1} (Detector: {detector}) Confidence: {face_data.get('confidence', 'N/A'):.2f}"
            cv2.imshow(window_title, face_to_show)
        else:
            print(f"Лицо {i+1}: нет данных изображения.")
            
    cv2.waitKey(0) # Ждать нажатия любой клавиши
    cv2.destroyAllWindows() # Закрыть все окна OpenCV
else:
    print("\nЛица не были извлечены.")