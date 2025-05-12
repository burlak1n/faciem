import asyncio
import os
import sys
import cv2
import numpy as np
import time

# Добавляем текущую директорию в PYTHONPATH для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_processing import init_opencv, init_deepface_model, run_detectors_on_image

# Директория для сохранения обнаруженных лиц
SAVED_FACES_DIR = "data/detected_faces_output"
print(cv2.__version__)
print(hasattr(cv2, 'dnn'))
async def main():
    # Инициализация OpenCV и DeepFace
    init_opencv()
    # init_deepface_model() # <--- ВРЕМЕННО ЗАКОММЕНТИРУЕМ ЭТУ СТРОКУ
    
    # Укажите путь к тестовому изображению
    test_image_path = None
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # Если путь не указан, ищем любое изображение в директории с загруженными изображениями
        import config
        download_dir = config.LOCAL_DOWNLOAD_PATH
        
        files = os.listdir(download_dir)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) > 1:
            # Берем последний файл по списку из директории (порядок может быть непредсказуем без сортировки)
            # Пользовательский код был image_files[-1]
            test_image_path = os.path.join(download_dir, image_files[0])
        elif len(image_files) == 1:
            test_image_path = os.path.join(download_dir, image_files[0])
    
    if not test_image_path or not os.path.exists(test_image_path):
        print("Ошибка: Укажите путь к существующему изображению как аргумент командной строки")
        print("Пример: python test_detectors.py путь/к/изображению.jpg")
        return
    
    print(f"Запуск кросс-детекторной обработки на изображении: {test_image_path}")

    # Создаем директорию для сохранения, если ее нет
    os.makedirs(SAVED_FACES_DIR, exist_ok=True)
    original_image_basename = os.path.splitext(os.path.basename(test_image_path))[0]
    
    # detectors_to_run = ["fastmtcnn"]

    print("\nИзмерение времени выполнения run_detectors_on_image...")
    start_time_detection = time.perf_counter()
    all_detected_results = await run_detectors_on_image(test_image_path, detectors_to_use=None)
    end_time_detection = time.perf_counter()
    detection_duration = end_time_detection - start_time_detection
    print(f"Функция run_detectors_on_image выполнена за: {detection_duration:.4f} сек")
    
    # Этот print(f"all_detected_results: {all_detected_results}") может выводить очень много данных, особенно если лица - это большие массивы. 
    # Закомментирую его или сделаю более выборочным, если потребуется.
    # print(f"all_detected_results: {all_detected_results}") 

    if all_detected_results:
        print(f"\nОбщее количество обнаруженных объектов лиц (дополнительная фильтрация может потребоваться): {len(all_detected_results)}")
        
        print("\nИзмерение времени обработки и сохранения результатов...")
        start_time_processing = time.perf_counter()

        for idx, face_data in enumerate(all_detected_results):
            print(f"  Результат {idx + 1}:")
            print(f"    Детектор: {face_data.detector_name}")
            print(f"    Время выполнения детектора: {face_data.execution_time:.4f} сек")
            print(f"    Координаты (x,y,w,h): {face_data.bounding_box}")
            
            confidence_str = f"{face_data.confidence:.4f}" if face_data.confidence is not None else "N/A"
            print(f"    Уверенность: {confidence_str}")
            
            if face_data.face_image is not None:
                print(f"    Размер изобр. лица: {face_data.face_image.shape}")
                
                # Подготовка изображения лица к сохранению
                face_np = face_data.face_image.copy()
                
                # Конвертация типа данных и диапазона значений, если необходимо
                if face_np.dtype != np.uint8:
                    if np.max(face_np) <= 1.0 and np.min(face_np) >= 0: # Предполагаем диапазон [0,1] для float
                        face_np = (face_np * 255).astype(np.uint8)
                    else: # Если диапазон другой, просто пытаемся конвертировать тип
                        face_np = face_np.astype(np.uint8)
                
                # DeepFace обычно возвращает RGB, cv2.imwrite ожидает BGR
                # Убедимся, что изображение имеет 3 канала перед конвертацией цвета
                if len(face_np.shape) == 3 and face_np.shape[2] == 3:
                    face_to_save = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                elif len(face_np.shape) == 2 or (len(face_np.shape) == 3 and face_np.shape[2] == 1): # Если изображение уже Ч/Б
                    face_to_save = face_np 
                else:
                    print(f"      Предупреждение: Не удалось определить формат цветности для сохранения лица от детектора {face_data.detector_name}. Попытка сохранить как есть.")
                    face_to_save = face_np

                # Сохранение изображения лица
                try:
                    save_filename = f"{original_image_basename}_detector_{face_data.detector_name}_face_{idx}.jpg"
                    save_path = os.path.join(SAVED_FACES_DIR, save_filename)
                    cv2.imwrite(save_path, face_to_save)
                    print(f"      Изображение лица сохранено: {save_path}")
                except Exception as e_save:
                    print(f"      Ошибка при сохранении изображения лица: {e_save}")
            else:
                print(f"    Изобр. лица: отсутствует")
        
        end_time_processing = time.perf_counter()
        processing_duration = end_time_processing - start_time_processing
        print(f"\nОбработка и сохранение {len(all_detected_results)} результатов заняли: {processing_duration:.4f} сек")
    else:
        print("Ни один из указанных детекторов не нашел лиц на изображении.")

    print("Кросс-детекторная обработка завершена")

if __name__ == "__main__":
    asyncio.run(main()) 