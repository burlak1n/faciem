import asyncio
import functools
import time
import tensorflow as tf
import numpy as np
import cv2
import os
from deepface import DeepFace
from loguru import logger
from typing import Callable, Optional, Any, List, Tuple, Dict
import importlib
import sys
from dataclasses import dataclass, field

# Предполагается, что config.py находится в том же каталоге или доступен
try:
    from . import config
except ImportError:
    import config

@dataclass
class DetectedFace:
    detector_name: str
    raw_data: Dict[str, Any] = field(repr=False)  # Store the full original dict from DeepFace

    @property
    def bounding_box(self) -> Optional[Dict[str, int]]:
        facial_area_data = self.raw_data.get('facial_area')
        if isinstance(facial_area_data, dict) and all(k in facial_area_data for k in ('x', 'y', 'w', 'h')):
            try:
                return {
                    'x': int(facial_area_data['x']), 'y': int(facial_area_data['y']),
                    'w': int(facial_area_data['w']), 'h': int(facial_area_data['h'])
                }
            except (ValueError, TypeError) as e:
                logger.warning(f"Детектор {self.detector_name}: Ошибка преобразования координат facial_area в int: {facial_area_data}. Ошибка: {e}")
                return None
        return None

    @property
    def confidence(self) -> Optional[float]:
        conf = self.raw_data.get('confidence')
        if conf is not None:
            try:
                return float(conf)
            except (ValueError, TypeError) as e:
                logger.warning(f"Детектор {self.detector_name}: Не удалось преобразовать confidence в float: {conf}. Ошибка: {e}")
                return None
        return None

    @property
    def face_image(self) -> Optional[np.ndarray]:
        img = self.raw_data.get('face')
        return img if isinstance(img, np.ndarray) else None

    def __str__(self) -> str:
        other_keys = {k: type(v).__name__ for k, v in self.raw_data.items() if k not in ['facial_area', 'confidence', 'face']}
        confidence_val = self.confidence
        confidence_str = f"{confidence_val:.4f}" if confidence_val is not None else "N/A"
        
        return (f"DetectedFace(detector='{self.detector_name}', "
                f"bounding_box={self.bounding_box}, confidence={confidence_str}, "
                f"face_shape={self.face_image.shape if self.face_image is not None else 'N/A'}, "
                f"other_raw_keys={other_keys})")

def check_tensorflow_gpu():
    """Проверяет доступность GPU для TensorFlow и выводит информацию."""
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            logger.info("Найдены следующие GPU, доступные для TensorFlow:")
            for device in gpu_devices:
                logger.info(str(device))
            logger.info("Попытка выполнить тестовую операцию на GPU...")
            try:
                with tf.device('/device:GPU:0'): 
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                logger.info(f"Тестовая операция на GPU выполнена успешно. Результат (форма): {c.shape}")
            except RuntimeError as e:
                logger.error(f"Ошибка при попытке тестовой операции на GPU: {e}")
            except Exception as e_op:
                logger.error(f"Непредвиденная ошибка во время тестовой операции на GPU: {e_op}")
        else:
            logger.warning("GPU не найдены TensorFlow. TensorFlow будет использовать CPU.")
    except ImportError:
        logger.error("TensorFlow не импортирован или не установлен. Проверка GPU невозможна.")
    except Exception as e:
        logger.error(f"Произошла ошибка во время проверки GPU для TensorFlow: {e}")

def init_deepface_model():
    """Инициализирует (прогревает) модель DeepFace."""
    check_tensorflow_gpu() 
    try:
        DeepFace.build_model(config.DEEPFACE_MODEL_NAME)
        logger.info(f"DeepFace модель '{config.DEEPFACE_MODEL_NAME}' и детектор '{config.DEEPFACE_DETECTOR_BACKEND}' готовы (или будут загружены при первом использовании).")
    except Exception as e:
        logger.error(f"Ошибка при инициализации/прогреве DeepFace: {e}")

# Функция для инициализации и проверки OpenCV
def init_opencv():
    """Инициализация OpenCV и проверка пути к данным"""
    logger.info(f"OpenCV версия: {cv2.__version__}")
    
    # Проверяем текущий путь к данным OpenCV
    opencv_path = cv2.data.haarcascades
    logger.info(f"Текущий путь к данным OpenCV: {opencv_path}")
    
    # Проверяем наличие файлов каскадов
    cascade_file = os.path.join(opencv_path, 'haarcascade_frontalface_default.xml')
    if os.path.exists(cascade_file):
        logger.info(f"Файл каскада найден: {cascade_file}")
    else:
        logger.warning(f"Файл каскада не найден: {cascade_file}")
        
        # Если указан пользовательский путь, проверяем его
        if config.OPENCV_DATA_PATH:
            user_cascade = os.path.join(config.OPENCV_DATA_PATH, 'haarcascade_frontalface_default.xml')
            if os.path.exists(user_cascade):
                logger.info(f"Файл каскада найден в пользовательском пути: {user_cascade}")
                # Добавляем путь к переменным OpenCV
                os.environ['OPENCV_HAAR_CASCADES_DIR'] = config.OPENCV_DATA_PATH
            else:
                logger.error(f"Файл каскада не найден и в пользовательском пути: {user_cascade}")

# Адаптированные helper-функции
# Теперь они принимают loop и deepface_semaphore как аргументы

async def run_extraction_task_with_semaphore(
    loop: asyncio.AbstractEventLoop,
    deepface_semaphore: asyncio.Semaphore,
    photo_id_param: Any, 
    url_param: str, 
    image_np_param: np.ndarray,
    image_base64_cb_param: Optional[str], 
    photo_date_param: Optional[str]
) -> Tuple[Any, str, Optional[str], Optional[str], Any]:
    """Обертка для DeepFace.extract_faces с семафором."""
    task_start_time = time.monotonic()
    try:
        logger.debug(f"Ожидание семафора для детекции лиц: {url_param} (ID: {photo_id_param})")
        async with deepface_semaphore:
            logger.debug(f"Семафор получен, запуск DeepFace.extract_faces для: {url_param} (ID: {photo_id_param})")
            df_op_start_time = time.monotonic()
            
            # Получаем значение alignment из активной конфигурации
            alignment = config.ACTIVE_MODEL_CONFIG.get("alignment", False)
            
            _partial_df_extract = functools.partial(
                DeepFace.extract_faces,
                detector_backend=config.DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=False,
                align=alignment  # Используем значение из конфигурации
            )
            logger.info(f"Запуск DeepFace.extract_faces с detector_backend={config.DEEPFACE_DETECTOR_BACKEND}, align={alignment}")
            try:
                detected_faces_list_result = await loop.run_in_executor(None, _partial_df_extract, image_np_param)
                logger.info(f"DeepFace.extract_faces успешно выполнен. Найдено лиц: {len(detected_faces_list_result) if detected_faces_list_result else 0}")
                if detected_faces_list_result:
                    for i, face in enumerate(detected_faces_list_result):
                        confidence = face.get('confidence', 0)
                        logger.info(f"Лицо {i}: confidence={confidence}, shape={face['face'].shape if 'face' in face else 'нет данных'}")
            except Exception as e:
                logger.error(f"Ошибка во время DeepFace.extract_faces: {e}")
                raise e
            
            # Сохраняем лица, если настройка включена
            if config.SAVE_EXTRACTED_FACES and detected_faces_list_result:
                await save_detected_faces(photo_id_param, detected_faces_list_result)
                
            df_op_duration_ms = (time.monotonic() - df_op_start_time) * 1000
            logger.debug(f"[BENCHMARK] DeepFace.extract_faces (ID: {photo_id_param}) заняла {df_op_duration_ms:.2f} мс.")
        task_duration_ms = (time.monotonic() - task_start_time) * 1000
        logger.debug(f"[BENCHMARK] Полная задача run_extraction_task (ID: {photo_id_param}, вкл. ожидание семафора) заняла {task_duration_ms:.2f} мс.")
        return photo_id_param, url_param, image_base64_cb_param, photo_date_param, detected_faces_list_result
    except Exception as e_extract_task:
        task_duration_ms = (time.monotonic() - task_start_time) * 1000
        logger.error(f"Ошибка в задаче детекции лиц (run_extraction_task_with_semaphore) для {url_param} (ID: {photo_id_param}): {e_extract_task}. Затрачено: {task_duration_ms:.2f} мс.")
        return photo_id_param, url_param, image_base64_cb_param, photo_date_param, e_extract_task

async def run_represent_task_with_semaphore(
    loop: asyncio.AbstractEventLoop,
    deepface_semaphore: asyncio.Semaphore,
    photo_id_meta_param: Any, 
    url_meta_param: str, 
    face_idx_mapping_param: int, 
    face_np_param: np.ndarray,
    photo_date_meta_param: Optional[str]
) -> Dict[str, Any]:
    """Обертка для DeepFace.represent с семафором и измерением времени."""
    task_start_time = time.monotonic()
    # df_op_start_time будет определен после получения семафора
    df_op_duration_ms = -1.0 # Инициализируем на случай ошибки до начала операции DF
    try:
        logger.debug(f"Ожидание семафора для эмбеддинга: {url_meta_param} (ID: {photo_id_meta_param}, FaceIdx: {face_idx_mapping_param})")
        async with deepface_semaphore:
            logger.debug(f"Семафор получен, запуск DeepFace.represent для: {url_meta_param} (ID: {photo_id_meta_param}, FaceIdx: {face_idx_mapping_param})")
            df_op_start_time = time.monotonic() # Начало самой операции DeepFace
            
            # Получаем значение alignment из активной конфигурации
            alignment = config.ACTIVE_MODEL_CONFIG.get("alignment", False)
            
            _partial_df_represent = functools.partial(
                DeepFace.represent,
                model_name=config.DEEPFACE_MODEL_NAME,
                enforce_detection=False, 
                detector_backend=config.DEEPFACE_DETECTOR_BACKEND, 
                align=alignment  # Используем значение из конфигурации
            )
            embedding_obj_list_task_result = await loop.run_in_executor(None, _partial_df_represent, face_np_param)
            df_op_end_time = time.monotonic() # Конец самой операции DeepFace
            df_op_duration_ms = (df_op_end_time - df_op_start_time) * 1000
        
        total_task_duration_ms = (time.monotonic() - task_start_time) * 1000
        logger.debug(f"[BENCHMARK] DeepFace.represent (ID: {photo_id_meta_param}, FaceIdx: {face_idx_mapping_param}) заняла {df_op_duration_ms:.2f} мс (внутри семафора).")
        logger.debug(f"[BENCHMARK] Полная задача run_represent_task (ID: {photo_id_meta_param}, FaceIdx: {face_idx_mapping_param}, вкл. ожидание семафора) заняла {total_task_duration_ms:.2f} мс.")
        
        return {
            "status": "success", "photo_id": photo_id_meta_param, "url": url_meta_param, 
            "face_idx": face_idx_mapping_param, "photo_date": photo_date_meta_param, 
            "embedding_obj_list": embedding_obj_list_task_result, "duration_ms": df_op_duration_ms
        }
    except Exception as e_repr_task:
        # Если ошибка произошла уже после начала операции DF, df_op_duration_ms будет посчитан
        # Иначе он останется -1 или будет временем до ошибки внутри семафора
        if 'df_op_start_time' in locals() and 'df_op_end_time' not in locals(): # Ошибка внутри with, после df_op_start_time
             df_op_duration_ms = (time.monotonic() - df_op_start_time) * 1000

        total_task_duration_ms = (time.monotonic() - task_start_time) * 1000
        log_msg = f"Ошибка в задаче эмбеддинга (run_represent_task_with_semaphore) для лица {face_idx_mapping_param} с {url_meta_param} (ID: {photo_id_meta_param}): {e_repr_task}. Затрачено (общ.): {total_task_duration_ms:.2f} мс."
        if df_op_duration_ms >= 0:
            log_msg += f" Время операции DF (до ошибки): {df_op_duration_ms:.2f} мс."
        logger.error(log_msg)
        
        return {
            "status": "error", "photo_id": photo_id_meta_param, "url": url_meta_param,
            "face_idx": face_idx_mapping_param, "photo_date": photo_date_meta_param,
            "error": e_repr_task, "duration_ms": df_op_duration_ms
        }

async def save_detected_faces(photo_id: Any, detected_faces: list) -> None:
    """
    Сохраняет обнаруженные лица в отдельные файлы в указанной директории.
    
    Args:
        photo_id: Идентификатор фотографии
        detected_faces: Список обнаруженных лиц от DeepFace.extract_faces
    """
    # Проверяем, включено ли сохранение лиц
    if not config.SAVE_EXTRACTED_FACES or not detected_faces:
        return
        
    try:
        # Создаем директорию для сохранения лиц, если она не существует
        os.makedirs(config.EXTRACTED_FACES_PATH, exist_ok=True)
        
        # Создаем поддиректорию для конкретного фото, чтобы организовать файлы
        photo_dir = os.path.join(config.EXTRACTED_FACES_PATH, str(photo_id))
        os.makedirs(photo_dir, exist_ok=True)
        
        for face_idx, face_data in enumerate(detected_faces):
            try:
                if 'face' not in face_data or face_data['face'] is None:
                    logger.warning(f"Пропуск сохранения лица {face_idx} для фото {photo_id}: данные лица отсутствуют")
                    continue
                    
                face_np = face_data['face']
                confidence = face_data.get('confidence', 0)
                
                # Проверка, что изображение не пустое и не чёрное
                if face_np.size == 0 or (np.mean(face_np) < 0.1 and not hasattr(config, 'DISABLE_DARK_CHECK')):  # Снижаем порог с 1 до 0.1
                    logger.warning(f"Пропуск сохранения лица {face_idx} для фото {photo_id}: изображение пустое или слишком тёмное (среднее значение {np.mean(face_np):.2f})")
                    continue
                
                # Проверяем размерность изображения
                if len(face_np.shape) != 3 or face_np.shape[2] != 3:
                    logger.warning(f"Пропуск сохранения лица {face_idx} для фото {photo_id}: неправильная размерность изображения {face_np.shape}")
                    continue
                
                # Проверка формата цветов (BGR в OpenCV)
                # Убедимся, что цвета в диапазоне [0, 255]
                if face_np.dtype != np.uint8:
                    if np.max(face_np) <= 1.0:  # Предполагаем [0, 1] float диапазон
                        face_np = (face_np * 255).astype(np.uint8)
                    else:
                        face_np = face_np.astype(np.uint8)
                
                # DeepFace может возвращать RGB, а OpenCV ожидает BGR для сохранения
                # Проверяем и конвертируем при необходимости
                # Это сложно определить автоматически, поэтому добавим настройку в конфиг
                # По умолчанию считаем, что DeepFace возвращает RGB
                if config.FACE_COLOR_CONVERT:
                    face_np_to_save = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                else:
                    face_np_to_save = face_np.copy()  # Используем копию, чтобы не менять оригинал
                
                # Выведем информацию о изображении для отладки
                logger.debug(f"Лицо {face_idx} для фото {photo_id}: shape={face_np.shape}, dtype={face_np.dtype}, mean={np.mean(face_np):.2f}, min={np.min(face_np)}, max={np.max(face_np)}")
                
                # Изменяем размер изображения, если задан размер в конфигурации
                if config.FACE_IMAGE_SIZE:
                    face_np_to_save = cv2.resize(face_np_to_save, (config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE))
                
                # Формируем имя файла с информацией о лице
                face_filename = f"{photo_id}_face_{face_idx}_conf_{confidence:.4f}.jpg"
                face_path = os.path.join(photo_dir, face_filename)
                
                # Сохраняем лицо в файл
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config.FACE_IMAGE_QUALITY]
                success = cv2.imwrite(face_path, face_np_to_save, encode_params)
                
                # Проверяем результат сохранения
                if success and os.path.exists(face_path) and os.path.getsize(face_path) > 0:
                    logger.debug(f"Лицо {face_idx} для фото {photo_id} сохранено в {face_path}")
                else:
                    logger.warning(f"Не удалось сохранить лицо {face_idx} для фото {photo_id} в {face_path}")
                    
            except Exception as e_face:
                logger.error(f"Ошибка при сохранении лица {face_idx} для фото {photo_id}: {e_face}")
                
    except Exception as e:
        logger.error(f"Ошибка при сохранении лиц для фото {photo_id}: {e}")

# NEW FUNCTION
async def run_detectors_on_image(image_path: str, detectors_to_use: Optional[List[str]] = None) -> List[DetectedFace]:
    """
    Обрабатывает изображение с использованием нескольких детекторов лиц и возвращает
    стандартизированный список обнаруженных лиц.
    """
    if detectors_to_use is None:
        # Список детекторов по умолчанию, можно вынести в config.py
        detectors_to_use = ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe", "fastmtcnn"]
        detectors_to_use = [
            'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
            'retinaface', 'mediapipe', 'yolov8', 'yolov11s',
            'yolov11n', 'yolov11m', 'yunet', 'centerface',
        ]

    logger.info(f"Обработка изображения: {image_path} с детекторами: {detectors_to_use}")
    
    image_np = cv2.imread(image_path)
    if image_np is None:
        logger.error(f"Не удалось загрузить изображение: {image_path}")
        return []

    all_detected_faces: List[DetectedFace] = []

    for detector_name in detectors_to_use:
        logger.info(f"Используем детектор: {detector_name}")
        try:
            # DeepFace.extract_faces возвращает список словарей
            # Каждый словарь содержит: 'face' (np.array), 'facial_area' (dict с x,y,w,h), 'confidence'
            # и потенциально другие ключи в зависимости от детектора
            extracted_data_list = await asyncio.to_thread(
                DeepFace.extract_faces,
                img_path=image_np,
                detector_backend=detector_name,
                enforce_detection=False, 
                align=False 
            )
            if extracted_data_list:
                for i, face_dict_from_deepface in enumerate(extracted_data_list):
                    if not isinstance(face_dict_from_deepface, dict):
                        logger.warning(f"  Детектор {detector_name}, элемент {i} не является словарем, пропуск. Тип: {type(face_dict_from_deepface)}")
                        continue
                    print(f"face_dict_from_deepface: {face_dict_from_deepface}")
                    face_obj = DetectedFace(
                        detector_name=detector_name,
                        raw_data=face_dict_from_deepface  # Сохраняем весь словарь
                    )
                    print(f"face_dict_from_deepface: {face_obj}")
                    all_detected_faces.append(face_obj)
                    
                    # Логирование основной информации через свойства
                    logger.debug(
                        f"  Детектор {detector_name}, Лицо {i}: "
                        f"bbox={face_obj.bounding_box}, "
                        f"confidence={f'{face_obj.confidence:.4f}' if face_obj.confidence is not None else 'N/A'}, "
                        f"shape={face_obj.face_image.shape if face_obj.face_image is not None else 'N/A'}."
                        # Можно добавить вывод face_obj.__str__() для более детального лога
                        # f" Details: {face_obj}" 
                    )
                    if face_obj.bounding_box is None and 'facial_area' in face_dict_from_deepface : # Логируем если bounding_box не был создан, но facial_area было
                         logger.warning(f"  Детектор {detector_name}, Лицо {i}: 'facial_area' присутствовало в raw_data, но bounding_box не удалось создать. raw_facial_area: {face_dict_from_deepface.get('facial_area')}")


            else:
                logger.info(f"Детектор {detector_name}: лица не найдены.")
        except Exception as e:
            logger.error(f"Ошибка при использовании детектора {detector_name}: {e}")
            # Можно добавить более специфичную обработку ошибок или передать ошибку дальше

    logger.info(f"Обнаружено всего {len(all_detected_faces)} лиц всеми указанными детекторами.")
    return all_detected_faces

# Добавляем новую функцию для тестирования детекторов лиц
async def test_face_detectors(image_path: str):
    """
    Функция для тестирования различных детекторов лиц на одном изображении.
    Позволяет выявить, какой детектор лучше работает на конкретных данных.
    
    Args:
        image_path: Путь к тестовому изображению
    """
    logger.info(f"Начинаем тест детекторов лиц для изображения: {image_path}")
    
    # Загружаем тестовое изображение
    image_np = cv2.imread(image_path)
    if image_np is None:
        logger.error(f"Не удалось загрузить изображение: {image_path}")
        return
    
    logger.info(f"Изображение загружено, размер: {image_np.shape}")
    
    # Список детекторов для тестирования
    detectors = ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe"]
    
    # Тестируем каждый детектор
    for detector in detectors:
        logger.info(f"Тестируем детектор: {detector}")
        try:
            # Используем низкий порог enforce_detection, чтобы избежать ошибок, если лицо не найдено
            faces = DeepFace.extract_faces(
                img_path=image_np, 
                detector_backend=detector,
                enforce_detection=False,
                align=False
            )
            
            if faces:
                logger.info(f"Детектор {detector}: найдено {len(faces)} лиц")
                for i, face in enumerate(faces):
                    confidence = face.get('confidence', 0)
                    face_np = face.get('face')
                    if face_np is not None:
                        logger.info(f"  Лицо {i}: confidence={confidence:.4f}, shape={face_np.shape}, mean_pixel={np.mean(face_np):.2f}")
                    else:
                        logger.warning(f"  Лицо {i}: данные лица отсутствуют")
            else:
                logger.warning(f"Детектор {detector}: лица не найдены")
        except Exception as e:
            logger.error(f"Ошибка при использовании детектора {detector}: {e}")
    
    logger.info("Тест детекторов завершен")
