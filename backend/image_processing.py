import asyncio
import functools
import time
import tensorflow as tf
import numpy as np
from deepface import DeepFace
from loguru import logger
from typing import Callable, Optional, Any, List, Tuple, Dict

# Предполагается, что config.py находится в том же каталоге или доступен
try:
    from . import config
except ImportError:
    import config

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
            _partial_df_extract = functools.partial(
                DeepFace.extract_faces,
                detector_backend=config.DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            detected_faces_list_result = await loop.run_in_executor(None, _partial_df_extract, image_np_param)
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
            _partial_df_represent = functools.partial(
                DeepFace.represent,
                model_name=config.DEEPFACE_MODEL_NAME,
                enforce_detection=False, 
                detector_backend=config.DEEPFACE_DETECTOR_BACKEND, 
                align=True 
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
