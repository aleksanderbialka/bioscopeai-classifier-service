import cv2
import numpy as np

from bioscopeai_classifier_service.config import settings


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess raw BGR image for EfficientNetV2.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, settings.ml_model.IMG_SIZE)

    tensor = image_resized.astype("float32") / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    return tensor
