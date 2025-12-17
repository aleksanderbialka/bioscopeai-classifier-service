from typing import Any

import cv2
import numpy as np
from loguru import logger

from bioscopeai_classifier_service.config import settings


def preprocess_image(image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """
    Preprocess raw BGR image according to model metadata.

    Args:
        image: Input image in BGR format (OpenCV default)
        metadata: Model metadata containing preprocessing config

    Returns:
        Preprocessed tensor ready for model inference
    """
    preprocessing = metadata.get("preprocessing", {})
    input_config = metadata.get("input", {})

    color_conversion = preprocessing.get("color_conversion", "BGR_to_RGB")
    if color_conversion == "BGR_to_RGB":
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
        logger.warning(f"Unknown color conversion: {color_conversion}, skipping")

    target_size = tuple(preprocessing.get("resize", settings.ml_model.IMG_SIZE))
    image_resized = cv2.resize(image_rgb, target_size)

    normalization = input_config.get("normalization", "scale_0_1")
    if normalization == "scale_0_1":
        tensor = image_resized.astype("float32") / 255.0
    else:
        tensor = image_resized.astype("float32")
        logger.warning(f"Unknown normalization: {normalization}, using raw values")

    tensor = np.expand_dims(tensor, axis=0)

    logger.info(
        f"Preprocessed image: shape={tensor.shape}, dtype={tensor.dtype}, "
        f"range=[{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    return tensor
