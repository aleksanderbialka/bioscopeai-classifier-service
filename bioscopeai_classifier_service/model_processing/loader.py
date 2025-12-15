from functools import lru_cache
from pathlib import Path

from keras.models import load_model
from loguru import logger

from bioscopeai_classifier_service.config import settings


@lru_cache(maxsize=1)
def load_classifier_model():  # type: ignore[no-untyped-def]
    """Load and cache the classification model."""
    logger.info("Loading classification model into memory...")
    model_path = Path(settings.ml_model.MODEL_PATH)

    if not model_path.exists():
        error_msg = f"Model file not found at: {model_path}"
        raise FileNotFoundError(error_msg)
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    return model
