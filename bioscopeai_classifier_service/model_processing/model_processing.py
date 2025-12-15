from typing import Any

import numpy as np
from loguru import logger

from bioscopeai_classifier_service.config import settings

from .loader import load_classifier_model
from .preprocess import preprocess_image


class ModelProcessingService:
    """
    Stateless ML inference service.
    """

    def __init__(self) -> None:
        self.model = load_classifier_model()
        self.class_names: list[str] = settings.ml_model.CLASS_NAMES

    async def classify(
        self,
        image: np.ndarray,
        image_id: str,
        model_name: str,
    ) -> dict[str, Any]:
        """
        Run classification on a single image.
        """
        logger.debug("Running inference | image_id=%s", image_id)

        input_tensor = preprocess_image(image)

        preds = self.model.predict(input_tensor, verbose=0)
        preds = preds[0]

        pred_idx = int(np.argmax(preds))
        pred_label = self.class_names[pred_idx]
        confidence = float(preds[pred_idx])

        return {
            "image_id": image_id,
            "label": pred_label,
            "confidence": confidence,
            "model_name": model_name,
            "status": "success",
        }


_service: ModelProcessingService | None = None


def get_model_processing_service() -> ModelProcessingService:
    global _service
    if _service is None:
        _service = ModelProcessingService()
    return _service
