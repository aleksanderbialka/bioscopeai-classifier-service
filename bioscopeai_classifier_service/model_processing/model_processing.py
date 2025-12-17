from typing import Any

import numpy as np
from loguru import logger

from bioscopeai_classifier_service.config import settings

from .inference import run_inference
from .loader import load_classifier_model, load_metadata
from .preprocess import preprocess_image


class ModelProcessingService:
    """
    Stateless ML inference service.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = load_metadata()
        self.model = load_classifier_model()
        self.class_names: list[str] = self.metadata.get("output", {}).get(
            "classes", settings.ml_model.CLASS_NAMES
        )
        self._validate_configuration()

        logger.info(
            f"ModelProcessingService initialized | "
            f"model={self.metadata.get('model_name')} "
            f"v{self.metadata.get('model_version')} | "
            f"classes={len(self.class_names)}"
        )

    def _validate_configuration(self) -> None:
        """Validate that config matches metadata."""
        # Check class names
        config_classes = settings.ml_model.CLASS_NAMES
        if config_classes != self.class_names:
            logger.warning(
                "Config class names differ from metadata. Using metadata classes. "
                f"Config: {len(config_classes)}, Metadata: {len(self.class_names)}"
            )

        # Check image size
        metadata_size = tuple(self.metadata.get("preprocessing", {}).get("resize", []))
        config_size = settings.ml_model.IMG_SIZE
        if metadata_size and metadata_size != config_size:
            logger.warning(
                f"Image size mismatch: Config={config_size}, Metadata={metadata_size}"
            )

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

        input_tensor = preprocess_image(image=image, metadata=self.metadata)

        inference_result: dict[str, Any] = run_inference(
            model=self.model,
            input_tensor=input_tensor,
            class_names=self.class_names,
        )
        top_k = self.metadata.get("postprocessing", {}).get("top_k", 1)

        result = {
            "image_id": image_id,
            "label": inference_result["label"],
            "confidence": inference_result["confidence"],
            "model_name": model_name,
            "status": "success",
        }
        if top_k > 1:
            result["top_predictions"] = inference_result.get("all_predictions", [])[
                :top_k
            ]
        return result


_service: ModelProcessingService | None = None


def get_model_processing_service() -> ModelProcessingService:
    global _service
    if _service is None:
        _service = ModelProcessingService()
    return _service
