from .model_processing import (
    get_model_processing_service,
    ModelProcessingService,
)


class ClassificationLogicService:
    """Service for processing classification results."""

    def __init__(self) -> None:
        self.model_processing_service: ModelProcessingService = (
            get_model_processing_service()
        )

    async def process_classification_job(self, classification_job_event: str) -> str:
        """Process a single classification job message."""
        event: str = classification_job_event
        return event


def get_classification_logic_service() -> ClassificationLogicService:
    return ClassificationLogicService()
