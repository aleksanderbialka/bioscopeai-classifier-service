from loguru import logger

from bioscopeai_classifier_service.services import (
    ClassificationLogicService,
    get_classification_logic_service,
)

from .base_consumer import BaseKafkaConsumer


class ClassificationJobConsumer(BaseKafkaConsumer):
    """Kafka consumer for classification results."""

    def __init__(self) -> None:
        super().__init__()
        self.classification_logic_service: ClassificationLogicService = (
            get_classification_logic_service()
        )

    async def process_message(self, message: str) -> None:
        """Process a single classification result message."""
        try:
            await self.classification_logic_service.process_classification_job(
                classification_job_event=message
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to process classification job message")
        else:
            await self.commit_message()

    def _get_topic_name(self) -> str:
        return self.kafka_settings.CLASSIFICATION_JOBS_TOPIC

    def _get_group_id(self) -> str:
        return self.kafka_settings.CLASSIFICATION_CONSUMER_GROUP


def get_classification_job_consumer() -> ClassificationJobConsumer:
    """Get the singleton instance of ClassificationJobConsumer."""
    return ClassificationJobConsumer.get_instance()
