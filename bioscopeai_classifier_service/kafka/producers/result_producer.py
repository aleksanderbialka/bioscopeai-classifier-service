from typing import Any

from loguru import logger

from .base_producer import BaseKafkaProducer


class ResultProducer(BaseKafkaProducer):
    """Kafka producer for classification job messages."""

    def __init__(self) -> None:
        super().__init__()
        self._topic_prefix: str = self.kafka_settings.CLASSIFICATION_JOBS_TOPIC

    async def send_event(self, message: dict[str, Any]) -> None:
        """Send a classification job message to the specified Kafka topic."""
        if self._producer:
            self._topic = self.kafka_settings.CLASSIFICATION_RESULTS_TOPIC
            try:
                await self._producer.send_and_wait(
                    topic=self._topic,
                    value=message,
                )
                logger.debug(f"Sent event to topic {self._topic}: {message}")
            except Exception:
                logger.exception("Failed to send event")
                raise
        else:
            msg = "Producer is not initialized."
            logger.error(msg)
            raise RuntimeError(msg)


def get_result_producer() -> ResultProducer:
    """Get the singleton instance of ResultProducer."""
    return ResultProducer.get_instance()
