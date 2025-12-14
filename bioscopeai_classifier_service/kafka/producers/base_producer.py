import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, cast, Self

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from loguru import logger

from bioscopeai_classifier_service.config import KafkaSettings, settings


class BaseKafkaProducer(ABC):
    """Abstract base class for Kafka producers."""

    _instances: dict[type["BaseKafkaProducer"], "BaseKafkaProducer"] = {}

    def __new__(cls) -> Self:
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cast("Self", cls._instances[cls])

    def __init__(self) -> None:
        if hasattr(self, "_producer"):
            logger.debug(f"{self.__class__.__name__} already initialized.")
            return
        self._producer: AIOKafkaProducer | None = None
        self.kafka_settings: KafkaSettings = settings.kafka

    async def initialize(self) -> None:
        if self.is_initialized:
            logger.debug(f"{self.__class__.__name__} is already initialized.")
            return
        if self._producer is None:
            self._producer = self._create_base_producer()
        # Retry logic for starting the producer, in case of connection issues
        # Kafka may not be available immediately
        for attempt in range(1, 6):
            try:
                await self._producer.start()
            except KafkaConnectionError:
                logger.exception(
                    f"Attempt {attempt} to start {self.__class__.__name__} failed"
                )
                await asyncio.sleep(2)
            else:
                logger.info(f"{self.__class__.__name__} started successfully")
                return
        msg = f"Failed to start {self.__class__.__name__} after multiple attempts."
        logger.error(msg)
        raise RuntimeError(msg)

    @abstractmethod
    async def send_event(self, message: dict[str, Any]) -> None:
        """Send a message to the specified Kafka topic."""

    async def shutdown(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._producer = None

    def _create_base_producer(self) -> AIOKafkaProducer:
        return AIOKafkaProducer(
            bootstrap_servers=self.kafka_settings.BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode(),
        )

    @classmethod
    def get_instance(cls) -> Self:
        """Get the singleton instance of this producer class."""
        return cls()

    @property
    def is_initialized(self) -> bool:
        """Check if the producer has been initialized."""
        return self._producer is not None and not self._producer._closed
