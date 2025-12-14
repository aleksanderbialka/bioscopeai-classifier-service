import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import cast, Self

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError
from loguru import logger

from bioscopeai_classifier_service.config import KafkaSettings, settings


class BaseKafkaConsumer(ABC):
    """Abstract base class for Kafka consumers."""

    _instances: dict[type["BaseKafkaConsumer"], "BaseKafkaConsumer"] = {}

    def __new__(cls) -> Self:
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cast("Self", cls._instances[cls])

    def __init__(
        self,
    ) -> None:
        if hasattr(self, "_consumer"):
            logger.debug(f"{self.__class__.__name__} already initialized.")
            return

        self._consumer: AIOKafkaConsumer | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self.kafka_settings: KafkaSettings = settings.kafka
        self._stop_event: asyncio.Event = asyncio.Event()
        self.auto_commit_interval_ms: int = 60 * 1000
        self.enable_auto_commit: bool = False
        self._max_retries: int = 5
        self._retry_delay: float = 2.0

    # Abstract methods
    @abstractmethod
    async def process_message(self, message: str) -> None:
        """Process a single message (implement business logic here)."""

    @abstractmethod
    def _get_topic_name(self) -> str:
        """Get the Kafka topic name to subscribe to."""

    @abstractmethod
    def _get_group_id(self) -> str:
        """Get the Kafka consumer group ID."""

    # Base functionality
    async def _initialize(self) -> None:
        if self.is_kafka_ready:
            logger.debug(f"{self.__class__.__name__} already initialized.")
            return

        if self._consumer is None:
            self._consumer = self._create_base_consumer()
        # Retry logic for starting the consumer, in case of connection issues
        # Kafka may not be available immediately
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._consumer.start()
            except KafkaConnectionError:
                logger.exception(
                    f"Attempt {attempt}/{self._max_retries} to start consumer failed"
                )
                await asyncio.sleep(self._retry_delay)
            else:
                logger.info("Kafka consumer started successfully")
                return
        msg = "Failed to start Kafka consumer after multiple attempts."
        logger.error(msg)
        raise RuntimeError(msg)

    async def start_consuming(self) -> None:
        """Start consuming messages in background task."""
        if self._consumer_task is not None and not self._consumer_task.done():
            logger.warning(f"{self.__class__.__name__} already consuming")
            return

        await self._initialize()
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info(f"{self.__class__.__name__} started consuming")

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        logger.info(f"Stopping {self.__class__.__name__}...")
        self._stop_event.set()

        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await asyncio.wait_for(self._consumer_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                logger.info("Consumer task stopped")

        await self._shutdown()
        logger.info(f"{self.__class__.__name__} stopped")

    async def _consume_loop(self) -> None:
        """Internal loop for consuming messages."""
        try:
            async for message in self._consume_messages():
                await self.process_message(message)
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
            raise
        except Exception:
            logger.exception("Error in consumer loop")
            raise

    async def _consume_messages(self) -> AsyncGenerator[str]:
        """Internal loop for consuming messages."""
        if not self._consumer:
            logger.error("Consumer not initialized")
            return
        try:
            async for _msg in self._consumer:
                if self.should_stop_processing:
                    break
                try:
                    message_content = _msg.value.decode("utf-8").strip()
                    if message_content:
                        logger.debug(
                            f"Consumed message from topic {_msg.topic},"
                            f" partition {_msg.partition},"
                            f" offset {_msg.offset}: {message_content}"
                        )
                    yield message_content
                except UnicodeDecodeError:
                    logger.exception("Failed to decode message")
                    continue
                except Exception:  # noqa: BLE001
                    logger.exception("Error processing raw message")
                    continue
        except Exception:
            logger.exception("Error while consuming messages")
            raise

    async def _shutdown(self) -> None:
        """Shutdown connection if necessary"""
        self._stop_event.set()
        if self._consumer:
            try:
                await asyncio.wait_for(self._consumer.stop(), timeout=5.0)
                logger.info("Consumer stopped successfully")
            except TimeoutError:
                logger.exception("Consumer stop timed out")
            except Exception:  # noqa: BLE001
                logger.exception("Error stopping consumer")
            finally:
                self._consumer = None

    async def commit_message(self) -> None:
        """Commit the current message offset."""
        if self._consumer:
            try:
                await self._consumer.commit()
            except Exception:
                logger.exception("Failed to commit offset")
                raise

    def _create_base_consumer(self) -> AIOKafkaConsumer:
        return AIOKafkaConsumer(
            self._get_topic_name(),
            bootstrap_servers=self.kafka_settings.BOOTSTRAP_SERVERS,
            group_id=self._get_group_id(),
            enable_auto_commit=self.enable_auto_commit,
            auto_commit_interval_ms=self.auto_commit_interval_ms,
        )

    @property
    def is_kafka_ready(self) -> bool:
        """Check if Kafka consumer is ready to consume messages."""
        return self._consumer is not None and not self._consumer._closed

    @property
    def should_stop_processing(self) -> bool:
        """Check if processing should stop."""
        return self._stop_event.is_set()

    @classmethod
    def get_instance(cls) -> Self:
        """Get the singleton instance of this consumer class."""
        return cls()
