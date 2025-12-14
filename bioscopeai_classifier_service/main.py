"""Main entry point for BioScopeAI Classifier Service."""

import asyncio
import signal
import sys

from loguru import logger

from bioscopeai_classifier_service.config.logging_config import setup_logger
from bioscopeai_classifier_service.kafka.consumers.classification_consumer import (
    get_classification_job_consumer,
)
from bioscopeai_classifier_service.kafka.producers.result_producer import (
    get_result_producer,
)


class ClassifierService:
    """Main service orchestrator for the classifier microservice."""

    def __init__(self) -> None:
        """Initialize the classifier service."""
        self.shutdown_event = asyncio.Event()
        self.consumer = get_classification_job_consumer()
        self.producer = get_result_producer()

    async def start(self) -> None:
        """Start the classifier service."""
        logger.info("Starting BioScopeAI Classifier Service...")

        try:
            await self.producer.initialize()

            await self.consumer.start_consuming()
            logger.info("Classifier service started successfully")

            await self.shutdown_event.wait()

        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Service interrupted")
        except BaseException:
            logger.exception("Error starting classifier service")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the classifier service gracefully."""
        logger.info("Stopping BioScopeAI Classifier Service...")

        try:
            await self.consumer.stop_consuming()
            await self.producer.shutdown()
            logger.info("Classifier service stopped successfully")
        except BaseException:  # noqa: BLE001
            logger.exception("Error stopping classifier service")

    def handle_shutdown(self, sig: int, _frame: object) -> None:
        """Handle shutdown signals."""
        logger.info(
            f"Received signal {signal.Signals(sig).name}, initiating shutdown..."
        )
        self.shutdown_event.set()


async def run_service() -> None:
    """Run the classifier service."""
    service = ClassifierService()

    signal.signal(signal.SIGINT, service.handle_shutdown)
    signal.signal(signal.SIGTERM, service.handle_shutdown)

    await service.start()


def main() -> None:
    """Main entry point."""
    setup_logger()
    logger.info("=" * 60)
    logger.info("BioScopeAI Classifier Service")
    logger.info("=" * 60)

    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except BaseException:  # noqa: BLE001
        logger.exception("Fatal error in classifier service")
        sys.exit(1)
    finally:
        logger.info("Service shutdown complete")


if __name__ == "__main__":
    main()
