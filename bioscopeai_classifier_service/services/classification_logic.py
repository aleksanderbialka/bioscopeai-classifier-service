import json
from typing import Any, cast

import aiohttp
import cv2
import numpy as np
from loguru import logger

from bioscopeai_classifier_service.config import settings
from bioscopeai_classifier_service.kafka.producers.result_producer import (
    get_result_producer,
    ResultProducer,
)
from bioscopeai_classifier_service.model_processing.model_processing import (
    get_model_processing_service,
    ModelProcessingService,
)


HTTP_OK = 200


class ClassificationLogicService:
    """Service responsible for classification logic."""

    def __init__(self) -> None:
        self.model_processing_service: ModelProcessingService = (
            get_model_processing_service()
        )
        self.result_producer: ResultProducer = get_result_producer()
        self.http_session = aiohttp.ClientSession()
        self._base_url = settings.service_auth.CORE_API_BASE_URL
        self._auth_headers = self._prepare_auth_headers()

    def _prepare_auth_headers(self) -> dict[str, str]:
        """Prepare authentication headers for internal API calls."""
        token = settings.service_auth.SERVICE_TOKEN.get_secret_value()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_event(event: str) -> dict[str, Any]:
        try:
            payload: dict[str, Any] = json.loads(event)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in classification job event")
            msg = "Invalid JSON"
            raise ValueError(msg) from exc

        required_fields = {"image_id", "model_name"}
        missing = required_fields - payload.keys()
        if missing:
            msg = f"Missing required fields: {missing}"
            raise ValueError(msg)

        return payload

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            msg = "Failed to decode image"
            raise ValueError(msg)

        return image

    def get_presigned_url_api_path(self, image_id: str) -> str:
        """Get API path for fetching presigned URL."""
        return f"{self._base_url}/api/images/{image_id}/download"

    async def _get_presigned_url(self, image_id: str) -> str:
        """Fetch presigned URL from the API."""
        url: str = self.get_presigned_url_api_path(image_id)
        logger.debug(f"Fetching presigned URL from {url}")

        async with self.http_session.get(
            url, headers=self._auth_headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status != HTTP_OK:
                error_body = await response.text()
                logger.error(
                    f"Failed to fetch presigned URL | status={response.status} body={error_body}",
                )
                msg = f"Failed to fetch presigned URL: {response.status}"
                raise RuntimeError(msg)

            data = await response.json()
            presigned_url = data.get("url")
            if not presigned_url:
                msg = "Presigned URL not found in response"
                raise RuntimeError(msg)

            return cast("str", presigned_url)

    async def _fetch_image(self, image_id: str) -> bytes:
        """Fetch image from MinIO using presigned URL."""
        # First, get the presigned URL from the API
        presigned_url = await self._get_presigned_url(image_id)
        logger.debug("Fetching image from MinIO using presigned URL")

        # Fetch image directly from MinIO using presigned URL
        async with self.http_session.get(
            presigned_url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != HTTP_OK:
                error_body = await response.text()
                logger.error(
                    f"Failed to fetch image from MinIO | "
                    f"status={response.status} body={error_body}",
                )
                msg = f"Failed to fetch image from MinIO: {response.status}"
                raise RuntimeError(msg)

            return cast("bytes", await response.read())

    async def process_classification_job(
        self, classification_job_event: str
    ) -> dict[str, Any]:
        """
        Process a single classification job:
        - parse event
        - fetch image
        - preprocess
        - classify
        - return result
        """
        logger.info("Processing classification job event")

        payload = self._parse_event(classification_job_event)

        image_bytes = await self._fetch_image(image_id=payload["image_id"])
        image = self._decode_image(image_bytes)

        result = await self.model_processing_service.classify(
            image=image,
            image_id=payload["image_id"],
            model_name=payload["model_name"],
        )

        logger.info(
            f"Classification finished | image_id={result['image_id']} "
            f"label={result['label']} conf={result['confidence']:.4f}"
        )

        try:
            await self.result_producer.send_event(
                message={
                    "classification_id": payload["classification_id"],
                    "image_id": payload["image_id"],
                    "model_name": payload["model_name"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                },
            )
        except Exception:
            logger.exception("Failed to send classification result")
            raise

        return result


_classification_logic_service: ClassificationLogicService | None = None


def get_classification_logic_service() -> ClassificationLogicService:
    """Get singleton instance of ClassificationLogicService."""
    global _classification_logic_service
    if _classification_logic_service is None:
        logger.info("Initializing ClassificationLogicService (singleton)")
        _classification_logic_service = ClassificationLogicService()
    return _classification_logic_service
