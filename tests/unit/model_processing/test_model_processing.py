"""Unit tests for model_processing.py - ModelProcessingService orchestration."""

import pytest

from bioscopeai_classifier_service.model_processing.model_processing import (
    ModelProcessingService,
)


class TestModelProcessingService:
    def test_initialization_loads_model_and_metadata(
        self, mock_load_model, mock_load_metadata, monkeypatch
    ):
        ModelProcessingService._instance = None
        monkeypatch.setattr(
            "bioscopeai_classifier_service.model_processing.model_processing.load_classifier_model",
            mock_load_model,
        )
        monkeypatch.setattr(
            "bioscopeai_classifier_service.model_processing.model_processing.load_metadata",
            mock_load_metadata,
        )

        service = ModelProcessingService()

        assert service.model is not None
        assert service.metadata is not None
        assert isinstance(service.class_names, list)

    @pytest.mark.asyncio
    async def test_classify_returns_correct_structure(
        self, mock_model_processing_service, sample_image_bgr
    ):
        result = await mock_model_processing_service.classify(
            sample_image_bgr, "test-id", "test-model"
        )

        assert isinstance(result, dict)
        assert all(
            k in result
            for k in ["image_id", "label", "confidence", "model_name", "status"]
        )
        assert result["image_id"] == "test-id"
        assert result["model_name"] == "test-model"
        assert result["status"] == "success"

    def test_model_property_returns_loaded_model(self, mock_model_processing_service):
        assert mock_model_processing_service.model is not None
        assert hasattr(mock_model_processing_service.model, "predict")

    def test_metadata_property_returns_loaded_metadata(
        self, mock_model_processing_service
    ):
        assert mock_model_processing_service.metadata is not None
        assert isinstance(mock_model_processing_service.metadata, dict)

    def test_class_names_loaded_from_metadata(self, mock_model_processing_service):
        assert isinstance(mock_model_processing_service.class_names, list)
        assert len(mock_model_processing_service.class_names) > 0
