"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def pytest_configure() -> None:
    """Setup test environment before Pydantic Settings initialization."""
    test_config_path = Path(__file__).parent / "test-config.yaml"
    os.environ["CONFIG_FILE"] = str(test_config_path)
    print(
        f"\n[pytest] CONFIG_FILE: {test_config_path} (exists: {test_config_path.exists()})"
    )


@pytest.fixture
def mock_metadata() -> dict[str, Any]:
    """Standard model metadata for testing."""
    return {
        "model_name": "bioscopeai_classifier",
        "model_version": "1.0.0",
        "architecture": "ResNet50",
        "model_file": "bioscopeai_classifier_model.keras",
        "preprocessing": {
            "color_conversion": "BGR_to_RGB",
            "resize": [300, 300],
        },
        "input": {
            "shape": [300, 300, 3],
            "normalization": "scale_0_1",
        },
        "output": {
            "classes": [
                "bone_cells_group",
                "bone_cells_individual",
                "other",
                "rbc_group",
                "rbc_individual",
                "vascular_fragments",
            ],
        },
        "postprocessing": {
            "top_k": 1,
        },
    }


@pytest.fixture
def mock_metadata_with_top_k(mock_metadata) -> dict[str, Any]:
    """Metadata with top_k=3."""
    metadata = mock_metadata.copy()
    metadata["postprocessing"]["top_k"] = 3
    return metadata


@pytest.fixture
def sample_image_bgr() -> np.ndarray:
    """100x100x3 BGR image with gradient."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 100).astype(np.uint8)
    image[:, :, 1] = np.linspace(100, 200, 100).astype(np.uint8)
    image[:, :, 2] = np.linspace(200, 50, 100).astype(np.uint8)
    return image


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """100x100x3 RGB image."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 128
    image[:, :, 1] = 64
    image[:, :, 2] = 192
    return image


@pytest.fixture
def sample_image_large() -> np.ndarray:
    """512x512x3 image for downscale testing."""
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_small() -> np.ndarray:
    """50x50x3 image for upscale testing."""
    return np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)


@pytest.fixture
def mock_keras_model():
    """Mock Keras model with predict method."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]])
    return model


@pytest.fixture
def mock_load_classifier_model(monkeypatch):
    """Mock loader to avoid loading real ML model."""
    from bioscopeai_classifier_service.model_processing import loader

    original_load = loader.load_classifier_model
    loader.load_classifier_model.cache_clear()

    model = MagicMock()
    model.predict.return_value = np.array([[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]])

    def _mock_load():
        return model

    _mock_load.cache_clear = lambda: None
    _mock_load.cache_info = lambda: type("CacheInfo", (), {"currsize": 0})()

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.load_classifier_model",
        _mock_load,
    )
    yield _mock_load

    loader.load_classifier_model = original_load


@pytest.fixture
def mock_load_model(monkeypatch):
    """Alias for mock_load_classifier_model."""
    from bioscopeai_classifier_service.model_processing import loader

    original_load = loader.load_classifier_model
    loader.load_classifier_model.cache_clear()

    model = MagicMock()
    model.predict.return_value = np.array([[0.8, 0.2]])

    def _mock_load():
        return model

    _mock_load.cache_clear = lambda: None
    _mock_load.cache_info = lambda: type("CacheInfo", (), {"currsize": 0})()

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.load_classifier_model",
        _mock_load,
    )
    yield _mock_load

    loader.load_classifier_model = original_load


@pytest.fixture
def mock_load_keras_model(monkeypatch, mock_metadata):
    """Mock keras.models.load_model, Path.exists and load_metadata."""

    model = MagicMock()
    model.predict.return_value = np.array([[0.8, 0.2]])

    def _mock_load(path):
        return model

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.load_model", _mock_load
    )

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.Path.exists",
        lambda self: True,
    )

    def _mock_metadata():
        return mock_metadata

    _mock_metadata.cache_clear = lambda: None
    _mock_metadata.cache_info = lambda: type("CacheInfo", (), {"currsize": 0})()

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.load_metadata",
        _mock_metadata,
    )

    return _mock_load


@pytest.fixture
def mock_verify_checksum(monkeypatch):
    """Mock checksum verification."""
    mock_verify = MagicMock(return_value=True)
    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.verify_checksum",
        mock_verify,
    )
    return mock_verify


@pytest.fixture
def mock_preprocess_image(monkeypatch):
    """Mock preprocess_image function."""
    mock_preprocess = MagicMock(
        return_value=np.random.rand(1, 224, 224, 3).astype(np.float32)
    )
    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.model_processing.preprocess_image",
        mock_preprocess,
    )
    return mock_preprocess


@pytest.fixture
def mock_run_inference(monkeypatch):
    """Mock run_inference function."""
    mock_inference = MagicMock(
        return_value=[
            {"label": "bone_cells_group", "confidence": 40.0},
            {"label": "bone_cells_individual", "confidence": 25.0},
            {"label": "other", "confidence": 15.0},
            {"label": "rbc_group", "confidence": 10.0},
            {"label": "rbc_individual", "confidence": 7.0},
            {"label": "vascular_fragments", "confidence": 3.0},
        ]
    )
    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.model_processing.run_inference",
        mock_inference,
    )
    return mock_inference


@pytest.fixture
def mock_model_processing_service(
    mock_load_classifier_model, mock_load_metadata, monkeypatch
):
    """Mock ModelProcessingService instance."""
    from bioscopeai_classifier_service.model_processing.model_processing import (
        ModelProcessingService,
    )

    ModelProcessingService._instance = None

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.model_processing.load_classifier_model",
        mock_load_classifier_model,
    )
    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.model_processing.load_metadata",
        mock_load_metadata,
    )

    return ModelProcessingService()


@pytest.fixture
def mock_load_metadata(monkeypatch, mock_metadata):
    """Mock loader to return test metadata."""
    from bioscopeai_classifier_service.model_processing import loader

    original_load = loader.load_metadata
    loader.load_metadata.cache_clear()

    def _mock_metadata():
        return mock_metadata

    _mock_metadata.cache_clear = lambda: None
    _mock_metadata.cache_info = lambda: type("CacheInfo", (), {"currsize": 0})()

    monkeypatch.setattr(
        "bioscopeai_classifier_service.model_processing.loader.load_metadata",
        _mock_metadata,
    )
    yield _mock_metadata

    loader.load_metadata = original_load


@pytest.fixture
def class_names() -> list[str]:
    """Classification labels from test-config.yaml."""
    return [
        "bone_cells_group",
        "bone_cells_individual",
        "other",
        "rbc_group",
        "rbc_individual",
        "vascular_fragments",
    ]


@pytest.fixture
def class_names_multiclass() -> list[str]:
    """Multi-class labels - same as binary for this model."""
    return [
        "bone_cells_group",
        "bone_cells_individual",
        "other",
        "rbc_group",
        "rbc_individual",
        "vascular_fragments",
    ]


@pytest.fixture
def mock_predictions_binary() -> np.ndarray:
    """Mock model predictions for 6-class classification."""
    return np.array([[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]])


@pytest.fixture
def mock_predictions_multiclass() -> np.ndarray:
    """Mock model predictions for 6-class classification."""
    return np.array([[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]])


@pytest.fixture
def temp_model_file(tmp_path) -> Path:
    """Temporary model file."""
    model_file = tmp_path / "test_model.keras"
    model_file.write_bytes(b"fake model content")
    return model_file


@pytest.fixture
def temp_metadata_file(tmp_path, mock_metadata) -> Path:
    """Temporary metadata.json."""
    import json

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(mock_metadata), encoding="utf-8")
    return metadata_file


@pytest.fixture
def temp_checksum_file(tmp_path) -> Path:
    """Temporary checksum.sha256."""
    checksum_file = tmp_path / "checksum.sha256"
    checksum_file.write_text(
        "5d0da60d5d0da60d5d0da60d5d0da60d5d0da60d5d0da60d5d0da60d5d0da60d  test_model.keras\n",
        encoding="utf-8",
    )
    return checksum_file
