"""Unit tests for loader.py - Model and metadata loading utilities."""

import pytest

from bioscopeai_classifier_service.model_processing import loader


class TestLoadMetadata:
    def test_load_returns_cached_metadata(self, mock_load_metadata):
        result = loader.load_metadata()

        assert isinstance(result, dict)
        assert "model_name" in result

    def test_metadata_contains_expected_fields(self, mock_load_metadata):
        result = loader.load_metadata()

        assert "model_name" in result
        assert "architecture" in result
        assert "input" in result

    def test_metadata_caching_works(self):
        """lru_cache should return same instance."""

        result1 = loader.load_metadata()
        result2 = loader.load_metadata()

        assert result1 is result2

    def test_clear_cache(self, mock_load_metadata):
        """Can clear lru_cache."""

        loader.load_metadata()
        loader.load_metadata.cache_clear()

        assert loader.load_metadata.cache_info().currsize == 0


class TestLoadClassifierModel:
    def test_load_returns_model(self, mock_load_keras_model):
        loader.load_classifier_model.cache_clear()
        result = loader.load_classifier_model()

        assert result is not None
        assert hasattr(result, "predict")

    def test_model_caching_works(self, mock_load_keras_model):
        """lru_cache should return same model instance."""

        loader.load_classifier_model.cache_clear()
        result1 = loader.load_classifier_model()
        result2 = loader.load_classifier_model()

        assert result1 is result2

    def test_clear_model_cache(self, mock_load_keras_model):
        """Can clear lru_cache."""

        loader.load_classifier_model()
        loader.load_classifier_model.cache_clear()

        assert loader.load_classifier_model.cache_info().currsize == 0

    def test_model_predict_method_exists(self, mock_load_keras_model):
        loader.load_classifier_model.cache_clear()
        model = loader.load_classifier_model()

        assert callable(model.predict)


class TestVerifyChecksum:
    def test_verify_valid_sha256_checksum(self, tmp_path):
        model_file = tmp_path / "model.keras"
        model_file.write_bytes(b"test_model_data")
        expected_checksum = (
            "ac91ddbc5b33997e522a0a57a5b460355700b6e1c67bdcd17f70ac2acae24ab8"
        )

        result = loader.verify_checksum(model_file, expected_checksum)

        assert result is True

    def test_verify_invalid_checksum(self, tmp_path):
        model_file = tmp_path / "model.keras"
        model_file.write_bytes(b"test_model_data")
        invalid_checksum = "invalid_checksum_value"

        result = loader.verify_checksum(model_file, invalid_checksum)

        assert result is False

    def test_missing_model_file_raises_error(self, tmp_path):
        non_existent_file = tmp_path / "nonexistent.keras"

        with pytest.raises(FileNotFoundError):
            loader.verify_checksum(non_existent_file, "some_checksum")

    def test_checksum_with_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.keras"
        empty_file.write_bytes(b"")
        expected_checksum = (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

        result = loader.verify_checksum(empty_file, expected_checksum)

        assert result is True

    def test_checksum_case_sensitivity(self, tmp_path):
        model_file = tmp_path / "model.keras"
        model_file.write_bytes(b"test_data")
        correct_hash = (
            "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
        )

        result = loader.verify_checksum(model_file, correct_hash.upper())

        assert result is False
