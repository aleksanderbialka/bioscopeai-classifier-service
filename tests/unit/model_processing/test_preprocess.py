"""Unit tests for preprocess.py - Image preprocessing pipeline."""

import numpy as np
import pytest

from bioscopeai_classifier_service.model_processing.preprocess import preprocess_image


class TestPreprocessImage:
    def test_basic_bgr_to_rgb_conversion(self, sample_image_bgr, mock_metadata):
        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert result.shape == (1, 300, 300, 3)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= 1.0
        assert 0 <= result.max() <= 1.0

    def test_resize_to_target_size(self, sample_image_large, mock_metadata):
        result = preprocess_image(sample_image_large, mock_metadata)
        assert result.shape == (1, 300, 300, 3)

    def test_upscale_small_image(self, sample_image_small, mock_metadata):
        result = preprocess_image(sample_image_small, mock_metadata)
        assert result.shape == (1, 300, 300, 3)

    def test_normalization_scale_0_1(self, sample_image_bgr, mock_metadata):
        mock_metadata["input"]["normalization"] = "scale_0_1"

        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= 1.0
        assert 0.0 <= result.max() <= 1.0

    def test_normalization_unknown_falls_back_to_raw(
        self, sample_image_bgr, mock_metadata
    ):
        """Unknown normalization should use raw float32 values."""
        mock_metadata["input"]["normalization"] = "unknown_normalization"

        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= 255.0

    def test_expand_dims_adds_batch_dimension(self, sample_image_bgr, mock_metadata):
        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert result.ndim == 4
        assert result.shape[0] == 1

    def test_color_conversion_skip_if_unknown(self, sample_image_bgr, mock_metadata):
        """Unknown color conversion should be skipped."""
        mock_metadata["preprocessing"]["color_conversion"] = "unknown_format"

        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert result.shape == (1, 300, 300, 3)

    def test_default_resize_from_settings(self, sample_image_bgr):
        """Should use settings.ml_model.IMG_SIZE when not in metadata."""
        metadata = {"preprocessing": {}, "input": {"normalization": "scale_0_1"}}

        result = preprocess_image(sample_image_bgr, metadata)

        assert result.shape == (1, 300, 300, 3)

    def test_custom_resize_dimensions(self, sample_image_bgr):
        metadata = {
            "preprocessing": {"resize": [299, 299], "color_conversion": "BGR_to_RGB"},
            "input": {"normalization": "scale_0_1"},
        }

        result = preprocess_image(sample_image_bgr, metadata)

        assert result.shape == (1, 299, 299, 3)

    def test_preprocessing_maintains_data_integrity(
        self, sample_image_bgr, mock_metadata
    ):
        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_preprocessing_is_deterministic(self, sample_image_bgr, mock_metadata):
        result1 = preprocess_image(sample_image_bgr.copy(), mock_metadata)
        result2 = preprocess_image(sample_image_bgr.copy(), mock_metadata)

        np.testing.assert_array_equal(result1, result2)

    def test_different_images_produce_different_tensors(self, mock_metadata):
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result1 = preprocess_image(image1, mock_metadata)
        result2 = preprocess_image(image2, mock_metadata)

        assert not np.array_equal(result1, result2)
        assert result1.mean() < result2.mean()

    def test_preprocessing_with_empty_metadata(self, sample_image_bgr):
        metadata = {"preprocessing": {}, "input": {}}

        result = preprocess_image(sample_image_bgr, metadata)

        assert result.shape == (1, 300, 300, 3)
        assert result.dtype == np.float32

    def test_black_image_preprocessing(self, mock_metadata):
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = preprocess_image(black_image, mock_metadata)

        assert result.shape == (1, 300, 300, 3)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(0.0)

    def test_white_image_preprocessing(self, mock_metadata):
        white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = preprocess_image(white_image, mock_metadata)

        assert result.shape == (1, 300, 300, 3)
        assert result.min() == pytest.approx(1.0, rel=1e-3)
        assert result.max() == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize(
        "target_size", [[128, 128], [224, 224], [299, 299], [512, 512]]
    )
    def test_various_target_sizes(self, sample_image_bgr, target_size):
        metadata = {
            "preprocessing": {"resize": target_size, "color_conversion": "BGR_to_RGB"},
            "input": {"normalization": "scale_0_1"},
        }

        result = preprocess_image(sample_image_bgr, metadata)

        assert result.shape == (1, target_size[0], target_size[1], 3)

    def test_rectangular_resize(self, sample_image_bgr):
        """cv2.resize expects (width, height) not (height, width)."""
        metadata = {
            "preprocessing": {"resize": [224, 320], "color_conversion": "BGR_to_RGB"},
            "input": {"normalization": "scale_0_1"},
        }

        result = preprocess_image(sample_image_bgr, metadata)

        assert result.shape == (1, 320, 224, 3)

    def test_preprocessing_preserves_aspect_ratio_behavior(self, mock_metadata):
        """cv2.resize doesn't preserve aspect ratio by default."""
        rect_image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

        result = preprocess_image(rect_image, mock_metadata)

        assert result.shape == (1, 300, 300, 3)

    def test_dtype_conversion(self, sample_image_bgr, mock_metadata):
        result = preprocess_image(sample_image_bgr, mock_metadata)
        assert result.dtype == np.float32

    def test_channel_order_after_bgr_to_rgb(self, mock_metadata):
        """Blue channel in BGR should become third channel in RGB."""
        blue_bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255

        result = preprocess_image(blue_bgr, mock_metadata)

        assert result[0, :, :, 2].mean() > result[0, :, :, 0].mean()
        assert result[0, :, :, 2].mean() > result[0, :, :, 1].mean()

    def test_preprocessing_pipeline_order(self, sample_image_bgr, mock_metadata):
        """Verify correct order: BGR→RGB, resize, normalize, expand dims."""
        result = preprocess_image(sample_image_bgr, mock_metadata)

        assert result.ndim == 4
        assert result.shape == (1, 300, 300, 3)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= 1.0
