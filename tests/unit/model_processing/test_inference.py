"""Unit tests for inference.py - ML model prediction execution."""

import numpy as np
import pytest

from bioscopeai_classifier_service.model_processing.inference import run_inference


class TestRunInference:
    def test_basic_binary_classification(self, mock_keras_model, class_names):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert result["label"] == "bone_cells_group"
        assert result["confidence"] == pytest.approx(0.4, rel=1e-5)
        assert len(result["all_predictions"]) == 6
        assert result["all_predictions"][0]["label"] == "bone_cells_group"
        mock_keras_model.predict.assert_called_once_with(input_tensor, verbose=0)

    def test_inference_with_high_confidence(self, mock_keras_model, class_names):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.99, 0.003, 0.002, 0.002, 0.002, 0.001]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert result["label"] == "bone_cells_group"
        assert result["confidence"] == pytest.approx(0.99)

    def test_inference_with_low_confidence(self, mock_keras_model, class_names):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.51, 0.15, 0.13, 0.11, 0.07, 0.03]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert result["label"] == "bone_cells_group"
        assert result["confidence"] == pytest.approx(0.51)

    def test_inference_second_class_prediction(self, mock_keras_model, class_names):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.1, 0.7, 0.08, 0.06, 0.04, 0.02]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert result["label"] == "bone_cells_individual"
        assert result["confidence"] == pytest.approx(0.7)
        assert result["all_predictions"][0]["label"] == "bone_cells_individual"

    def test_multiclass_classification(self, mock_keras_model, class_names_multiclass):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        predictions = np.array([[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]])
        mock_keras_model.predict.return_value = predictions

        result = run_inference(mock_keras_model, input_tensor, class_names_multiclass)

        assert result["label"] == "bone_cells_group"
        assert result["confidence"] == pytest.approx(0.5)
        assert len(result["all_predictions"]) == 6

    def test_all_predictions_sorted_descending(
        self, mock_keras_model, class_names_multiclass
    ):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.05, 0.5, 0.15, 0.1, 0.08, 0.12]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names_multiclass)

        all_preds = result["all_predictions"]
        for i in range(len(all_preds) - 1):
            assert all_preds[i]["confidence"] >= all_preds[i + 1]["confidence"]

        assert all_preds[0]["label"] == "bone_cells_individual"
        assert all_preds[0]["confidence"] == pytest.approx(0.5)

    def test_inference_result_structure(self, mock_keras_model, class_names):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert all(k in result for k in ["label", "confidence", "all_predictions"])
        assert isinstance(result["label"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["all_predictions"], list)

    def test_confidence_sum_equals_one(self, mock_keras_model, class_names):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        total = sum(pred["confidence"] for pred in result["all_predictions"])
        assert total == pytest.approx(1.0, rel=1e-5)

    def test_inference_with_numpy_array_types(self, mock_keras_model, class_names):
        """Verify confidence values are Python floats, not numpy types."""
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert type(result["confidence"]) is float
        for pred in result["all_predictions"]:
            assert type(pred["confidence"]) is float

    def test_inference_with_equal_probabilities(self, mock_keras_model, class_names):
        """Edge case: equal probabilities should pick first class."""
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array(
            [[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]]
        )

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert result["label"] == "bone_cells_group"
        assert result["confidence"] == pytest.approx(0.5)

    def test_inference_with_single_class(self, mock_keras_model):
        """Edge case: single class classification."""
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = np.array([[1.0]])

        result = run_inference(mock_keras_model, input_tensor, ["only_class"])

        assert result["label"] == "only_class"
        assert result["confidence"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "predictions,expected_label,expected_confidence",
        [
            (np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]]), "bone_cells_group", 0.9),
            (
                np.array([[0.02, 0.9, 0.02, 0.02, 0.02, 0.02]]),
                "bone_cells_individual",
                0.9,
            ),
            (np.array([[0.7, 0.1, 0.08, 0.06, 0.04, 0.02]]), "bone_cells_group", 0.7),
            (np.array([[0.05, 0.05, 0.05, 0.75, 0.05, 0.05]]), "rbc_group", 0.75),
        ],
    )
    def test_various_prediction_scenarios(
        self,
        mock_keras_model,
        class_names,
        predictions,
        expected_label,
        expected_confidence,
    ):
        input_tensor = np.random.rand(1, 224, 224, 3).astype("float32")
        mock_keras_model.predict.return_value = predictions

        result = run_inference(mock_keras_model, input_tensor, class_names)

        assert result["label"] == expected_label
        assert result["confidence"] == pytest.approx(expected_confidence)
