from operator import itemgetter
from typing import Any

import numpy as np


def run_inference(
    model: Any,
    input_tensor: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Execute ML inference on preprocessed input."""
    preds = model.predict(input_tensor, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    label = class_names[pred_idx]

    all_predictions = [
        {
            "label": class_names[i],
            "confidence": float(preds[i]),
        }
        for i in range(len(class_names))
    ]
    all_predictions.sort(key=itemgetter("confidence"), reverse=True)

    return {
        "label": label,
        "confidence": confidence,
        "all_predictions": all_predictions,
    }
