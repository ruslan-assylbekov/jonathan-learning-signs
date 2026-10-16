"""The trained classifier behind the GestureClassifier port.

Loads ``model-vN.joblib`` together with its ``model-vN.json`` metadata. The
metadata is what lets the app refuse to start on a feature mismatch
(CLAUDE.md 6.3): the pickled pipeline itself cannot say which normalization
it was trained on.

joblib files are pickles and execute code when loaded. Only load artifacts
this project produced; never one supplied by a user or fetched at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.domain.entities import Prediction
from app.domain.gesture.features import Features


class ModelMetadataError(RuntimeError):
    pass


class JoblibClassifier:
    """Implements ``GestureClassifier`` with a fitted scikit-learn pipeline."""

    def __init__(self, model_path: Path) -> None:
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise ModelMetadataError(
                f"{metadata_path} is missing. A model without metadata cannot be checked "
                "against FEATURE_VERSION, so it is not loaded."
            )

        self._metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._pipeline: Any = joblib.load(model_path)
        self._classes: tuple[str, ...] = tuple(str(label) for label in self._pipeline.classes_)

        declared = tuple(self._metadata.get("classes", ()))
        if declared != self._classes:
            raise ModelMetadataError(
                f"{model_path.name} predicts {self._classes} but its metadata declares "
                f"{declared}; the two files are from different training runs."
            )

        # Frames arrive one at a time per connection; a thread pool per call
        # only adds setup cost (see ml/train.py, set_single_threaded).
        estimator = self._pipeline.steps[-1][1]
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1

    @property
    def feature_version(self) -> int:
        return int(self._metadata["feature_version"])

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def predict(self, features: Features) -> Prediction:
        probabilities = self._pipeline.predict_proba(features.vector.reshape(1, -1))[0]
        best = int(np.argmax(probabilities))
        return Prediction(label=self._classes[best], confidence=float(probabilities[best]))
