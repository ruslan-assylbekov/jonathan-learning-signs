"""Loading a trained model behind the classifier port."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.domain.gesture.features import FEATURE_VERSION, Handedness, extract_features
from app.infrastructure.ml.model import JoblibClassifier, ModelMetadataError
from app.main import build_app
from tests import synthetic

OPEN = synthetic.hand()
FIST = synthetic.hand(
    dict.fromkeys(("thumb", "index", "middle", "ring", "pinky"), (80.0, 80.0, 70.0))
)


def features(landmarks: list[list[float]]) -> np.ndarray:
    return extract_features(landmarks, Handedness.RIGHT, aspect=1.0).vector


def write_model(
    directory: Path,
    *,
    feature_version: int = FEATURE_VERSION,
    declared_classes: list[str] | None = None,
    with_metadata: bool = True,
) -> Path:
    """A tiny real pipeline, trained on two synthetic poses."""
    rng = np.random.default_rng(0)
    samples = [features(OPEN) + rng.normal(0, 0.01, 78) for _ in range(20)]
    samples += [features(FIST) + rng.normal(0, 0.01, 78) for _ in range(20)]
    labels = ["open"] * 20 + ["fist"] * 20
    pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(samples, labels)

    path = directory / "model-v9.joblib"
    joblib.dump(pipeline, path)
    if with_metadata:
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": 9,
                    "feature_version": feature_version,
                    "classes": declared_classes or [str(c) for c in pipeline.classes_],
                    "model_type": "logreg",
                }
            ),
            encoding="utf-8",
        )
    return path


def test_predicts_with_the_trained_pipeline(tmp_path: Path) -> None:
    classifier = JoblibClassifier(write_model(tmp_path))

    open_hand = classifier.predict(extract_features(OPEN, Handedness.RIGHT, aspect=1.0))
    fist = classifier.predict(extract_features(FIST, Handedness.RIGHT, aspect=1.0))

    assert open_hand.label == "open"
    assert fist.label == "fist"
    assert 0.5 < open_hand.confidence <= 1.0


def test_reports_the_feature_version_it_was_trained_on(tmp_path: Path) -> None:
    classifier = JoblibClassifier(write_model(tmp_path, feature_version=FEATURE_VERSION))

    assert classifier.feature_version == FEATURE_VERSION
    assert set(classifier.classes) == {"open", "fist"}


def test_refuses_a_model_without_metadata(tmp_path: Path) -> None:
    """Without metadata nobody can check what normalization it expects."""
    with pytest.raises(ModelMetadataError, match="missing"):
        JoblibClassifier(write_model(tmp_path, with_metadata=False))


def test_refuses_metadata_from_a_different_run(tmp_path: Path) -> None:
    with pytest.raises(ModelMetadataError, match="different training runs"):
        JoblibClassifier(write_model(tmp_path, declared_classes=["А", "Б"]))


def test_the_app_refuses_to_start_on_a_feature_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md 6.3: a mismatch shows up as poor accuracy, never as an error,
    unless startup makes it one."""
    path = write_model(tmp_path, feature_version=FEATURE_VERSION - 1)
    monkeypatch.setattr(settings, "model_path", str(path))

    with pytest.raises(RuntimeError, match="feature version"):
        build_app()


def test_the_app_switches_to_a_real_model_when_one_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "model_path", str(write_model(tmp_path)))

    body = TestClient(build_app()).get("/health").json()

    assert body["stub"] is False
    assert body["classifier"] == "JoblibClassifier"
