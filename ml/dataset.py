"""Load extracted landmarks as model-ready arrays.

Shared by train.py, evaluate.py and build_templates.py so all three see the
same rows, the same classes and the same features. Features come from
``app.domain.gesture.features`` and nowhere else (CLAUDE.md 6.1): this module
only reads the parquet and calls it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.domain.gesture.features import (
    FEATURE_LENGTH,
    FEATURE_VERSION,
    FINGERS,
    JOINTS,
    Handedness,
    aspect_of,
    extract_features,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "ml" / "data" / "bukva_landmarks.parquet"
LABELS_FILE = ROOT / "ml" / "labels.json"
ARTIFACTS = ROOT / "ml" / "artifacts"

COORD_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in "xyz"]


@dataclass(frozen=True)
class Split:
    """One side of the official, signer-disjoint split."""

    features: npt.NDArray[np.float64]
    labels: npt.NDArray[np.str_]
    signers: npt.NDArray[np.str_]
    videos: npt.NDArray[np.str_]
    angles: list[dict[str, dict[str, float]]]

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class Dataset:
    train: Split
    test: Split
    classes: list[str]
    sha256: str
    feature_version: int = FEATURE_VERSION


def model_versions() -> list[int]:
    """Versions with a model-vN.json in ml/artifacts, ascending."""
    versions = []
    for path in ARTIFACTS.glob("model-v*.json"):
        suffix = path.stem.removeprefix("model-v")
        if suffix.isdigit():
            versions.append(int(suffix))
    return sorted(versions)


def load_classes() -> list[str]:
    payload: dict[str, Any] = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return list(payload["classes"])


def load(parquet: Path = DEFAULT_PARQUET, classes: list[str] | None = None) -> Dataset:
    """Read the parquet, keep the MVP classes, and compute features.

    Rows outside ``classes`` (by default ml/labels.json) are dropped here, once,
    so no caller can accidentally train on a class the app will never offer.
    Experiments that need a different set pass it explicitly.
    """
    if not parquet.is_file():
        raise SystemExit(f"{parquet} not found. Run `make extract` first.")

    classes = classes if classes is not None else load_classes()
    frame = pd.read_parquet(parquet)
    frame = frame[frame["letter"].isin(classes)].reset_index(drop=True)

    return Dataset(
        train=_split(frame[frame["train"]]),
        test=_split(frame[~frame["train"]]),
        classes=classes,
        sha256=_sha256(parquet),
    )


def _split(frame: pd.DataFrame) -> Split:
    coords = frame[COORD_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 21, 3)
    vectors: list[npt.NDArray[np.float64]] = []
    angles: list[dict[str, dict[str, float]]] = []

    for landmarks, handedness, width, height in zip(
        coords,
        frame["handedness"],
        frame["width"],
        frame["height"],
        strict=True,
    ):
        features = extract_features(
            landmarks,
            Handedness(handedness),
            aspect=aspect_of(int(width), int(height)),
            # Raw clips, unflipped, with the physical hand already stored --
            # the same convention the browser follows (CLAUDE.md 7).
            mirrored=False,
        )
        vectors.append(features.vector)
        angles.append(features.angles)

    return Split(
        features=np.vstack(vectors) if vectors else np.empty((0, FEATURE_LENGTH)),
        labels=frame["letter"].to_numpy(dtype=str),
        signers=frame["signer_id"].to_numpy(dtype=str),
        videos=frame["video_id"].to_numpy(dtype=str),
        angles=angles,
    )


def angle_matrix(split: Split) -> npt.NDArray[np.float64]:
    """Joint angles as an (n, 5, 3) array, finger-major, for template work."""
    return np.array(
        [
            [[sample[finger][joint] for joint in JOINTS] for finger in FINGERS]
            for sample in split.angles
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
