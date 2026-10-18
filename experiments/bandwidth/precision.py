"""Does rounding landmarks before sending cost accuracy?

Rounding each coordinate to 4 decimals shrinks a frame message from about 1.4
kB to 0.6 kB (results/summary.md). A coordinate is a fraction of the frame,
so 4 decimals is 1/10,000 of the width -- 0.13 px at 1280 -- well below
MediaPipe's own jitter. This checks that claim against the model instead of
assuming it: model-v2 on the held-out test signers, with landmarks at full
precision and rounded to 2, 3 and 4 decimals. A frame whose rounding makes
two joints coincide is rejected by features.py (INVALID_FRAME) and counted as
a miss, since the learner would get nothing for it.

Usage (from the repository root):
    uv run --project backend --group ml python experiments/bandwidth/precision.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import dataset  # noqa: E402

from app.domain.gesture.features import (  # noqa: E402
    FEATURE_LENGTH,
    Handedness,
    InvalidLandmarksError,
    aspect_of,
    extract_features,
)

RESULTS = Path(__file__).with_name("results")
DECIMALS = (None, 4, 3, 2)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model = joblib.load(ROOT / "ml" / "artifacts" / "model-v2.joblib")
    frame = pd.read_parquet(dataset.DEFAULT_PARQUET)
    test = frame[~frame["train"] & frame["letter"].isin(dataset.load_classes())]
    coords = test[dataset.COORD_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 21, 3)
    baseline = None
    rows = []

    for decimals in DECIMALS:
        # Math.round semantics (floor(x + 0.5)), as the browser rounds.
        points = (
            coords if decimals is None else np.floor(coords * 10**decimals + 0.5) / 10**decimals
        )
        features = np.zeros((len(test), FEATURE_LENGTH))
        valid = np.ones(len(test), dtype=bool)
        for i, (landmarks, hand, width, height) in enumerate(
            zip(points, test["handedness"], test["width"], test["height"], strict=True)
        ):
            try:
                features[i] = extract_features(
                    landmarks,
                    Handedness(hand),
                    aspect=aspect_of(int(width), int(height)),
                    mirrored=False,
                ).vector
            except InvalidLandmarksError:
                # Rounding collapsed two joints onto one point: the server
                # would answer INVALID_FRAME, so this frame is simply lost.
                valid[i] = False
        predicted = np.where(valid, model.predict(features), "<invalid>")
        if baseline is None:
            baseline = predicted
        rows.append(
            {
                "decimals": "full" if decimals is None else decimals,
                "frames": len(test),
                "accuracy": round(float(accuracy_score(test["letter"], predicted)), 4),
                "macro_f1": round(float(f1_score(test["letter"], predicted, average="macro")), 4),
                "predictions_changed_vs_full": int((predicted != baseline).sum()),
                "frames_made_invalid": int((~valid).sum()),
            }
        )

    result = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    result.to_csv(RESULTS / "precision.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
