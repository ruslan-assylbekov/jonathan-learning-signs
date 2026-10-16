"""Is no_sign over-predicted because of class weighting, or by nature?

Outcome: no_sign was removed from the model (CLAUDE.md 6.4).

Prompted by model-v1's held-out report, where no_sign scored F1 0.06 and real
letters lost up to a quarter of their frames to it. Answered here on the
TRAINING signers only, with the same signer-grouped folds as train.py: the
test signers have been seen once already and must not steer the fix.

Usage (from the repository root):
    uv run --project backend --group ml python experiments/models/no_sign_weighting.py

Writes experiments/models/no_sign_weighting.csv.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import dataset  # noqa: E402

NO_SIGN = "no_sign"
OUT = Path(__file__).with_name("no_sign_weighting.csv")


def variants() -> dict[str, dict[str, object]]:
    return {
        "balanced": {"class_weight": "balanced"},
        "unweighted": {"class_weight": None},
        "no_sign_halved": {"class_weight": "no_sign_halved"},
    }


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Explicitly include no_sign: ml/labels.json dropped it after this
    # experiment, and the default class list would quietly leave it out.
    train = dataset.load(classes=[*dataset.load_classes(), NO_SIGN]).train
    rows = []

    for name, options in variants().items():
        for fold, (fit_index, holdout_index) in enumerate(
            GroupKFold(n_splits=5).split(train.features, train.labels, groups=train.signers)
        ):
            weights = options["class_weight"]
            if weights == "no_sign_halved":
                classes, counts = np.unique(train.labels[fit_index], return_counts=True)
                balanced = len(fit_index) / (len(classes) * counts)
                weights = dict(zip(classes, balanced, strict=True))
                weights[NO_SIGN] *= 0.5

            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000, class_weight=weights)
            ).fit(train.features[fit_index], train.labels[fit_index])

            truth = train.labels[holdout_index]
            predicted = model.predict(train.features[holdout_index])
            letters = truth != NO_SIGN
            rows.append(
                {
                    "variant": name,
                    "fold": fold,
                    "macro_f1_all": f1_score(truth, predicted, average="macro"),
                    "macro_f1_letters": f1_score(
                        truth[letters],
                        predicted[letters],
                        labels=sorted(set(truth[letters])),
                        average="macro",
                    ),
                    "no_sign_f1": f1_score(truth == NO_SIGN, predicted == NO_SIGN, zero_division=0),
                    "letter_frames_lost_to_no_sign": float((predicted[letters] == NO_SIGN).mean()),
                    "share_predicted_no_sign": float((predicted == NO_SIGN).mean()),
                    "share_truly_no_sign": float((truth == NO_SIGN).mean()),
                }
            )
            print(f"{name} fold {fold} done")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    print()
    print(
        frame.groupby("variant")
        .mean(numeric_only=True)
        .drop(columns="fold")
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
