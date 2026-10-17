"""Select and train the gesture classifier (CLAUDE.md 6.4, 8.4).

Model selection happens on the training signers only, with GroupKFold over
signer ids: frames from one person never sit on both sides of a fold, which a
random frame-level split would allow and which would flatter every number.
The official test signers are touched once, at the end, to report the chosen
model -- never to choose it.

Usage:
    uv run --group ml python ml/train.py [--version N]   (default: next unused)

Writes:
    ml/artifacts/model-vN.joblib    the fitted sklearn pipeline
    ml/artifacts/model-vN.json      metadata the backend checks at startup
    experiments/models/cv_folds-vN.csv  every candidate, every fold
    experiments/models/summary-vN.json  candidates ranked, chosen model, test score
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import dataset
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS = dataset.ROOT / "ml" / "artifacts"
EXPERIMENTS = dataset.ROOT / "experiments" / "models"
FOLDS = 5
SEED = 0
LATENCY_SAMPLES = 500


def candidates() -> dict[str, Callable[[], Pipeline]]:
    """Everything worth comparing, as factories so each fold starts clean.

    Letters do not have equal numbers of frames (101-125 clips each, minus
    frames with no hand detected), so every candidate that can weight classes
    does. kNN cannot, and is kept as the honest unweighted baseline.
    """
    return {
        "knn": lambda: make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")
        ),
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "random_forest": lambda: make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            )
        ),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                early_stopping=True,
                max_iter=400,
                random_state=SEED,
            ),
        ),
    }


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=int,
        help="artifact version to write (default: one above the highest existing)",
    )
    parser.add_argument("--only", help="train just this candidate, skipping selection")
    args = parser.parse_args()
    existing = dataset.model_versions()
    if args.version is None:
        args.version = (existing[-1] if existing else 0) + 1
    elif args.version in existing:
        # Overwriting a model the app or the report already depends on is how
        # numbers quietly stop matching their artifact.
        raise SystemExit(
            f"model-v{args.version} already exists. Pick a new --version, or "
            "delete the old artifact deliberately first."
        )
    print(f"training model-v{args.version}")

    print("loading and extracting features ...")
    data = dataset.load()
    print(
        f"  train {len(data.train)} frames / {len(set(data.train.signers))} signers, "
        f"test {len(data.test)} frames / {len(set(data.test.signers))} signers, "
        f"{len(data.classes)} classes"
    )

    pool = candidates()
    if args.only:
        pool = {args.only: pool[args.only]}

    folds = cross_validate(pool, data.train)
    ranking = rank(folds)
    chosen = ranking[0]["model"]
    print(f"\nchosen: {chosen} (highest mean macro F1 across signer folds)")

    model = fit(pool[chosen](), data.train)
    test = score(model, data.test)
    latency = measure_latency(model, data.test.features)
    print(
        f"held-out test signers: accuracy {test['accuracy']:.4f}, macro F1 {test['macro_f1']:.4f}"
    )
    print(
        f"inference: median {latency['median_ms']:.3f} ms, p95 {latency['p95_ms']:.3f} ms per frame"
    )

    save(model, chosen, data, folds, ranking, test, latency, args.version)


def cross_validate(pool: dict[str, Callable[[], Pipeline]], train: dataset.Split) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=FOLDS)
    rows: list[dict[str, Any]] = []

    for name, factory in pool.items():
        print(f"\n{name}")
        for fold, (fit_index, holdout_index) in enumerate(
            splitter.split(train.features, train.labels, groups=train.signers)
        ):
            # A signer appearing on both sides would make this a memorization
            # test. GroupKFold guarantees it does not; assert it anyway, since
            # the whole evaluation rests on it.
            assert not set(train.signers[fit_index]) & set(train.signers[holdout_index])

            started = time.perf_counter()
            model = fit(factory(), subset(train, fit_index))
            seconds = time.perf_counter() - started
            result = score(model, subset(train, holdout_index))
            rows.append({"model": name, "fold": fold, "fit_seconds": round(seconds, 2), **result})
            print(
                f"  fold {fold}: accuracy {result['accuracy']:.4f}  "
                f"macro F1 {result['macro_f1']:.4f}  ({seconds:.1f}s)"
            )
    return pd.DataFrame(rows)


def rank(folds: pd.DataFrame) -> list[dict[str, Any]]:
    summary = (
        folds.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index()
    )
    print("\nsigner-grouped cross-validation, ranked by macro F1:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return [
        {
            str(key): (round(value, 4) if isinstance(value, float) else value)
            for key, value in row.items()
        }
        for row in summary.to_dict(orient="records")
    ]


def fit(model: Pipeline, split: dataset.Split) -> Pipeline:
    """Fit, passing balanced sample weights to the steps that take them."""
    estimator = model.steps[-1][1]
    if isinstance(estimator, MLPClassifier):
        step = model.steps[-1][0]
        model.fit(
            split.features, split.labels, **{f"{step}__sample_weight": balanced(split.labels)}
        )
    else:
        model.fit(split.features, split.labels)
    return model


def balanced(labels: npt.NDArray[np.str_]) -> npt.NDArray[np.float64]:
    """Weight each sample inversely to its class size, as class_weight='balanced'."""
    classes, counts = np.unique(labels, return_counts=True)
    per_class = dict(zip(classes, len(labels) / (len(classes) * counts), strict=True))
    return np.array([per_class[label] for label in labels])


def score(model: Pipeline, split: dataset.Split) -> dict[str, float]:
    predicted = model.predict(split.features)
    return {
        "accuracy": float(accuracy_score(split.labels, predicted)),
        "macro_f1": float(f1_score(split.labels, predicted, average="macro")),
    }


def subset(split: dataset.Split, index: npt.NDArray[np.intp]) -> dataset.Split:
    return dataset.Split(
        features=split.features[index],
        labels=split.labels[index],
        signers=split.signers[index],
        videos=split.videos[index],
        angles=[split.angles[i] for i in index],
    )


def measure_latency(model: Pipeline, features: npt.NDArray[np.float64]) -> dict[str, float]:
    """Per-frame inference, one frame at a time -- how the server calls it.

    Batch throughput would look far better and is not what a live session
    sees. Covers the model only; feature extraction is timed by the server's
    own server_ms.
    """
    set_single_threaded(model)
    rng = np.random.default_rng(SEED)
    rows = features[
        rng.choice(len(features), size=min(LATENCY_SAMPLES, len(features)), replace=False)
    ]

    model.predict_proba(rows[:1])  # warm-up
    timings = []
    for row in rows:
        started = time.perf_counter()
        model.predict_proba(row.reshape(1, -1))
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": round(float(np.median(timings)), 4),
        "p95_ms": round(float(np.percentile(timings, 95)), 4),
        "samples": len(timings),
    }


def set_single_threaded(model: Pipeline) -> None:
    """One frame at a time gains nothing from a thread pool and pays its setup
    on every call; the server classifies frames serially per connection."""
    estimator = model.steps[-1][1]
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1


def save(
    model: Pipeline,
    name: str,
    data: dataset.Dataset,
    folds: pd.DataFrame,
    ranking: list[dict[str, Any]],
    test: dict[str, float],
    latency: dict[str, float],
    version: int,
) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS / f"model-v{version}.joblib"
    joblib.dump(model, model_path, compress=3)

    metadata = {
        "version": version,
        "feature_version": data.feature_version,
        "classes": [str(label) for label in model.classes_],
        "model_type": name,
        "metrics": {
            "test": {key: round(value, 4) for key, value in test.items()},
            "cv_signer_grouped": next(row for row in ranking if row["model"] == name),
            "latency_model_only": latency,
        },
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_sha256": data.sha256,
        "participants": {
            "train_signers": len(set(data.train.signers)),
            "test_signers": len(set(data.test.signers)),
            "train_frames": len(data.train),
            "test_frames": len(data.test),
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    meta_path = ARTIFACTS / f"model-v{version}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    folds.to_csv(EXPERIMENTS / f"cv_folds-v{version}.csv", index=False)
    (EXPERIMENTS / f"summary-v{version}.json").write_text(
        json.dumps(
            {"ranking": ranking, "chosen": name, "test": test, "latency_model_only": latency},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    size_mb = model_path.stat().st_size / 1e6
    print(f"\nsaved {model_path} ({size_mb:.2f} MB) and {meta_path.name}")
    if size_mb > 10:
        print("  WARNING: over the 10 MB CLAUDE.md 18 allows committing; do not commit as is.")


if __name__ == "__main__":
    main()
