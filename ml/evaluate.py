"""Report a trained model on the held-out test signers (CLAUDE.md 8.4).

The test split is the official Bukva one, signer-disjoint from training, so
every number here is about people the model has never seen. Nothing in this
script feeds back into model choice; train.py made that choice on training
signers alone.

Usage:
    uv run --group ml python ml/evaluate.py [--version 1]

Writes experiments/models/eval-vN.json and eval-vN.md, plus the confusion
matrix as confusion-vN.csv (rows: true class, columns: predicted).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from typing import Any

import dataset
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from app.config import settings
from app.domain.gesture.features import FEATURE_VERSION
from app.domain.gesture.stabilizer import NO_SIGN

ARTIFACTS = dataset.ROOT / "ml" / "artifacts"
EXPERIMENTS = dataset.ROOT / "experiments" / "models"


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.confidence_threshold,
        help="confidence the stabilizer requires (default: CONFIDENCE_THRESHOLD)",
    )
    args = parser.parse_args()

    metadata = json.loads((ARTIFACTS / f"model-v{args.version}.json").read_text(encoding="utf-8"))
    if metadata["feature_version"] != FEATURE_VERSION:
        raise SystemExit(
            f"model-v{args.version} was trained on feature version "
            f"{metadata['feature_version']}, the code produces {FEATURE_VERSION}. Retrain."
        )
    model = joblib.load(ARTIFACTS / f"model-v{args.version}.joblib")

    print("loading and extracting features ...")
    test = dataset.load().test
    classes = [str(label) for label in model.classes_]

    probabilities = model.predict_proba(test.features)
    predicted = np.array(classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)

    report = build_report(test, classes, predicted, confidence, args.threshold, metadata)
    write(report, test.labels, predicted, classes, args.version)
    print_summary(report)


def build_report(
    test: dataset.Split,
    classes: list[str],
    predicted: npt.NDArray[np.str_],
    confidence: npt.NDArray[np.float64],
    threshold: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    truth = test.labels
    correct = predicted == truth
    per_class = classification_report(
        truth, predicted, labels=classes, output_dict=True, zero_division=0
    )

    return {
        "model": f"model-v{metadata['version']} ({metadata['model_type']})",
        "test_frames": int(len(truth)),
        "test_signers": int(len(set(test.signers))),
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro")), 4),
        "per_class": {
            label: {
                "precision": round(per_class[label]["precision"], 4),
                "recall": round(per_class[label]["recall"], 4),
                "f1": round(per_class[label]["f1-score"], 4),
                "support": int(per_class[label]["support"]),
            }
            for label in classes
        },
        "top_confusions": top_confusions(truth, predicted, classes),
        "threshold": threshold_behaviour(truth, predicted, confidence, correct, threshold),
        "per_video_majority": per_video_accuracy(test, predicted),
        "not_measured": [
            "accuracy on own webcam data (CLAUDE.md 8.4 domain-shift check): "
            "no webcam set recorded yet",
        ],
    }


def top_confusions(
    truth: npt.NDArray[np.str_],
    predicted: npt.NDArray[np.str_],
    classes: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """The mistakes that matter most: share of a class's frames taken by another."""
    matrix = confusion_matrix(truth, predicted, labels=classes)
    totals = matrix.sum(axis=1)
    pairs: list[dict[str, Any]] = [
        {
            "true": classes[i],
            "predicted": classes[j],
            "frames": int(matrix[i, j]),
            "share_of_true": round(float(matrix[i, j] / totals[i]), 4) if totals[i] else 0.0,
        }
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and matrix[i, j]
    ]
    return sorted(pairs, key=lambda pair: pair["share_of_true"], reverse=True)[:limit]


def threshold_behaviour(
    truth: npt.NDArray[np.str_],
    predicted: npt.NDArray[np.str_],
    confidence: npt.NDArray[np.float64],
    correct: npt.NDArray[np.bool_],
    threshold: float,
) -> dict[str, Any]:
    """How the stabilizer's confidence gate treats real frames.

    A frame counts toward holding a letter only when it is predicted right *and*
    above the threshold, so a correct prediction below it is a frame the learner
    earned and did not get. A wrong prediction above it is the worse failure:
    with no_sign excluded, that is a letter the app would accept as shown.
    """
    is_letter = truth != NO_SIGN
    wrong_and_confident = (~correct) & (confidence >= threshold) & (predicted != NO_SIGN)
    return {
        "value": threshold,
        "correct_frames_passing": round(float((confidence[correct] >= threshold).mean()), 4),
        "wrong_frames_passing_as_a_letter": round(float(wrong_and_confident.mean()), 4),
        "letter_frames_usable_for_hold": round(
            float((correct & (confidence >= threshold))[is_letter].mean()), 4
        ),
    }


def per_video_accuracy(test: dataset.Split, predicted: npt.NDArray[np.str_]) -> dict[str, Any]:
    """Accuracy when each clip is judged by the majority of its frames.

    Closer to what the app does -- it judges a held sign over several frames,
    not one frame -- though not identical, since the app needs consecutive hits.
    """
    frame = pd.DataFrame({"video": test.videos, "truth": test.labels, "predicted": predicted})
    votes = frame.groupby("video").agg(
        truth=("truth", "first"), predicted=("predicted", lambda column: column.mode().iloc[0])
    )
    return {
        "videos": int(len(votes)),
        "accuracy": round(float((votes["truth"] == votes["predicted"]).mean()), 4),
    }


def write(
    report: dict[str, Any],
    truth: npt.NDArray[np.str_],
    predicted: npt.NDArray[np.str_],
    classes: list[str],
    version: int,
) -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS / f"eval-v{version}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(truth, predicted, labels=classes), index=classes, columns=classes
    ).to_csv(EXPERIMENTS / f"confusion-v{version}.csv", encoding="utf-8")
    (EXPERIMENTS / f"eval-v{version}.md").write_text(markdown(report), encoding="utf-8")


def markdown(report: dict[str, Any]) -> str:
    """The same numbers, readable. Generated, never hand-edited (CLAUDE.md 16)."""
    lines = [
        f"# Evaluation: {report['model']}",
        "",
        "Generated by `ml/evaluate.py` on the official Bukva test split: "
        f"{report['test_frames']} frames from {report['test_signers']} signers, none of "
        "whom appear in training.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| frame accuracy | {report['accuracy']:.4f} |",
        f"| macro F1 | {report['macro_f1']:.4f} |",
        f"| clip accuracy (majority of frames, {report['per_video_majority']['videos']} clips) "
        f"| {report['per_video_majority']['accuracy']:.4f} |",
        "",
        f"## Confidence threshold ({report['threshold']['value']})",
        "",
        f"- correct frames clearing it: {report['threshold']['correct_frames_passing']:.4f}",
        f"- letter frames usable toward a hold: "
        f"{report['threshold']['letter_frames_usable_for_hold']:.4f}",
        f"- wrong frames clearing it as a letter: "
        f"{report['threshold']['wrong_frames_passing_as_a_letter']:.4f}",
        "",
        "## Most frequent confusions",
        "",
        "| true | predicted | share of true | frames |",
        "|---|---|---|---|",
        *[
            f"| {pair['true']} | {pair['predicted']} | {pair['share_of_true']:.4f} "
            f"| {pair['frames']} |"
            for pair in report["top_confusions"]
        ],
        "",
        "## Per class",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
        *[
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} "
            f"| {row['support']} |"
            for label, row in sorted(report["per_class"].items(), key=lambda item: item[1]["f1"])
        ],
        "",
        "## Not measured",
        "",
        *[f"- TODO: {item}" for item in report["not_measured"]],
        "",
        "Not comparable with the Bukva paper's 83.6% top-1: that is a video model over",
        "all 33 letters including dynamic ones; this is a single-frame landmark",
        "classifier over 25 static letters plus no_sign.",
        "",
    ]
    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    print(f"\n{report['model']} on {report['test_signers']} unseen signers")
    print(f"  frame accuracy {report['accuracy']:.4f}, macro F1 {report['macro_f1']:.4f}")
    print(f"  clip accuracy (majority vote) {report['per_video_majority']['accuracy']:.4f}")
    gate = report["threshold"]
    print(
        f"  threshold {gate['value']}: "
        f"{gate['correct_frames_passing']:.1%} of correct frames pass, "
        f"{gate['wrong_frames_passing_as_a_letter']:.1%} of frames are wrong but pass as a letter"
    )
    print("  worst classes by F1:")
    for label, row in sorted(report["per_class"].items(), key=lambda item: item[1]["f1"])[:5]:
        print(f"    {label:<8} F1 {row['f1']:.4f}  (recall {row['recall']:.4f})")
    print("  top confusions:")
    for pair in report["top_confusions"][:5]:
        print(f"    {pair['true']} -> {pair['predicted']}: {pair['share_of_true']:.1%}")


if __name__ == "__main__":
    main()
