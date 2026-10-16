"""Reference joint angles per letter, for per-finger hints (CLAUDE.md 6.5).

Built from the training signers only. Each letter's template is the median of
its examples' joint angles -- the median rather than the mean because a few
badly tracked frames would otherwise drag a whole finger's reference off.

Tolerance is derived per finger, not chosen. For every training example of a
letter we take each finger's worst-joint deviation from the template -- the
quantity feedback.hints_for compares -- and set that finger's tolerance at the
given percentile of it. Per finger because fingers are not equally noisy: ring
and little finger vary about twice as much as thumb and index, and a single
tolerance had to be wide enough for the worst of them. The held-out test
signers then measure how often genuine examples stay hint-free.

Usage:
    uv run --group ml python ml/build_templates.py [--percentile 90] [--apply]

Writes ml/artifacts/templates-v{FEATURE_VERSION}.json. With --apply, also
stores them in letters.template_angles / finger_tolerances in the database.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from typing import Any

import dataset
import numpy as np
import numpy.typing as npt

from app.domain.gesture.features import FEATURE_VERSION, FINGERS, JOINTS
from app.domain.gesture.feedback import LetterTemplate, hints_for

ARTIFACTS = dataset.ROOT / "ml" / "artifacts"
TOLERANCE_FLOOR_DEG = 10.0
TOLERANCE_CEILING_DEG = 60.0


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--apply", action="store_true", help="write templates to the database")
    args = parser.parse_args()

    print("loading and extracting features ...")
    data = dataset.load()
    letters = list(data.classes)

    templates = {letter: build(data.train, letter, args.percentile) for letter in letters}
    # Calibrate on training signers, report on test ones: the percentile is
    # picked by looking at train_quiet only, so the test figure stays honest.
    train_quiet = measure(data.train, templates)
    quiet_rate = measure(data.test, templates)
    print(f"  training signers' genuine examples earning no hint: {train_quiet['overall']:.1%}")

    payload = {
        "feature_version": FEATURE_VERSION,
        "percentile": args.percentile,
        "built_from": "train split only",
        "dataset_sha256": data.sha256,
        "train_quiet_rate": train_quiet["overall"],
        "test_quiet_rate": quiet_rate,
        "letters": templates,
    }
    path = ARTIFACTS / f"templates-v{FEATURE_VERSION}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report(templates, quiet_rate, path)

    if args.apply:
        asyncio.run(apply(templates))


def build(split: dataset.Split, letter: str, percentile: float) -> dict[str, Any]:
    mask = split.labels == letter
    angles = dataset.angle_matrix(split)[mask]  # (n, finger, joint)
    median = np.median(angles, axis=0)

    # Per example and finger: that finger's worst joint -- exactly what
    # hints_for compares against the finger's tolerance.
    worst = np.abs(angles - median).max(axis=2)  # (n, finger)
    tolerances = np.clip(
        np.percentile(worst, percentile, axis=0), TOLERANCE_FLOOR_DEG, TOLERANCE_CEILING_DEG
    )

    return {
        "angles": {
            finger: {joint: round(float(median[f, j]), 2) for j, joint in enumerate(JOINTS)}
            for f, finger in enumerate(FINGERS)
        },
        "tolerances": {finger: round(float(tolerances[f]), 2) for f, finger in enumerate(FINGERS)},
        "examples": int(mask.sum()),
    }


def measure(split: dataset.Split, templates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Share of genuine examples from unseen signers that earn no hint.

    The number that says whether feedback nags people who are already right.
    Uses the real hints_for, through the same Features objects the server
    builds, rather than re-deriving its logic here.
    """
    from app.domain.gesture.features import Features

    per_letter: dict[str, float] = {}
    for letter, entry in templates.items():
        template = LetterTemplate(
            symbol=letter, angles=entry["angles"], tolerances=entry["tolerances"]
        )
        indices = np.flatnonzero(split.labels == letter)
        quiet = [
            not hints_for(Features(vector=split.features[i], angles=split.angles[i]), template)
            for i in indices
        ]
        per_letter[letter] = round(float(np.mean(quiet)), 4) if quiet else float("nan")

    values: npt.NDArray[np.float64] = np.array(list(per_letter.values()))
    return {
        "overall": round(float(np.nanmean(values)), 4),
        "worst": sorted(per_letter.items(), key=lambda item: item[1])[:5],
        "per_letter": per_letter,
    }


async def apply(templates: dict[str, dict[str, Any]]) -> None:
    """Through the backend's loader, so dev and server share one code path."""
    from app.infrastructure.db import load_templates
    from app.infrastructure.db.session import engine, session_factory

    updated = await load_templates.apply(templates, session_factory)
    await engine.dispose()
    print(f"applied {updated} templates to the database")
    if updated != len(templates):
        print(f"  WARNING: expected {len(templates)}; is the database migrated and seeded?")


def report(templates: dict[str, dict[str, Any]], quiet_rate: dict[str, Any], path: Any) -> None:
    print(f"\n{len(templates)} templates -> {path}")
    print("  median tolerance per finger, degrees:")
    for finger in FINGERS:
        values = [entry["tolerances"][finger] for entry in templates.values()]
        at_ceiling = sum(value >= TOLERANCE_CEILING_DEG for value in values)
        print(f"    {finger:<7} {np.median(values):5.1f}   ({at_ceiling} letters at the ceiling)")
    print(
        f"  unseen signers' genuine examples earning no hint: {quiet_rate['overall']:.1%} "
        "(each finger alone is quiet at the percentile on training signers; all five at once less)"
    )
    print("  letters where feedback is noisiest on unseen signers:")
    for letter, rate in quiet_rate["worst"]:
        print(f"    {letter}: {rate:.1%} quiet")


if __name__ == "__main__":
    main()
