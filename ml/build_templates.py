"""Reference joint angles per letter, for per-finger hints (CLAUDE.md 6.5).

Built from the training signers only. Each letter's template is the median of
its examples' joint angles -- the median rather than the mean because a few
badly tracked frames would otherwise drag a whole finger's reference off.

Tolerance is derived, not chosen. For every training example of a letter we
take the largest deviation any of its fingers shows from the template -- the
quantity that decides whether feedback.hints_for says anything at all -- and
set the tolerance at the given percentile of it. At the default 80th, about
four in five genuine examples of the letter earn no hint. The held-out test
signers then measure how well that carries over to people the templates were
not built from.

Usage:
    uv run --group ml python ml/build_templates.py [--percentile 80] [--apply]

Writes ml/artifacts/templates-v{FEATURE_VERSION}.json. With --apply, also
stores them in letters.template_angles / tolerance_deg in the database.
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
from app.domain.gesture.stabilizer import NO_SIGN

ARTIFACTS = dataset.ROOT / "ml" / "artifacts"
TOLERANCE_FLOOR_DEG = 10.0
TOLERANCE_CEILING_DEG = 60.0


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--apply", action="store_true", help="write templates to the database")
    args = parser.parse_args()

    print("loading and extracting features ...")
    data = dataset.load()
    letters = [label for label in data.classes if label != NO_SIGN]

    templates = {letter: build(data.train, letter, args.percentile) for letter in letters}
    quiet_rate = measure(data.test, templates)

    payload = {
        "feature_version": FEATURE_VERSION,
        "percentile": args.percentile,
        "built_from": "train split only",
        "dataset_sha256": data.sha256,
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

    # Per example: the worst joint of each finger, then the worst finger. That
    # is exactly the deviation hints_for compares against the tolerance.
    worst = np.abs(angles - median).max(axis=2).max(axis=1)
    tolerance = float(
        np.clip(np.percentile(worst, percentile), TOLERANCE_FLOOR_DEG, TOLERANCE_CEILING_DEG)
    )

    return {
        "angles": {
            finger: {joint: round(float(median[f, j]), 2) for j, joint in enumerate(JOINTS)}
            for f, finger in enumerate(FINGERS)
        },
        "tolerance_deg": round(tolerance, 2),
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
            symbol=letter, angles=entry["angles"], tolerance_deg=entry["tolerance_deg"]
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
    from sqlalchemy import update

    from app.infrastructure.db import models
    from app.infrastructure.db.repositories import DEFAULT_ALPHABET
    from app.infrastructure.db.session import engine, session_factory

    async with session_factory() as session:
        updated = 0
        for letter, entry in templates.items():
            result = await session.execute(
                update(models.Letter)
                .where(
                    models.Letter.alphabet == DEFAULT_ALPHABET,
                    models.Letter.symbol == letter,
                )
                .values(template_angles=entry["angles"], tolerance_deg=entry["tolerance_deg"])
                .returning(models.Letter.id)
            )
            updated += len(result.all())
        await session.commit()
    await engine.dispose()
    print(f"applied {updated} templates to the database")
    if updated != len(templates):
        print(f"  WARNING: expected {len(templates)}; is the database migrated and seeded?")


def report(templates: dict[str, dict[str, Any]], quiet_rate: dict[str, Any], path: Any) -> None:
    tolerances = [entry["tolerance_deg"] for entry in templates.values()]
    print(f"\n{len(templates)} templates -> {path}")
    print(
        f"  tolerance: min {min(tolerances):.1f}, median {np.median(tolerances):.1f}, "
        f"max {max(tolerances):.1f} degrees"
    )
    print(
        f"  unseen signers' genuine examples earning no hint: {quiet_rate['overall']:.1%} "
        "(target by construction on training signers: the percentile)"
    )
    print("  letters where feedback is noisiest on unseen signers:")
    for letter, rate in quiet_rate["worst"]:
        print(f"    {letter}: {rate:.1%} quiet")


if __name__ == "__main__":
    main()
