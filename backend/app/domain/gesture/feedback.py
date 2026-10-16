"""Per-finger hints: which finger is wrong, and which way.

CLAUDE.md 6.5. Each letter carries ``template_angles`` -- the mean joint angles
measured from the dataset by ``ml/build_templates.py`` -- plus a tolerance in
degrees. A finger whose angles deviate beyond the tolerance earns a hint code.
At most two hints per frame, largest deviation first, so the learner is told
one or two things rather than five.

This module emits **codes**, never sentences: the human-readable text lives in
``frontend/src/i18n.ts`` so it can be translated without touching the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.domain.gesture.features import FINGERS, JOINTS, Features

MAX_HINTS: Final = 2
"""Hints sent per frame (CLAUDE.md 6.5)."""


class HintCode(StrEnum):
    """What the learner should do with the finger.

    Flexion is 0 when a joint is straight and grows as it bends, so a user
    angle *above* the template means the finger is too bent.
    """

    SHOULD_BE_STRAIGHT = "SHOULD_BE_STRAIGHT"
    SHOULD_BE_BENT = "SHOULD_BE_BENT"


@dataclass(frozen=True, slots=True)
class Hint:
    """One piece of advice about one finger.

    ``deviation_deg`` exists to rank hints and to make tests legible; the
    WebSocket schema serializes only ``finger`` and ``code`` (CLAUDE.md 7).
    """

    finger: str
    code: HintCode
    deviation_deg: float


@dataclass(frozen=True, slots=True)
class LetterTemplate:
    """Reference pose for one letter.

    Args:
        symbol: the letter this template describes.
        angles: ``angles[finger][joint]`` in degrees, every finger and joint
            present -- a gap here would silently disable hints for that finger.
        tolerance_deg: how far a joint may deviate before it earns a hint.
    """

    symbol: str
    angles: dict[str, dict[str, float]]
    tolerance_deg: float

    def __post_init__(self) -> None:
        if self.tolerance_deg <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance_deg}")

        missing = [
            f"{finger}.{joint}"
            for finger in FINGERS
            for joint in JOINTS
            if joint not in self.angles.get(finger, {})
        ]
        if missing:
            raise ValueError(f"template for {self.symbol!r} is missing angles: {missing}")


def hints_for(features: Features, template: LetterTemplate) -> list[Hint]:
    """Rank the user's fingers against the template and advise on the worst.

    Returns an empty list when every finger is within tolerance -- which is the
    normal case once the learner has the sign right.
    """
    worst = (_worst_joint(features, template, finger) for finger in FINGERS)
    offenders = [hint for hint in worst if hint is not None]
    offenders.sort(key=lambda hint: hint.deviation_deg, reverse=True)
    return offenders[:MAX_HINTS]


def _worst_joint(features: Features, template: LetterTemplate, finger: str) -> Hint | None:
    """The single most deviating joint of one finger, if it is out of tolerance.

    A finger is reported once, by its worst joint: three hints about one thumb
    would crowd out the other fingers.
    """
    signed = max(
        (features.angles[finger][joint] - template.angles[finger][joint] for joint in JOINTS),
        key=abs,
    )
    deviation = abs(signed)
    if deviation <= template.tolerance_deg:
        return None

    code = HintCode.SHOULD_BE_STRAIGHT if signed > 0 else HintCode.SHOULD_BE_BENT
    return Hint(finger=finger, code=code, deviation_deg=deviation)
