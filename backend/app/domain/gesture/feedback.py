"""Per-finger hints: which finger is wrong, which way, and how far.

CLAUDE.md 6.5. Each letter carries ``template_angles`` -- the median joint
angles of its training examples, built by ``ml/build_templates.py`` -- and a
tolerance per finger. A finger whose worst joint deviates beyond its own
tolerance earns a hint. At most two hints per frame, the fingers furthest
outside their tolerance first.

Tolerance is per finger because fingers are not equally noisy: across the
dataset the ring and little fingers vary about twice as much as the thumb and
index (largely occlusion -- MediaPipe guesses what it cannot see). One number
per letter, applied to the worst of five fingers, had to be wide enough for the
noisiest and so said nothing useful about the rest.

Hints name the target state, not just a direction, because "bend your index
finger" is ambiguous when the letter wants it half bent. This module emits
**codes**, never sentences: the wording lives in ``frontend/src/i18n.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.domain.gesture.features import FINGERS, JOINTS, Features

MAX_HINTS: Final = 2
"""Hints sent per frame (CLAUDE.md 6.5)."""

HALF_BENT_FROM_DEG: Final = 55.0
FULLY_BENT_FROM_DEG: Final = 130.0
"""Boundaries on a finger's total flexion (sum of its three joint angles).

Read off the data, not chosen: in the templates built from Bukva, every one of
the index, middle, ring and little fingers leaves a gap with no letter in it
between about 53 and 61 degrees, and again between about 118 and 138. The
boundaries sit inside those gaps. They do not apply to the thumb, whose values
form one continuous band -- its bend across the palm is mostly rotation, which
these angles barely register -- so thumb hints give a direction only.
"""


class FingerState(StrEnum):
    STRAIGHT = "straight"
    HALF_BENT = "half_bent"
    FULLY_BENT = "fully_bent"


class HintCode(StrEnum):
    """What to do with a finger, in terms of where it should end up."""

    STRAIGHTEN = "STRAIGHTEN"
    """Target is straight; the finger is bent."""
    BEND_HALFWAY = "BEND_HALFWAY"
    """Target is half bent; the finger is straighter than that."""
    ONLY_HALFWAY = "ONLY_HALFWAY"
    """Target is half bent; the finger is bent further than that."""
    BEND_FULLY = "BEND_FULLY"
    """Target is fully bent; the finger is straighter than that."""
    BEND_MORE = "BEND_MORE"
    """Direction only: thumb, or a target state the boundaries do not cover."""
    BEND_LESS = "BEND_LESS"
    """Direction only, the other way."""


@dataclass(frozen=True, slots=True)
class Hint:
    """One piece of advice about one finger.

    ``deviation_deg`` exists to rank hints and to make tests legible; the
    WebSocket schema serializes only ``finger`` and ``code`` (CLAUDE.md 7).
    """

    finger: str
    code: HintCode
    deviation_deg: float
    excess: float
    """How many tolerances out the finger is. Ranks hints across fingers whose
    tolerances differ, where raw degrees would favour the noisy ones."""


@dataclass(frozen=True, slots=True)
class LetterTemplate:
    """Reference pose for one letter.

    Args:
        symbol: the letter this template describes.
        angles: ``angles[finger][joint]`` in degrees, every finger and joint
            present -- a gap here would silently disable hints for that finger.
        tolerances: degrees each finger's worst joint may deviate before it
            earns a hint, every finger present.
    """

    symbol: str
    angles: dict[str, dict[str, float]]
    tolerances: dict[str, float]

    def __post_init__(self) -> None:
        missing = [
            f"{finger}.{joint}"
            for finger in FINGERS
            for joint in JOINTS
            if joint not in self.angles.get(finger, {})
        ]
        missing += [f"tolerance.{finger}" for finger in FINGERS if finger not in self.tolerances]
        if missing:
            raise ValueError(f"template for {self.symbol!r} is missing: {missing}")

        bad = {finger: value for finger, value in self.tolerances.items() if value <= 0.0}
        if bad:
            raise ValueError(f"template for {self.symbol!r} has non-positive tolerance: {bad}")

    def state_of(self, finger: str) -> FingerState | None:
        """The state the letter wants this finger in, or None for the thumb."""
        if finger == "thumb":
            return None
        return finger_state(sum(self.angles[finger].values()))


def finger_state(total_flexion_deg: float) -> FingerState:
    if total_flexion_deg >= FULLY_BENT_FROM_DEG:
        return FingerState.FULLY_BENT
    if total_flexion_deg >= HALF_BENT_FROM_DEG:
        return FingerState.HALF_BENT
    return FingerState.STRAIGHT


def hints_for(features: Features, template: LetterTemplate) -> list[Hint]:
    """Rank the user's fingers against the template and advise on the worst.

    Returns an empty list when every finger is within its tolerance -- the
    normal case once the learner has the sign right.
    """
    candidates = (_hint_for_finger(features, template, finger) for finger in FINGERS)
    offenders = [hint for hint in candidates if hint is not None]
    offenders.sort(key=lambda hint: hint.excess, reverse=True)
    return offenders[:MAX_HINTS]


def _hint_for_finger(features: Features, template: LetterTemplate, finger: str) -> Hint | None:
    """Judge one finger by its single most deviating joint.

    A finger is reported once: three hints about one thumb would crowd out the
    other fingers.
    """
    signed = max(
        (features.angles[finger][joint] - template.angles[finger][joint] for joint in JOINTS),
        key=abs,
    )
    deviation = abs(signed)
    tolerance = template.tolerances[finger]
    if deviation <= tolerance:
        return None

    too_bent = signed > 0  # flexion grows as a joint bends
    return Hint(
        finger=finger,
        code=_code(template.state_of(finger), too_bent=too_bent),
        deviation_deg=deviation,
        excess=deviation / tolerance,
    )


def _code(target: FingerState | None, *, too_bent: bool) -> HintCode:
    if target is FingerState.STRAIGHT:
        return HintCode.STRAIGHTEN if too_bent else HintCode.BEND_MORE
    if target is FingerState.HALF_BENT:
        return HintCode.ONLY_HALFWAY if too_bent else HintCode.BEND_HALFWAY
    if target is FingerState.FULLY_BENT:
        return HintCode.BEND_LESS if too_bent else HintCode.BEND_FULLY
    return HintCode.BEND_LESS if too_bent else HintCode.BEND_MORE
