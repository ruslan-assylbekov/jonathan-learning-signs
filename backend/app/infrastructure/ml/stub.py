"""A stand-in classifier, so the pipeline can be run before a model exists.

NOT A MODEL AND NOT RSL. CLAUDE.md 20 forbids inventing handshape data, and
the real classes come from Bukva once it is extracted and trained. So this
stub refuses to name letters at all. Instead it reads something it can measure
honestly from the landmarks -- which fingers are extended -- and labels that.

A label is ``demo:`` followed by five digits, thumb to pinky, 1 for extended:
``demo:01100`` is index and middle extended. Every class is therefore defined
by geometry alone, which also gives the feedback engine a well-defined
template: an extended finger should be straight, a folded one bent.

Everything here disappears when ``model-v1`` lands. Nothing in it may be
reported as a result (CLAUDE.md 16, honesty rule).
"""

from __future__ import annotations

from typing import Final

from app.domain.entities import Letter, Prediction
from app.domain.gesture.features import FEATURE_VERSION, FINGERS, JOINTS, Features
from app.domain.gesture.feedback import LetterTemplate

LABEL_PREFIX: Final = "demo:"

EXTENDED_MAX_FLEXION: Final = 35.0
"""A finger counts as extended when its worst joint bends less than this."""

FOLDED_REFERENCE_FLEXION: Final = 90.0
"""Reference bend for a folded finger, for hint generation."""

STUB_TOLERANCE_DEG: Final = 40.0
"""Deliberately loose: this is a stub, not a graded exercise."""

#: Targets offered in stub mode: ordinary gestures anyone can make. They are
#: not sign language, and the UI says so. Readable names live in the frontend's
#: i18n, like every other piece of display text.
STUB_TARGETS: Final[tuple[str, ...]] = (
    "demo:11111",
    "demo:00000",
    "demo:01000",
    "demo:01100",
    "demo:10000",
    "demo:00001",
)


def label_for(extended: tuple[bool, ...]) -> str:
    return LABEL_PREFIX + "".join("1" if flag else "0" for flag in extended)


def extended_from_label(label: str) -> tuple[bool, ...]:
    """Parse a stub label back into per-finger flags.

    Raises:
        ValueError: not a stub label, so a caller cannot silently treat a real
            letter as one.
    """
    if not label.startswith(LABEL_PREFIX):
        raise ValueError(f"not a stub label: {label!r}")
    digits = label.removeprefix(LABEL_PREFIX)
    if len(digits) != len(FINGERS) or set(digits) - {"0", "1"}:
        raise ValueError(f"malformed stub label: {label!r}")
    return tuple(digit == "1" for digit in digits)


class StubClassifier:
    """Labels a hand by which of its fingers are extended.

    Implements ``GestureClassifier``. Confidence is a heuristic -- how far each
    finger sits from the extended/folded boundary -- not a probability.
    """

    @property
    def feature_version(self) -> int:
        return FEATURE_VERSION

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(
            label_for(tuple(bool(mask >> bit & 1) for bit in range(len(FINGERS))))
            for mask in range(2 ** len(FINGERS))
        )

    def predict(self, features: Features) -> Prediction:
        bends = tuple(_worst_bend(features, finger) for finger in FINGERS)
        extended = tuple(bend < EXTENDED_MAX_FLEXION for bend in bends)

        # Certainty of one finger: distance from the boundary, saturating at the
        # boundary's own width. The weakest finger decides the whole hand.
        margins = [
            min(1.0, abs(bend - EXTENDED_MAX_FLEXION) / EXTENDED_MAX_FLEXION) for bend in bends
        ]
        confidence = 0.5 + 0.5 * min(margins)

        return Prediction(label=label_for(extended), confidence=confidence)


class StubTemplates:
    """Templates derived from the stub's own class definition.

    Implements ``TemplateSource``. Because a label says exactly which fingers
    should be extended, the reference pose follows from it rather than being
    made up: extended means straight, folded means bent.
    """

    async def template_for(self, label: str) -> LetterTemplate | None:
        try:
            extended = extended_from_label(label)
        except ValueError:
            return None  # a real letter: no stub template for it

        angles = {
            finger: dict.fromkeys(JOINTS, 0.0 if is_extended else FOLDED_REFERENCE_FLEXION)
            for finger, is_extended in zip(FINGERS, extended, strict=True)
        }
        return LetterTemplate(
            symbol=label,
            angles=angles,
            tolerances=dict.fromkeys(FINGERS, STUB_TOLERANCE_DEG),
        )


class StubLetterCatalog:
    """The stub targets as a letter catalog.

    Implements ``LetterCatalog``, so the WebSocket protocol keeps addressing
    targets by ``letter_id`` (CLAUDE.md 7) and nothing has to change when the
    database-backed catalog replaces this.
    """

    async def all(self) -> list[Letter]:
        return [
            Letter(id=index, symbol=symbol, sort_order=index)
            for index, symbol in enumerate(STUB_TARGETS)
        ]

    async def by_id(self, letter_id: int) -> Letter | None:
        if 0 <= letter_id < len(STUB_TARGETS):
            return Letter(id=letter_id, symbol=STUB_TARGETS[letter_id], sort_order=letter_id)
        return None


def _worst_bend(features: Features, finger: str) -> float:
    return max(features.angles[finger][joint] for joint in JOINTS)
