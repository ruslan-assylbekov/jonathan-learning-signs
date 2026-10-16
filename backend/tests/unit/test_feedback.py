"""Per-finger hint generation (CLAUDE.md 6.5)."""

from __future__ import annotations

import pytest

from app.domain.gesture.features import (
    FINGERS,
    JOINTS,
    Features,
    Handedness,
    extract_features,
)
from app.domain.gesture.feedback import (
    FULLY_BENT_FROM_DEG,
    HALF_BENT_FROM_DEG,
    MAX_HINTS,
    FingerState,
    HintCode,
    LetterTemplate,
    finger_state,
    hints_for,
)
from tests import synthetic

Turns = dict[str, tuple[float, float, float]]

#: One finger in each state, so every code path has a finger to exercise it.
REFERENCE: Turns = {
    "thumb": (10.0, 20.0, 15.0),
    "index": (0.0, 0.0, 0.0),  # straight, total 0
    "middle": (30.0, 30.0, 30.0),  # half bent, total 90
    "ring": (60.0, 70.0, 60.0),  # fully bent, total 190
    "pinky": (60.0, 70.0, 60.0),
}
TOLERANCE = 15.0


def template(tolerances: dict[str, float] | None = None) -> LetterTemplate:
    angles = {finger: dict(zip(JOINTS, REFERENCE[finger], strict=True)) for finger in FINGERS}
    return LetterTemplate(
        symbol="A",
        angles=angles,
        tolerances=tolerances or dict.fromkeys(FINGERS, TOLERANCE),
    )


def features_for(turns: Turns) -> Features:
    return extract_features(synthetic.hand(turns), Handedness.RIGHT, aspect=1.0)


def with_finger(finger: str, turns: tuple[float, float, float]) -> Turns:
    return dict(REFERENCE) | {finger: turns}


def only_hint(turns: Turns, tmpl: LetterTemplate | None = None) -> tuple[str, HintCode]:
    hints = hints_for(features_for(turns), tmpl or template())
    assert len(hints) == 1, hints
    return hints[0].finger, hints[0].code


def test_a_matching_hand_gets_no_hints() -> None:
    assert hints_for(features_for(REFERENCE), template()) == []


def test_deviation_within_tolerance_is_not_worth_mentioning() -> None:
    almost = with_finger("index", (0.0, TOLERANCE - 1.0, 0.0))

    assert hints_for(features_for(almost), template()) == []


# --- What each hint says ----------------------------------------------------


def test_a_bent_finger_that_should_be_straight_is_told_to_straighten() -> None:
    assert only_hint(with_finger("index", (0.0, 70.0, 0.0))) == ("index", HintCode.STRAIGHTEN)


def test_a_straight_finger_that_should_be_half_bent_is_told_halfway() -> None:
    assert only_hint(with_finger("middle", (0.0, 0.0, 0.0))) == ("middle", HintCode.BEND_HALFWAY)


def test_a_fully_bent_finger_that_should_be_half_bent_is_told_only_halfway() -> None:
    over = with_finger("middle", (60.0, 70.0, 60.0))

    assert only_hint(over) == ("middle", HintCode.ONLY_HALFWAY)


def test_a_straight_finger_that_should_be_fully_bent_is_told_fully() -> None:
    assert only_hint(with_finger("ring", (0.0, 0.0, 0.0))) == ("ring", HintCode.BEND_FULLY)


@pytest.mark.parametrize(
    ("thumb", "code"),
    [((10.0, 60.0, 15.0), HintCode.BEND_LESS), ((10.0, 0.0, 0.0), HintCode.BEND_MORE)],
    ids=["too-bent", "too-straight"],
)
def test_the_thumb_gets_a_direction_only(thumb: tuple[float, float, float], code: HintCode) -> None:
    """Thumb flexion has no clusters in the data to name states after."""
    assert only_hint(with_finger("thumb", thumb)) == ("thumb", code)


# --- Per-finger tolerance ---------------------------------------------------


def test_each_finger_is_held_to_its_own_tolerance() -> None:
    """The same 30-degree slip is an error for a precise finger and noise for
    a noisy one -- the reason tolerance is per finger at all."""
    loose_ring = dict.fromkeys(FINGERS, 20.0) | {"ring": 45.0}
    slips = dict(REFERENCE) | {"index": (0.0, 30.0, 0.0), "ring": (60.0, 40.0, 60.0)}

    hints = hints_for(features_for(slips), template(loose_ring))

    assert [hint.finger for hint in hints] == ["index"]


def test_hints_rank_by_how_far_outside_tolerance_not_raw_degrees() -> None:
    tolerances = dict.fromkeys(FINGERS, TOLERANCE) | {"index": 10.0, "ring": 40.0}
    both = dict(REFERENCE) | {"index": (0.0, 30.0, 0.0), "ring": (60.0, 20.0, 60.0)}

    hints = hints_for(features_for(both), template(tolerances))

    # ring is off by 50 degrees, index by 30, but index is three tolerances out
    # and ring barely more than one.
    assert [hint.finger for hint in hints] == ["index", "ring"]
    assert hints[0].excess > hints[1].excess


def test_hints_are_capped() -> None:
    all_wrong: Turns = {
        "thumb": (60.0, 70.0, 60.0),
        "index": (60.0, 70.0, 60.0),
        "middle": (0.0, 0.0, 0.0),
        "ring": (0.0, 0.0, 0.0),
        "pinky": (0.0, 0.0, 0.0),
    }

    assert len(hints_for(features_for(all_wrong), template())) == MAX_HINTS


def test_a_finger_is_reported_once_by_its_worst_joint() -> None:
    sloppy = with_finger("index", (40.0, 80.0, 50.0))

    hints = hints_for(features_for(sloppy), template())

    assert [hint.finger for hint in hints] == ["index"]
    assert hints[0].deviation_deg == pytest.approx(80.0)


# --- Finger states ----------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "state"),
    [
        (0.0, FingerState.STRAIGHT),
        (HALF_BENT_FROM_DEG - 0.1, FingerState.STRAIGHT),
        (HALF_BENT_FROM_DEG, FingerState.HALF_BENT),
        (FULLY_BENT_FROM_DEG - 0.1, FingerState.HALF_BENT),
        (FULLY_BENT_FROM_DEG, FingerState.FULLY_BENT),
        (260.0, FingerState.FULLY_BENT),
    ],
)
def test_finger_state_boundaries(total: float, state: FingerState) -> None:
    assert finger_state(total) is state


def test_the_thumb_has_no_named_state() -> None:
    assert template().state_of("thumb") is None
    assert template().state_of("middle") is FingerState.HALF_BENT


# --- Template validation ----------------------------------------------------


def test_template_rejects_a_missing_finger() -> None:
    angles = {
        finger: dict(zip(JOINTS, REFERENCE[finger], strict=True))
        for finger in FINGERS
        if finger != "pinky"
    }

    with pytest.raises(ValueError, match="pinky"):
        LetterTemplate(symbol="A", angles=angles, tolerances=dict.fromkeys(FINGERS, TOLERANCE))


def test_template_rejects_a_missing_tolerance() -> None:
    tolerances = dict.fromkeys(("thumb", "index", "middle", "ring"), TOLERANCE)

    with pytest.raises(ValueError, match="tolerance.pinky"):
        template(tolerances)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_template_rejects_a_nonsensical_tolerance(bad: float) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        template(dict.fromkeys(FINGERS, TOLERANCE) | {"index": bad})
