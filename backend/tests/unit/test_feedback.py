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
    MAX_HINTS,
    HintCode,
    LetterTemplate,
    hints_for,
)
from tests import synthetic

#: An arbitrary reference pose; the real ones come from ml/build_templates.py.
REFERENCE = {
    "thumb": (10.0, 20.0, 15.0),
    "index": (0.0, 0.0, 0.0),
    "middle": (0.0, 0.0, 0.0),
    "ring": (40.0, 90.0, 60.0),
    "pinky": (40.0, 90.0, 60.0),
}

TOLERANCE = 15.0


def template(tolerance: float = TOLERANCE) -> LetterTemplate:
    angles = {finger: dict(zip(JOINTS, REFERENCE[finger], strict=True)) for finger in FINGERS}
    return LetterTemplate(symbol="A", angles=angles, tolerance_deg=tolerance)


def features_for(turns: dict[str, tuple[float, float, float]]) -> Features:
    return extract_features(synthetic.hand(turns), Handedness.RIGHT, aspect=1.0)


def test_a_matching_hand_gets_no_hints() -> None:
    assert hints_for(features_for(REFERENCE), template()) == []


def test_deviation_within_tolerance_is_not_worth_mentioning() -> None:
    almost = dict(REFERENCE) | {"index": (0.0, TOLERANCE - 1.0, 0.0)}

    assert hints_for(features_for(almost), template()) == []


def test_a_too_bent_finger_is_told_to_straighten() -> None:
    bent = dict(REFERENCE) | {"index": (0.0, 70.0, 0.0)}

    hints = hints_for(features_for(bent), template())

    assert len(hints) == 1
    assert hints[0].finger == "index"
    assert hints[0].code is HintCode.SHOULD_BE_STRAIGHT
    assert hints[0].deviation_deg == pytest.approx(70.0)


def test_a_too_straight_finger_is_told_to_bend() -> None:
    straightened = dict(REFERENCE) | {"ring": (0.0, 0.0, 0.0)}

    hints = hints_for(features_for(straightened), template())

    assert len(hints) == 1
    assert hints[0].finger == "ring"
    assert hints[0].code is HintCode.SHOULD_BE_BENT
    assert hints[0].deviation_deg == pytest.approx(90.0)


def test_a_finger_is_reported_once_by_its_worst_joint() -> None:
    sloppy = dict(REFERENCE) | {"index": (40.0, 80.0, 50.0)}

    hints = hints_for(features_for(sloppy), template())

    assert [hint.finger for hint in hints] == ["index"]
    assert hints[0].deviation_deg == pytest.approx(80.0)


def test_hints_are_capped_and_ordered_by_deviation() -> None:
    all_wrong = {
        "thumb": (90.0, 20.0, 15.0),
        "index": (0.0, 30.0, 0.0),
        "middle": (0.0, 60.0, 0.0),
        "ring": (40.0, 90.0, 60.0),
        "pinky": (0.0, 0.0, 0.0),
    }

    hints = hints_for(features_for(all_wrong), template())

    assert len(hints) == MAX_HINTS
    assert [hint.finger for hint in hints] == ["pinky", "thumb"]
    assert hints[0].deviation_deg >= hints[1].deviation_deg


def test_a_looser_tolerance_forgives_more() -> None:
    sloppy = dict(REFERENCE) | {"index": (0.0, 30.0, 0.0)}

    assert hints_for(features_for(sloppy), template(tolerance=15.0)) != []
    assert hints_for(features_for(sloppy), template(tolerance=45.0)) == []


def test_template_rejects_a_missing_finger() -> None:
    angles = {
        finger: dict(zip(JOINTS, REFERENCE[finger], strict=True))
        for finger in FINGERS
        if finger != "pinky"
    }

    with pytest.raises(ValueError, match="pinky"):
        LetterTemplate(symbol="A", angles=angles, tolerance_deg=TOLERANCE)


def test_template_rejects_a_missing_joint() -> None:
    angles = {finger: dict(zip(JOINTS, REFERENCE[finger], strict=True)) for finger in FINGERS}
    del angles["thumb"]["dip"]

    with pytest.raises(ValueError, match="thumb.dip"):
        LetterTemplate(symbol="A", angles=angles, tolerance_deg=TOLERANCE)


@pytest.mark.parametrize("tolerance", [0.0, -5.0])
def test_template_rejects_a_nonsensical_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        template(tolerance=tolerance)
