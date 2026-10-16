"""Normalization invariants for the feature extractor (CLAUDE.md 13)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.domain.gesture.features import (
    FEATURE_LENGTH,
    FEATURE_VERSION,
    FINGERS,
    JOINTS,
    NUM_COORDS,
    Features,
    Handedness,
    InvalidLandmarksError,
    extract_features,
)
from tests import synthetic

BENT = {
    "thumb": (12.0, 30.0, 25.0),
    "index": (5.0, 80.0, 70.0),
    "middle": (0.0, 0.0, 0.0),
    "ring": (40.0, 95.0, 60.0),
    "pinky": (35.0, 90.0, 55.0),
}


def right(landmarks: list[list[float]], *, mirrored: bool = False) -> Features:
    return extract_features(landmarks, Handedness.RIGHT, mirrored=mirrored)


def test_vector_has_the_documented_shape() -> None:
    features = right(synthetic.hand())

    assert features.vector.shape == (FEATURE_LENGTH,)
    assert features.vector.dtype == np.float64
    assert features.feature_version == FEATURE_VERSION


def test_extended_hand_has_straight_joints() -> None:
    features = right(synthetic.hand())

    for finger in FINGERS:
        for joint in JOINTS:
            assert features.angles[finger][joint] == pytest.approx(0.0, abs=1e-9)


def test_angles_round_trip_the_turns_they_were_built_from() -> None:
    features = right(synthetic.hand(BENT))

    for finger in FINGERS:
        for index, joint in enumerate(JOINTS):
            assert features.angles[finger][joint] == pytest.approx(BENT[finger][index])


def test_angles_are_appended_after_the_coordinates() -> None:
    features = right(synthetic.hand(BENT))

    assert features.vector[NUM_COORDS:] == pytest.approx(synthetic.flatten_turns(BENT))


def test_wrist_sits_at_the_origin_and_middle_mcp_at_unit_distance() -> None:
    coords = right(synthetic.hand(BENT)).vector[:NUM_COORDS].reshape(21, 3)

    assert coords[0] == pytest.approx([0.0, 0.0, 0.0])
    assert float(np.linalg.norm(coords[9])) == pytest.approx(1.0)


def test_invariant_under_translation() -> None:
    moved = [[x + 0.17, y - 0.23, z + 0.4] for x, y, z in synthetic.hand(BENT)]

    assert right(moved).vector == pytest.approx(right(synthetic.hand(BENT)).vector)


def test_invariant_under_scale() -> None:
    far = synthetic.hand(BENT, segment=0.012)
    near = synthetic.hand(BENT, segment=0.085)

    assert right(far).vector == pytest.approx(right(near).vector)


def test_mirrored_flag_undoes_a_mirrored_frame() -> None:
    landmarks = synthetic.hand(BENT)

    # A mirrored preview flips the image and MediaPipe's handedness label with
    # it, so the same physical right hand arrives labelled "Left".
    from_preview = extract_features(synthetic.mirror(landmarks), Handedness.LEFT, mirrored=True)

    assert from_preview.vector == pytest.approx(right(landmarks).vector)


def test_left_hand_is_folded_onto_the_right() -> None:
    right_hand = synthetic.hand(BENT)
    left_hand = synthetic.mirror(right_hand)

    from_left = extract_features(left_hand, Handedness.LEFT)

    assert from_left.vector == pytest.approx(right(right_hand).vector)


def test_orientation_is_kept_in_the_coordinates() -> None:
    """Angles are rotation-invariant; the coordinates deliberately are not.

    A sign shown sideways must not look identical to one shown upright, so the
    classifier can still learn orientation.
    """
    upright = synthetic.hand(BENT)
    turned = [list(_rotate(point, synthetic.hand()[0], 35.0)) for point in upright]

    upright_features = right(upright)
    turned_features = right(turned)

    assert turned_features.vector[NUM_COORDS:] == pytest.approx(
        upright_features.vector[NUM_COORDS:]
    )
    assert not np.allclose(
        turned_features.vector[:NUM_COORDS], upright_features.vector[:NUM_COORDS]
    )


@pytest.mark.parametrize(
    "landmarks",
    [
        pytest.param(synthetic.hand()[:20], id="too-few"),
        pytest.param(synthetic.hand() + [[0.1, 0.1, 0.1]], id="too-many"),
        pytest.param([[0.1, 0.2] for _ in range(21)], id="missing-z"),
        pytest.param([[0.1, 0.2, "x"] for _ in range(21)], id="not-numeric"),
    ],
)
def test_rejects_malformed_input(landmarks: list[list[float]]) -> None:
    with pytest.raises(InvalidLandmarksError):
        right(landmarks)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_values(bad: float) -> None:
    landmarks = synthetic.hand()
    landmarks[7][1] = bad

    with pytest.raises(InvalidLandmarksError):
        right(landmarks)


def test_rejects_a_hand_with_no_measurable_size() -> None:
    collapsed = [[0.5, 0.5, 0.0] for _ in range(21)]

    with pytest.raises(InvalidLandmarksError):
        right(collapsed)


def test_rejects_a_zero_length_bone() -> None:
    landmarks = synthetic.hand()
    landmarks[6] = list(landmarks[5])

    with pytest.raises(InvalidLandmarksError):
        right(landmarks)


def _rotate(point: list[float], about: list[float], degrees: float) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    dx, dy = point[0] - about[0], point[1] - about[1]
    return (
        about[0] + dx * math.cos(radians) - dy * math.sin(radians),
        about[1] + dx * math.sin(radians) + dy * math.cos(radians),
        point[2],
    )
