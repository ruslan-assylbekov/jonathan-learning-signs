"""Synthetic hand geometry for unit tests.

These are *constructed* landmarks, not recordings: each finger is built as a
planar chain with prescribed turn angles, so the angles the feature extractor
reports can be checked against the angles that went in. Recorded samples live
in ``tests/fixtures/`` and are used where realism matters.
"""

from __future__ import annotations

import math

from app.domain.gesture.features import FINGER_CHAINS, FINGERS, JOINTS, NUM_LANDMARKS

Landmarks = list[list[float]]

#: Direction each finger leaves the wrist in, degrees, spread like a real hand.
_BASE_HEADING: dict[str, float] = {
    "thumb": -150.0,
    "index": -105.0,
    "middle": -90.0,
    "ring": -75.0,
    "pinky": -55.0,
}

_WRIST = (0.5, 0.8, 0.0)
_SEGMENT = 0.06

StraightTurns = dict[str, tuple[float, float, float]]


def hand(
    turns: StraightTurns | None = None,
    *,
    wrist: tuple[float, float, float] = _WRIST,
    segment: float = _SEGMENT,
) -> Landmarks:
    """Build 21 landmarks from per-finger joint turn angles in degrees.

    ``turns[finger]`` gives the turn at the MCP, PIP and DIP joints. A finger
    with ``(0, 0, 0)`` is perfectly straight and colinear with the wrist, so
    ``hand()`` is a flat, fully extended hand with all 15 angles at zero.
    """
    turns = turns or {}
    points: Landmarks = [[0.0, 0.0, 0.0] for _ in range(NUM_LANDMARKS)]
    points[0] = list(wrist)

    for finger in FINGERS:
        heading = _BASE_HEADING[finger]
        x, y, z = wrist
        finger_turns = turns.get(finger, (0.0, 0.0, 0.0))
        chain = FINGER_CHAINS[finger]

        for bone in range(4):
            if bone > 0:
                heading += finger_turns[bone - 1]
            x += segment * math.cos(math.radians(heading))
            y += segment * math.sin(math.radians(heading))
            points[chain[bone + 1]] = [x, y, z]

    return points


def mirror(landmarks: Landmarks) -> Landmarks:
    """Horizontally flip landmarks the way a mirrored camera preview does."""
    return [[1.0 - x, y, z] for x, y, z in landmarks]


def flatten_turns(turns: StraightTurns) -> list[float]:
    """The 15 turn angles in the order ``extract_features`` emits them."""
    return [
        turns.get(finger, (0.0, 0.0, 0.0))[index]
        for finger in FINGERS
        for index, _ in enumerate(JOINTS)
    ]
