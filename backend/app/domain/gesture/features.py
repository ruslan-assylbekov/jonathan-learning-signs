"""Landmark normalization and feature extraction.

SINGLE SOURCE OF TRUTH (CLAUDE.md 6.1). No other module -- not ``ml/``, not the
frontend -- may normalize landmarks. The browser sends raw MediaPipe landmarks
and every step below happens here.

Pipeline (CLAUDE.md 6.2):

1. Input: 21 ``(x, y, z)`` landmarks, plus handedness and whether the source
   video was mirrored.
2. Undo mirroring (``x -> 1 - x``) and flip the handedness label with it.
3. Reflect left hands into right hands, so one model covers both.
4. Correct the frame aspect ratio: ``x *= aspect`` and ``z *= aspect``.
5. Translate the wrist (landmark 0) to the origin.
6. Scale by the wrist -> middle-finger-MCP distance (landmark 9).
7. Emit 63 normalized coordinates followed by 15 joint flexion angles in
   degrees (3 per finger, thumb to pinky).

Why step 4 exists: MediaPipe normalizes ``x`` by frame width but ``y`` by frame
height, so on a non-square frame the hand arrives anisotropically stretched and
the flexion angles come out skewed -- the same physical hand would yield
different features at 640x480 and at 1280x720. ``z`` shares the scale of ``x``,
so it is corrected alongside it. Steps 2 and 3 deliberately run first, while
the mirror axis is still ``x = 0.5``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

FEATURE_VERSION: Final = 2
"""Bump on ANY change to the steps above. A bump means: retrain, new artifact.

v2 added the aspect-ratio correction (step 4).
"""

NUM_LANDMARKS: Final = 21
NUM_COORDS: Final = NUM_LANDMARKS * 3
NUM_ANGLES: Final = 15
FEATURE_LENGTH: Final = NUM_COORDS + NUM_ANGLES

WRIST: Final = 0
MIDDLE_MCP: Final = 9

FINGERS: Final = ("thumb", "index", "middle", "ring", "pinky")

#: Landmark chains wrist -> MCP -> PIP -> DIP -> TIP. Flexion is measured at the
#: three middle joints of each chain. The thumb's joints are CMC, MCP and IP;
#: they are still reported under the keys "mcp", "pip", "dip" for uniformity.
FINGER_CHAINS: Final[dict[str, tuple[int, int, int, int, int]]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

JOINTS: Final = ("mcp", "pip", "dip")

_MIN_NORM: Final = 1e-9
_MIRROR_AXIS: Final = 1.0
"""Landmarks are normalized to [0, 1], so a mirror reflects about x = 0.5."""


class Handedness(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class InvalidLandmarksError(ValueError):
    """Raised for malformed or degenerate landmark input.

    The WebSocket handler maps this to the ``INVALID_FRAME`` error code.
    """


def aspect_of(width: int, height: int) -> float:
    """Frame aspect ratio for :func:`extract_features`.

    Call once per stream: the WebSocket session takes ``width``/``height`` from
    the ``start`` message, the ``ml/`` scripts from the dataset annotations.
    """
    if width <= 0 or height <= 0:
        raise InvalidLandmarksError(f"frame size must be positive, got {width}x{height}")
    return width / height


@dataclass(frozen=True, slots=True)
class Features:
    """Everything derived from one frame, computed once.

    ``vector`` feeds the classifier; ``angles`` feeds the feedback engine.
    """

    vector: npt.NDArray[np.float64]
    angles: dict[str, dict[str, float]]
    feature_version: int = FEATURE_VERSION


def extract_features(
    landmarks: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    handedness: Handedness,
    *,
    aspect: float,
    mirrored: bool = False,
) -> Features:
    """Normalize one frame of MediaPipe landmarks into a feature vector.

    Args:
        landmarks: 21 ``(x, y, z)`` triples as MediaPipe reports them.
        handedness: the label MediaPipe assigned to the detected hand. If
            ``mirrored`` is true this is the label for the mirrored image, and
            it is flipped here.
        aspect: frame width divided by frame height, from :func:`aspect_of`.
            Required on purpose -- a wrong aspect yields features that look
            plausible but are not comparable with the training data.
        mirrored: true when the source frame was horizontally flipped, as
            browser preview video usually is.

    Raises:
        InvalidLandmarksError: wrong shape, non-finite values, a non-positive
            aspect, or a hand too degenerate to normalize (zero-length bones).
    """
    points = _as_points(landmarks)
    if not math.isfinite(aspect) or aspect <= 0.0:
        raise InvalidLandmarksError(f"aspect must be a positive number, got {aspect!r}")

    if mirrored:
        points = _reflect_x(points)
        handedness = _flip(handedness)

    if handedness is Handedness.LEFT:
        points = _reflect_x(points)

    points = points * np.array([aspect, 1.0, aspect])

    points = points - points[WRIST]

    scale = float(np.linalg.norm(points[MIDDLE_MCP]))
    if scale < _MIN_NORM:
        raise InvalidLandmarksError("wrist and middle-finger MCP coincide")
    points = points / scale

    angles = {finger: _chain_angles(points, FINGER_CHAINS[finger]) for finger in FINGERS}
    flat_angles = [angles[finger][joint] for finger in FINGERS for joint in JOINTS]

    vector = np.concatenate([points.reshape(-1), np.asarray(flat_angles, dtype=np.float64)])
    return Features(vector=vector, angles=angles)


def _as_points(
    landmarks: Sequence[Sequence[float]] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if len(landmarks) != NUM_LANDMARKS:
        raise InvalidLandmarksError(f"expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
    try:
        points = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidLandmarksError(f"landmarks are not numeric: {exc}") from exc
    if points.shape != (NUM_LANDMARKS, 3):
        raise InvalidLandmarksError(f"expected shape (21, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise InvalidLandmarksError("landmarks contain NaN or infinity")
    return points


def _reflect_x(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    reflected = points.copy()
    reflected[:, 0] = _MIRROR_AXIS - reflected[:, 0]
    return reflected


def _flip(handedness: Handedness) -> Handedness:
    return Handedness.LEFT if handedness is Handedness.RIGHT else Handedness.RIGHT


def _chain_angles(
    points: npt.NDArray[np.float64],
    chain: tuple[int, int, int, int, int],
) -> dict[str, float]:
    return {
        joint: _flexion(points[chain[i]], points[chain[i + 1]], points[chain[i + 2]])
        for i, joint in enumerate(JOINTS)
    }


def _flexion(
    previous: npt.NDArray[np.float64],
    joint: npt.NDArray[np.float64],
    following: npt.NDArray[np.float64],
) -> float:
    """Angle in degrees by which the bone chain turns at ``joint``.

    0 means the two bones are in line (the joint is straight); larger means
    more bent, up to 180.
    """
    incoming = joint - previous
    outgoing = following - joint
    norms = float(np.linalg.norm(incoming)) * float(np.linalg.norm(outgoing))
    if norms < _MIN_NORM:
        raise InvalidLandmarksError("zero-length bone, cannot measure flexion")
    cosine = float(np.dot(incoming, outgoing)) / norms
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
