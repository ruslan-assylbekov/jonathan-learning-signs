"""Hold logic: a letter counts only when held still for several frames.

CLAUDE.md 6.4. A letter is shown correctly only if it is predicted as the
target with confidence at or above the threshold for ``hold_frames``
consecutive frames. Any miss resets the counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

NO_SIGN: Final = "no_sign"
"""Class name for "no hand sign present". Never advances the hold counter."""


@dataclass(frozen=True, slots=True)
class HoldState:
    """Outcome of one frame.

    ``correct`` says this single frame matched; ``succeeded`` says the hold is
    complete, which is the event worth recording as an attempt.
    """

    correct: bool
    consecutive_hits: int
    hold_progress: float
    succeeded: bool
    is_no_sign: bool


class Stabilizer:
    """Counts consecutive correct frames for one target letter.

    Not thread-safe and not shareable: one instance per WebSocket connection,
    reset when the target changes.
    """

    def __init__(self, target: str, *, confidence_threshold: float, hold_frames: int) -> None:
        if hold_frames < 1:
            raise ValueError("hold_frames must be at least 1")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")

        self._target = target
        self._threshold = confidence_threshold
        self._hold_frames = hold_frames
        self._hits = 0
        self._succeeded = False

    @property
    def target(self) -> str:
        return self._target

    def reset(self, target: str | None = None) -> None:
        """Start over, optionally on a new target letter."""
        if target is not None:
            self._target = target
        self._hits = 0
        self._succeeded = False

    def update(self, prediction: str, confidence: float) -> HoldState:
        """Feed one classified frame and report the resulting hold state.

        A ``no_sign`` prediction neither advances nor resets the counter: the
        user is simply not showing anything yet.
        """
        if prediction == NO_SIGN:
            return self._state(correct=False, is_no_sign=True)

        if prediction == self._target and confidence >= self._threshold:
            if not self._succeeded:
                self._hits += 1
                self._succeeded = self._hits >= self._hold_frames
                return self._state(correct=True, is_no_sign=False, just_succeeded=self._succeeded)
            return self._state(correct=True, is_no_sign=False)

        self._hits = 0
        return self._state(correct=False, is_no_sign=False)

    def _state(
        self,
        *,
        correct: bool,
        is_no_sign: bool,
        just_succeeded: bool = False,
    ) -> HoldState:
        return HoldState(
            correct=correct,
            consecutive_hits=self._hits,
            hold_progress=min(1.0, self._hits / self._hold_frames),
            succeeded=just_succeeded,
            is_no_sign=is_no_sign,
        )
