"""How one attempt changes a learner's standing on a letter.

Pure arithmetic, so it is tested without a database; the repository only
supplies the previous standing and stores the next one, inside the same
transaction as the attempt (CLAUDE.md 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

MASTERY_ALPHA: Final = 0.3
"""Weight of the newest attempt in ``mastery_score``.

Mastery is an exponential moving average rather than a lifetime success rate,
so that practice mode (CLAUDE.md 2, item 7) follows how a learner is doing now:
a letter failed twenty times last week and passed five times today should stop
being drilled. ``success_rate`` keeps the lifetime figure alongside it.
"""


@dataclass(frozen=True, slots=True)
class Standing:
    """A learner's record on one letter."""

    attempts_count: int
    success_rate: float
    mastery_score: float
    last_practiced_at: datetime


def next_standing(previous: Standing | None, *, success: bool, at: datetime) -> Standing:
    """The standing after one more attempt."""
    outcome = 1.0 if success else 0.0
    if previous is None:
        return Standing(
            attempts_count=1, success_rate=outcome, mastery_score=outcome, last_practiced_at=at
        )

    total = previous.attempts_count + 1
    return Standing(
        attempts_count=total,
        success_rate=(previous.success_rate * previous.attempts_count + outcome) / total,
        mastery_score=previous.mastery_score * (1 - MASTERY_ALPHA) + outcome * MASTERY_ALPHA,
        last_practiced_at=at,
    )
