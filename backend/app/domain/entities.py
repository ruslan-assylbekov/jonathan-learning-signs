"""Domain entities. Plain dataclasses -- no framework imports (CLAUDE.md 5)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.gesture.feedback import Hint
from app.domain.progress import Standing


@dataclass(frozen=True, slots=True)
class Letter:
    """One practice target.

    ``symbol`` is both what the learner is shown and what the classifier emits,
    so there is a single identifier for a target. Display text for it belongs
    in the frontend (``i18n.ts``), which is also where stub labels get readable
    names.
    """

    id: int
    symbol: str
    sort_order: int = 0


@dataclass(frozen=True, slots=True)
class Prediction:
    """What the classifier made of one frame."""

    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything the client is told about one frame (CLAUDE.md 7, ``eval``)."""

    prediction: Prediction
    correct: bool
    hold_progress: float
    succeeded: bool
    hints: list[Hint] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Account:
    """A stored user, as much of one as signing in needs."""

    id: int
    email: str
    password_hash: str


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One finished or abandoned try at a letter (CLAUDE.md 9, ``attempts``)."""

    user_id: int
    letter_id: int
    mode: str
    success: bool
    time_to_success_ms: float | None = None
    avg_confidence: float | None = None
    frames_count: int = 0


@dataclass(frozen=True, slots=True)
class LetterProgress:
    """A learner's standing on one letter, with the letter it is about."""

    letter_id: int
    symbol: str
    standing: Standing
