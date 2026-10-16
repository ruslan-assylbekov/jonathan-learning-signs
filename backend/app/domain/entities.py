"""Domain entities. Plain dataclasses -- no framework imports (CLAUDE.md 5)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.gesture.feedback import Hint


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
