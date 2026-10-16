"""Recording an attempt and the progress it changes (CLAUDE.md 9)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.db import models

MASTERY_ALPHA = 0.3
"""Weight of the newest attempt in ``mastery_score``.

Mastery is an exponential moving average rather than a lifetime success rate,
so that practice mode (CLAUDE.md 2, item 7) follows how a learner is doing now:
a letter failed twenty times last week and passed five times today should stop
being drilled.
"""


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    user_id: int
    letter_id: int
    mode: str
    success: bool
    time_to_success_ms: float | None = None
    avg_confidence: float | None = None
    frames_count: int = 0


class Attempts:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, attempt: AttemptRecord) -> None:
        """Insert the attempt and update progress in one transaction.

        Both or neither: progress that disagrees with the attempts it was
        derived from is worse than no progress at all (CLAUDE.md 9).
        """
        async with self._sessions() as session:
            session.add(
                models.Attempt(
                    user_id=attempt.user_id,
                    letter_id=attempt.letter_id,
                    mode=attempt.mode,
                    success=attempt.success,
                    time_to_success_ms=attempt.time_to_success_ms,
                    avg_confidence=attempt.avg_confidence,
                    frames_count=attempt.frames_count,
                )
            )
            await self._apply_progress(session, attempt)
            await session.commit()

    async def _apply_progress(self, session: AsyncSession, attempt: AttemptRecord) -> None:
        # Locked for the life of the transaction, so two connections finishing
        # the same letter at once cannot both read the old counts and each
        # write back a total that forgets the other.
        existing = (
            await session.execute(
                select(models.Progress)
                .where(
                    models.Progress.user_id == attempt.user_id,
                    models.Progress.letter_id == attempt.letter_id,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()

        outcome = 1.0 if attempt.success else 0.0
        now = datetime.now(UTC)

        if existing is None:
            session.add(
                models.Progress(
                    user_id=attempt.user_id,
                    letter_id=attempt.letter_id,
                    attempts_count=1,
                    success_rate=outcome,
                    mastery_score=outcome,
                    last_practiced_at=now,
                )
            )
            return

        total = existing.attempts_count + 1
        existing.success_rate = (existing.success_rate * existing.attempts_count + outcome) / total
        existing.mastery_score = (
            existing.mastery_score * (1 - MASTERY_ALPHA) + outcome * MASTERY_ALPHA
        )
        existing.attempts_count = total
        existing.last_practiced_at = now
