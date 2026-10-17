"""Recording an attempt and the progress it changes (CLAUDE.md 9)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.entities import AttemptRecord
from app.domain.ports import AttemptRepository
from app.domain.progress import Standing, next_standing

__all__ = ["AttemptRecord", "Attempts"]


class Attempts:
    def __init__(self, repository: AttemptRepository) -> None:
        self._repository = repository

    async def record(self, attempt: AttemptRecord) -> None:
        """Store the attempt and the standing it leads to, both or neither:
        progress that disagrees with the attempts behind it is worse than none."""
        now = datetime.now(UTC)

        def advance(previous: Standing | None) -> Standing:
            return next_standing(previous, success=attempt.success, at=now)

        await self._repository.record(attempt, advance)
