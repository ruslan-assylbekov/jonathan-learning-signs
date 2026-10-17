"""A learner's progress across letters (CLAUDE.md 10, GET /me/progress)."""

from __future__ import annotations

from app.domain.entities import LetterProgress
from app.domain.ports import ProgressRepository


class UserProgress:
    def __init__(self, repository: ProgressRepository) -> None:
        self._repository = repository

    async def for_user(self, user_id: int) -> list[LetterProgress]:
        """Letters never attempted are absent rather than zeroes: "not tried
        yet" and "tried and failed" are different things to show a learner."""
        return await self._repository.for_user(user_id)
