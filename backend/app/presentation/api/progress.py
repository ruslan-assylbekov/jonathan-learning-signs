"""GET /me/progress (CLAUDE.md 10)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel

from app.application.user_progress import UserProgress
from app.presentation.api.errors import api_error

router = APIRouter(tags=["progress"])


class ProgressOut(BaseModel):
    letter_id: int
    symbol: str
    attempts_count: int
    success_rate: float
    mastery_score: float
    last_practiced_at: datetime


def current_user(request: Request, authorization: str | None) -> int:
    """The caller's user id, from a bearer token."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    user_id = request.app.state.practice_deps.tokens.user_id(token) if token else None
    if user_id is None:
        raise api_error(401, "UNAUTHENTICATED", "a bearer token is required")
    return int(user_id)


@router.get("/me/progress")
async def my_progress(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> list[ProgressOut]:
    """Per-letter standing, most recently practised first."""
    user_progress: UserProgress = request.app.state.user_progress
    rows = await user_progress.for_user(current_user(request, authorization))
    return [
        ProgressOut(
            letter_id=row.letter_id,
            symbol=row.symbol,
            attempts_count=row.standing.attempts_count,
            success_rate=round(row.standing.success_rate, 4),
            mastery_score=round(row.standing.mastery_score, 4),
            last_practiced_at=row.standing.last_practiced_at,
        )
        for row in rows
    ]
