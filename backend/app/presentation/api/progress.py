"""GET /me/progress (CLAUDE.md 10)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from app.infrastructure.db import models

router = APIRouter(tags=["progress"])


class ProgressOut(BaseModel):
    letter_id: int
    symbol: str
    attempts_count: int
    success_rate: float
    mastery_score: float
    last_practiced_at: datetime | None


async def current_user(request: Request, authorization: Annotated[str | None, Header()]) -> int:
    """The caller's user id, from a bearer token."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    user_id = request.app.state.practice_deps.tokens.user_id(token) if token else None
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHENTICATED", "detail": "a bearer token is required"}},
        )
    return int(user_id)


@router.get("/me/progress")
async def my_progress(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> list[ProgressOut]:
    """Per-letter standing, most recently practised first.

    Letters never attempted are absent rather than reported as zeroes: "not
    tried yet" and "tried and failed" are different things to show a learner.
    """
    user_id = await current_user(request, authorization)
    query = (
        select(models.Progress, models.Letter.symbol)
        .join(models.Letter, models.Letter.id == models.Progress.letter_id)
        .where(models.Progress.user_id == user_id)
        .order_by(models.Progress.last_practiced_at.desc().nullslast())
    )

    async with request.app.state.sessions() as session:
        rows = (await session.execute(query)).all()

    return [
        ProgressOut(
            letter_id=progress.letter_id,
            symbol=symbol,
            attempts_count=progress.attempts_count,
            success_rate=round(progress.success_rate, 4),
            mastery_score=round(progress.mastery_score, 4),
            last_practiced_at=progress.last_practiced_at,
        )
        for progress, symbol in rows
    ]
