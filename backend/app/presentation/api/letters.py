"""GET /letters (CLAUDE.md 10)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["letters"])


class LetterOut(BaseModel):
    id: int
    symbol: str
    sort_order: int


@router.get("/letters")
async def list_letters(request: Request) -> list[LetterOut]:
    catalog = request.app.state.practice_deps.letters
    return [
        LetterOut(id=letter.id, symbol=letter.symbol, sort_order=letter.sort_order)
        for letter in catalog.all()
    ]


@router.get("/letters/{letter_id}")
async def get_letter(letter_id: int, request: Request) -> LetterOut:
    letter = request.app.state.practice_deps.letters.by_id(letter_id)
    if letter is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "UNKNOWN_LETTER", "detail": f"no letter {letter_id}"}},
        )
    return LetterOut(id=letter.id, symbol=letter.symbol, sort_order=letter.sort_order)
