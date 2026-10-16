"""Database-backed implementations of the domain ports (CLAUDE.md 5)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.entities import Letter
from app.domain.gesture.feedback import LetterTemplate
from app.infrastructure.db import models

DEFAULT_ALPHABET = "rsl"


class DbLetterCatalog:
    """Implements ``LetterCatalog`` over the ``letters`` table.

    Only static letters are offered: the dynamic ones are out of scope for the
    MVP (CLAUDE.md 8.1), and a letter whose staticness is still unknown
    (``is_static IS NULL``) is withheld rather than presented as practisable.
    """

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], *, alphabet: str = DEFAULT_ALPHABET
    ) -> None:
        self._sessions = sessions
        self._alphabet = alphabet

    async def all(self) -> list[Letter]:
        query = (
            select(models.Letter)
            .where(
                models.Letter.alphabet == self._alphabet,
                models.Letter.is_static.is_(True),
            )
            .order_by(models.Letter.sort_order)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_to_letter(row) for row in rows]

    async def by_id(self, letter_id: int) -> Letter | None:
        async with self._sessions() as session:
            row = await session.get(models.Letter, letter_id)
        return _to_letter(row) if row is not None else None


class DbTemplateSource:
    """Implements ``TemplateSource`` over ``letters.template_angles``.

    Returns None for a letter whose template has not been built yet, so the
    session runs without hints instead of comparing against nothing.
    """

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], *, alphabet: str = DEFAULT_ALPHABET
    ) -> None:
        self._sessions = sessions
        self._alphabet = alphabet

    async def template_for(self, label: str) -> LetterTemplate | None:
        query = select(models.Letter).where(
            models.Letter.alphabet == self._alphabet,
            models.Letter.symbol == label,
        )
        async with self._sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()

        # Angles and tolerances are built together by build_templates.py; one
        # without the other is a half-applied template, treated as none.
        if row is None or not row.template_angles or not row.finger_tolerances:
            return None

        # A malformed template is a seeding bug: LetterTemplate raises, and the
        # caller must not paper over it by silently dropping hints.
        return LetterTemplate(
            symbol=row.symbol,
            angles=row.template_angles,
            tolerances=row.finger_tolerances,
        )


def _to_letter(row: models.Letter) -> Letter:
    return Letter(id=row.id, symbol=row.symbol, sort_order=row.sort_order)
