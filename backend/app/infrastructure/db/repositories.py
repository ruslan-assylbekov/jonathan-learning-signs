"""Database-backed implementations of the domain ports (CLAUDE.md 5)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from sqlalchemy import ColumnElement, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.entities import Account, AttemptRecord, Letter, LetterProgress
from app.domain.gesture.feedback import LetterTemplate
from app.domain.ports import EmailTakenError
from app.domain.progress import Standing
from app.infrastructure.db import models

DEFAULT_ALPHABET = "rsl"


class DbLetterCatalog:
    """Implements ``LetterCatalog`` over the ``letters`` table.

    Offers a letter only if it is static -- dynamic ones are out of MVP scope
    and an undecided one (``is_static IS NULL``) is withheld (CLAUDE.md 8.1) --
    and, when ``classes`` is given, only if the model can predict it. Without
    that second filter a target outside the model's classes could be started
    and never once be recognised. ``by_id`` applies the same rules as ``all``,
    so a hidden letter cannot be reached by guessing its id.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        alphabet: str = DEFAULT_ALPHABET,
        classes: Iterable[str] | None = None,
    ) -> None:
        self._sessions = sessions
        self._alphabet = alphabet
        self._classes = frozenset(classes) if classes is not None else None

    def _offered(self) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = [
            models.Letter.alphabet == self._alphabet,
            models.Letter.is_static.is_(True),
        ]
        if self._classes is not None:
            conditions.append(models.Letter.symbol.in_(self._classes))
        return conditions

    async def all(self) -> list[Letter]:
        query = select(models.Letter).where(*self._offered()).order_by(models.Letter.sort_order)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_to_letter(row) for row in rows]

    async def by_id(self, letter_id: int) -> Letter | None:
        query = select(models.Letter).where(models.Letter.id == letter_id, *self._offered())
        async with self._sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
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


class DbUserRepository:
    """Implements ``UserRepository``."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, email: str, password_hash: str) -> int:
        user = models.User(email=email, password_hash=password_hash)
        async with self._sessions() as session:
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise EmailTakenError(f"{email} is already registered") from error
            await session.refresh(user)
        return int(user.id)

    async def find_by_email(self, email: str) -> Account | None:
        query = select(models.User).where(models.User.email == email)
        async with self._sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
        if row is None:
            return None
        return Account(id=row.id, email=row.email, password_hash=row.password_hash)


class DbAttemptRepository:
    """Implements ``AttemptRepository``: attempt and progress in one transaction."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(
        self,
        attempt: AttemptRecord,
        advance: Callable[[Standing | None], Standing],
    ) -> None:
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

            # Locked for the life of the transaction, so two connections
            # finishing the same letter at once cannot both read the old counts
            # and each write back a total that forgets the other.
            row = (
                await session.execute(
                    select(models.Progress)
                    .where(
                        models.Progress.user_id == attempt.user_id,
                        models.Progress.letter_id == attempt.letter_id,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()

            previous = _to_standing(row) if row is not None else None
            standing = advance(previous)

            if row is None:
                session.add(
                    models.Progress(
                        user_id=attempt.user_id,
                        letter_id=attempt.letter_id,
                        attempts_count=standing.attempts_count,
                        success_rate=standing.success_rate,
                        mastery_score=standing.mastery_score,
                        last_practiced_at=standing.last_practiced_at,
                    )
                )
            else:
                row.attempts_count = standing.attempts_count
                row.success_rate = standing.success_rate
                row.mastery_score = standing.mastery_score
                row.last_practiced_at = standing.last_practiced_at

            await session.commit()


class DbProgressRepository:
    """Implements ``ProgressRepository``."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def for_user(self, user_id: int) -> list[LetterProgress]:
        query = (
            select(models.Progress, models.Letter.symbol)
            .join(models.Letter, models.Letter.id == models.Progress.letter_id)
            .where(models.Progress.user_id == user_id)
            .order_by(models.Progress.last_practiced_at.desc().nullslast())
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        return [
            LetterProgress(
                letter_id=progress.letter_id, symbol=symbol, standing=_to_standing(progress)
            )
            for progress, symbol in rows
        ]


def _to_letter(row: models.Letter) -> Letter:
    return Letter(id=row.id, symbol=row.symbol, sort_order=row.sort_order)


def _to_standing(row: models.Progress) -> Standing:
    return Standing(
        attempts_count=row.attempts_count,
        success_rate=row.success_rate,
        mastery_score=row.mastery_score,
        last_practiced_at=row.last_practiced_at,  # type: ignore[arg-type]
    )
