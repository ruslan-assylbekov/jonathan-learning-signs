"""Registering and signing in (CLAUDE.md 10)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.auth.passwords import hash_password, verify_password
from app.infrastructure.auth.tokens import JwtTokens
from app.infrastructure.db import models


class EmailTakenError(ValueError):
    pass


class BadCredentialsError(ValueError):
    """Wrong email or wrong password -- deliberately not distinguished."""


@dataclass(frozen=True, slots=True)
class Session:
    user_id: int
    token: str


class Accounts:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], tokens: JwtTokens) -> None:
        self._sessions = sessions
        self._tokens = tokens

    async def register(self, email: str, password: str) -> Session:
        """Create an account.

        Raises:
            WeakPasswordError: from hashing, before anything is written.
            EmailTakenError: the unique index rejected it.
        """
        normalized = email.strip().lower()
        user = models.User(email=normalized, password_hash=hash_password(password))

        async with self._sessions() as session:
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise EmailTakenError(f"{normalized} is already registered") from error
            await session.refresh(user)

        return Session(user_id=user.id, token=self._tokens.issue(user.id))

    async def login(self, email: str, password: str) -> Session:
        """Exchange credentials for a token.

        Raises:
            BadCredentialsError: for an unknown email or a wrong password.
        """
        normalized = email.strip().lower()
        async with self._sessions() as session:
            user = (
                await session.execute(select(models.User).where(models.User.email == normalized))
            ).scalar_one_or_none()

        # Hash a throwaway password when the account does not exist, so that
        # answering an unknown email takes about as long as answering a known
        # one and timing does not reveal which addresses are registered.
        stored = user.password_hash if user else _DUMMY_HASH
        if not verify_password(password, stored) or user is None:
            raise BadCredentialsError("email or password is wrong")

        return Session(user_id=user.id, token=self._tokens.issue(user.id))


#: A real bcrypt hash of a value nobody can supply, for the timing guard above.
_DUMMY_HASH = hash_password("not-a-real-password-placeholder")
