"""Registering and signing in (CLAUDE.md 10).

Depends only on ports (CLAUDE.md 5): storage, hashing and tokens arrive from
main.py, which is why these can be tested with in-memory fakes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final

from app.domain.ports import EmailTakenError, PasswordHasher, TokenIssuer, UserRepository

__all__ = ["Accounts", "BadCredentialsError", "EmailTakenError", "Session", "WeakPasswordError"]

MIN_PASSWORD_LENGTH: Final = 8
MAX_PASSWORD_BYTES: Final = 72
"""bcrypt reads only the first 72 bytes, so a longer password would make every
character past the limit meaningless. Refused rather than silently truncated.
Bytes, not characters: 40 Cyrillic letters are 80 bytes."""


class WeakPasswordError(ValueError):
    pass


class BadCredentialsError(ValueError):
    """Wrong email or wrong password -- deliberately not distinguished."""


@dataclass(frozen=True, slots=True)
class Session:
    user_id: int
    token: str


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPasswordError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise WeakPasswordError(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes "
            "(Cyrillic letters take two bytes each)"
        )


class Accounts:
    def __init__(self, users: UserRepository, hasher: PasswordHasher, tokens: TokenIssuer) -> None:
        self._users = users
        self._hasher = hasher
        self._tokens = tokens
        # A real hash of a value nobody can supply, for the timing guard in
        # login. Computed once, at startup, where blocking costs nothing.
        self._dummy_hash = hasher.hash("not-a-real-password-placeholder")

    async def register(self, email: str, password: str) -> Session:
        """Create an account.

        Raises:
            WeakPasswordError: before anything is hashed or written.
            EmailTakenError: the address is already registered.
        """
        check_password(password)
        # bcrypt is ~0.25 s of pure CPU. Run on the event loop it would freeze
        # every open practice socket for each registration.
        password_hash = await asyncio.to_thread(self._hasher.hash, password)
        user_id = await self._users.create(_normalize(email), password_hash)
        return Session(user_id=user_id, token=self._tokens.issue(user_id))

    async def login(self, email: str, password: str) -> Session:
        """Exchange credentials for a token.

        Raises:
            BadCredentialsError: for an unknown email or a wrong password.
        """
        account = await self._users.find_by_email(_normalize(email))

        # Check against a throwaway hash when the account does not exist, so an
        # unknown email takes as long to refuse as a wrong password and timing
        # does not reveal which addresses are registered.
        stored = account.password_hash if account else self._dummy_hash
        matches = await asyncio.to_thread(self._hasher.verify, password, stored)
        if account is None or not matches:
            raise BadCredentialsError("email or password is wrong")

        return Session(user_id=account.id, token=self._tokens.issue(account.id))


def _normalize(email: str) -> str:
    return email.strip().lower()
