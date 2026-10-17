"""Accounts use case with in-memory fakes (CLAUDE.md 13: use cases with fake
repositories). No database, no bcrypt."""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.application.accounts import (
    Accounts,
    BadCredentialsError,
    EmailTakenError,
    WeakPasswordError,
)
from app.domain.entities import Account


class FakeUsers:
    def __init__(self) -> None:
        self.rows: dict[str, Account] = {}

    async def create(self, email: str, password_hash: str) -> int:
        if email in self.rows:
            raise EmailTakenError(email)
        account = Account(id=len(self.rows) + 1, email=email, password_hash=password_hash)
        self.rows[email] = account
        return account.id

    async def find_by_email(self, email: str) -> Account | None:
        return self.rows.get(email)


class FakeHasher:
    """Reversible, and records which thread it ran on."""

    def __init__(self) -> None:
        self.verified: list[str] = []
        self.threads: set[int] = set()

    def hash(self, password: str) -> str:
        self.threads.add(threading.get_ident())
        return f"hashed:{password}"

    def verify(self, password: str, password_hash: str) -> bool:
        self.threads.add(threading.get_ident())
        self.verified.append(password_hash)
        return password_hash == f"hashed:{password}"


class FakeTokens:
    def issue(self, user_id: int) -> str:
        return f"token-for-{user_id}"


def make() -> tuple[Accounts, FakeUsers, FakeHasher]:
    users, hasher = FakeUsers(), FakeHasher()
    return Accounts(users, hasher, FakeTokens()), users, hasher


def test_register_stores_a_normalised_email_and_returns_a_token() -> None:
    accounts, users, _ = make()

    session = asyncio.run(accounts.register("  Ruslan@Example.COM ", "correct horse"))

    assert session.token == "token-for-1"
    assert set(users.rows) == {"ruslan@example.com"}
    assert users.rows["ruslan@example.com"].password_hash == "hashed:correct horse"


def test_register_refuses_a_taken_email() -> None:
    accounts, _, _ = make()
    asyncio.run(accounts.register("a@example.com", "correct horse"))

    with pytest.raises(EmailTakenError):
        asyncio.run(accounts.register("A@example.com", "another password"))


@pytest.mark.parametrize(
    "password", ["short", "ж" * 40], ids=["under-8-characters", "over-72-bytes"]
)
def test_register_refuses_a_weak_password_before_hashing(password: str) -> None:
    accounts, users, hasher = make()
    hasher.threads.clear()

    with pytest.raises(WeakPasswordError):
        asyncio.run(accounts.register("a@example.com", password))
    assert users.rows == {}
    assert not hasher.threads


def test_login_with_the_right_password() -> None:
    accounts, _, _ = make()
    registered = asyncio.run(accounts.register("a@example.com", "correct horse"))

    session = asyncio.run(accounts.login("A@Example.com", "correct horse"))

    assert session.user_id == registered.user_id


def test_login_refuses_a_wrong_password() -> None:
    accounts, _, _ = make()
    asyncio.run(accounts.register("a@example.com", "correct horse"))

    with pytest.raises(BadCredentialsError):
        asyncio.run(accounts.login("a@example.com", "wrong password"))


def test_an_unknown_email_still_costs_a_hash_check() -> None:
    """The timing guard: refusing an unknown address must do the same work as
    refusing a wrong password, or response time reveals who is registered."""
    accounts, _, hasher = make()

    with pytest.raises(BadCredentialsError):
        asyncio.run(accounts.login("nobody@example.com", "whatever pass"))
    assert hasher.verified, "no hash was checked for the unknown address"


def test_hashing_runs_off_the_event_loop() -> None:
    """bcrypt is ~0.25 s of CPU; on the loop it would freeze every socket."""
    accounts, _, hasher = make()
    loop_thread = threading.get_ident()
    hasher.threads.clear()

    asyncio.run(accounts.register("a@example.com", "correct horse"))
    asyncio.run(accounts.login("a@example.com", "correct horse"))

    assert hasher.threads
    assert loop_thread not in hasher.threads
