"""Registration, sign-in, attempts and progress against real Postgres."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.record_attempt import MASTERY_ALPHA, AttemptRecord, Attempts
from app.config import settings
from app.infrastructure.auth.tokens import InsecureSecretError, JwtTokens
from app.infrastructure.db import models
from app.main import build_app

SECRET = "a-real-secret-for-these-tests"
PASSWORD = "correct horse battery"


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, pooled_sessions: async_sessionmaker[AsyncSession]
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "dev_auth", False)
    monkeypatch.setattr(settings, "jwt_secret", SECRET)
    with TestClient(build_app(pooled_sessions)) as test_client:
        yield test_client


def fresh_email() -> str:
    return f"{uuid4().hex[:12]}@tests.example.com"


def register(client: TestClient, email: str | None = None) -> dict[str, Any]:
    response = client.post(
        "/auth/register", json={"email": email or fresh_email(), "password": PASSWORD}
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


def test_registering_returns_a_usable_token(client: TestClient) -> None:
    body = register(client)

    assert body["user_id"] > 0
    assert JwtTokens(SECRET, ttl_minutes=60).user_id(body["token"]) == body["user_id"]


def test_the_same_email_cannot_register_twice(client: TestClient) -> None:
    email = fresh_email()
    register(client, email)

    again = client.post("/auth/register", json={"email": email, "password": PASSWORD})

    assert again.status_code == 409
    assert again.json()["detail"]["error"]["code"] == "EMAIL_TAKEN"


def test_email_case_and_padding_do_not_create_a_second_account(client: TestClient) -> None:
    email = fresh_email()
    register(client, email)

    again = client.post(
        "/auth/register", json={"email": f"  {email.upper()} ", "password": PASSWORD}
    )

    assert again.status_code == 409


def test_a_short_password_is_refused_before_anything_is_written(client: TestClient) -> None:
    response = client.post("/auth/register", json={"email": fresh_email(), "password": "short"})

    assert response.status_code == 422


def test_login_returns_a_token_for_the_right_password(client: TestClient) -> None:
    email = fresh_email()
    registered = register(client, email)

    response = client.post("/auth/login", json={"email": email, "password": PASSWORD})

    assert response.status_code == 200
    assert response.json()["user_id"] == registered["user_id"]


@pytest.mark.parametrize(
    ("email_known", "password"),
    [(True, "wrong password here"), (False, PASSWORD)],
    ids=["wrong-password", "unknown-email"],
)
def test_bad_credentials_are_indistinguishable(
    client: TestClient, email_known: bool, password: str
) -> None:
    """A wrong password and an unknown address must look the same, or the
    endpoint becomes a way to find out who has an account."""
    email = fresh_email()
    if email_known:
        register(client, email)

    response = client.post("/auth/login", json={"email": email, "password": password})

    assert response.status_code == 401
    assert response.json()["detail"]["error"]["code"] == "BAD_CREDENTIALS"


def test_progress_needs_a_token(client: TestClient) -> None:
    assert client.get("/me/progress").status_code == 401
    assert (
        client.get("/me/progress", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    )


def test_progress_starts_empty_and_follows_attempts(
    client: TestClient, pooled_sessions: async_sessionmaker[AsyncSession]
) -> None:
    body = register(client)
    headers = {"Authorization": f"Bearer {body['token']}"}

    assert client.get("/me/progress", headers=headers).json() == []

    letter_id = _any_letter_id(client, pooled_sessions)
    _record(pooled_sessions, body["user_id"], letter_id, success=True)
    _record(pooled_sessions, body["user_id"], letter_id, success=False)

    rows = client.get("/me/progress", headers=headers).json()

    assert len(rows) == 1
    assert rows[0]["attempts_count"] == 2
    assert rows[0]["success_rate"] == pytest.approx(0.5)
    assert rows[0]["last_practiced_at"] is not None


def test_mastery_follows_recent_attempts_not_the_lifetime_average(
    client: TestClient, pooled_sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Practice mode weights weak letters, so mastery has to move with recent
    performance rather than being anchored by old failures."""
    body = register(client)
    letter_id = _any_letter_id(client, pooled_sessions)

    for _ in range(5):
        _record(pooled_sessions, body["user_id"], letter_id, success=False)
    for _ in range(3):
        _record(pooled_sessions, body["user_id"], letter_id, success=True)

    row = client.get("/me/progress", headers={"Authorization": f"Bearer {body['token']}"}).json()[0]

    expected = 1 - (1 - MASTERY_ALPHA) ** 3
    assert row["mastery_score"] == pytest.approx(expected, abs=1e-4)
    assert row["success_rate"] == pytest.approx(3 / 8)
    assert row["mastery_score"] > row["success_rate"]


def test_an_attempt_and_its_progress_are_written_together(
    client: TestClient, pooled_sessions: async_sessionmaker[AsyncSession]
) -> None:
    body = register(client)
    letter_id = _any_letter_id(client, pooled_sessions)

    _record(pooled_sessions, body["user_id"], letter_id, success=True)

    async def counts() -> tuple[int, int]:
        async with pooled_sessions() as session:
            attempts = (
                (
                    await session.execute(
                        select(models.Attempt).where(models.Attempt.user_id == body["user_id"])
                    )
                )
                .scalars()
                .all()
            )
            progress = (
                (
                    await session.execute(
                        select(models.Progress).where(models.Progress.user_id == body["user_id"])
                    )
                )
                .scalars()
                .all()
            )
        return len(attempts), len(progress)

    assert _run(counts()) == (1, 1)


def test_a_placeholder_jwt_secret_is_refused(
    monkeypatch: pytest.MonkeyPatch, pooled_sessions: async_sessionmaker[AsyncSession]
) -> None:
    """A deployment signing tokens anyone can forge must not start."""
    monkeypatch.setattr(settings, "dev_auth", False)
    monkeypatch.setattr(settings, "jwt_secret", "change-me")

    with pytest.raises(InsecureSecretError):
        build_app(pooled_sessions)


def test_a_token_signed_with_another_secret_is_rejected(client: TestClient) -> None:
    forged = JwtTokens("a-different-secret", ttl_minutes=60).issue(1)

    response = client.get("/me/progress", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401


def test_an_expired_token_is_rejected(client: TestClient) -> None:
    stale = JwtTokens(SECRET, ttl_minutes=-1).issue(1)

    response = client.get("/me/progress", headers={"Authorization": f"Bearer {stale}"})

    assert response.status_code == 401


def _any_letter_id(client: TestClient, sessions: async_sessionmaker[AsyncSession]) -> int:
    """A real letter id, so the attempt's foreign key holds."""

    async def first() -> int:
        async with sessions() as session:
            letter = (await session.execute(select(models.Letter).limit(1))).scalar_one_or_none()
            if letter is None:
                pytest.skip("letters are not seeded; run alembic upgrade head")
            return int(letter.id)

    return int(_run(first()))


def _record(
    sessions: async_sessionmaker[AsyncSession], user_id: int, letter_id: int, *, success: bool
) -> None:
    _run(
        Attempts(sessions).record(
            AttemptRecord(
                user_id=user_id,
                letter_id=letter_id,
                mode="learn",
                success=success,
                time_to_success_ms=1200.0 if success else None,
                avg_confidence=0.9,
                frames_count=12,
            )
        )
    )


def _run(coroutine: Any) -> Any:
    """Drive a coroutine from these synchronous TestClient tests."""
    import asyncio

    return asyncio.run(coroutine)


@pytest.fixture(autouse=True)
def _clean_up(pooled_sessions: async_sessionmaker[AsyncSession]) -> Iterator[None]:
    """Remove the accounts these tests create, and with them their attempts."""
    yield

    async def purge() -> None:
        async with pooled_sessions() as session:
            await session.execute(
                delete(models.User).where(models.User.email.like("%@tests.example.com"))
            )
            await session.commit()

    _run(purge())
