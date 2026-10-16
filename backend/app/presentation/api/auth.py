"""POST /auth/register and /auth/login (CLAUDE.md 10)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from app.application.accounts import Accounts, BadCredentialsError, EmailTakenError
from app.infrastructure.auth.passwords import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    WeakPasswordError,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_BYTES)]


class TokenOut(BaseModel):
    token: str
    user_id: int


def _error(status: int, code: str, detail: str) -> HTTPException:
    """The error shape every endpoint uses (CLAUDE.md 10)."""
    return HTTPException(status_code=status, detail={"error": {"code": code, "detail": detail}})


@router.post("/register", status_code=201)
async def register(credentials: Credentials, request: Request) -> TokenOut:
    accounts: Accounts = request.app.state.accounts
    try:
        session = await accounts.register(credentials.email, credentials.password)
    except EmailTakenError as error:
        raise _error(409, "EMAIL_TAKEN", str(error)) from error
    except WeakPasswordError as error:
        raise _error(422, "WEAK_PASSWORD", str(error)) from error
    return TokenOut(token=session.token, user_id=session.user_id)


@router.post("/login")
async def login(credentials: Credentials, request: Request) -> TokenOut:
    accounts: Accounts = request.app.state.accounts
    try:
        session = await accounts.login(credentials.email, credentials.password)
    except BadCredentialsError as error:
        # One answer for both an unknown address and a wrong password, so this
        # endpoint cannot be used to enumerate who has an account.
        raise _error(401, "BAD_CREDENTIALS", "email or password is wrong") from error
    return TokenOut(token=session.token, user_id=session.user_id)
