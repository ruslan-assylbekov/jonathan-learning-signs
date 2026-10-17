"""JWT issuing and verification (CLAUDE.md 10, 11)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import jwt

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
INSECURE_SECRETS = {"change-me", "secret", ""}
MIN_SECRET_LENGTH = 32
"""HS256 is only as strong as its key; RFC 7518 asks for at least 256 bits.
deploy/.env.example generates 64 hex characters."""


class InsecureSecretError(RuntimeError):
    pass


class JwtTokens:
    """Issues and verifies access tokens. Implements ``TokenIssuer`` and ``TokenVerifier``.

    ``user_id`` is carried in ``sub`` as a string, because the JWT spec says
    ``sub`` is a string and some libraries reject an integer outright.
    """

    def __init__(
        self, secret: str, *, ttl_minutes: int, allow_insecure_secret: bool = False
    ) -> None:
        if secret in INSECURE_SECRETS or len(secret) < MIN_SECRET_LENGTH:
            if not allow_insecure_secret:
                raise InsecureSecretError(
                    f"JWT_SECRET is a placeholder or shorter than {MIN_SECRET_LENGTH} "
                    "characters. Generate one (openssl rand -hex 32) before issuing "
                    "tokens anyone could forge."
                )
            logger.warning("JWT_SECRET is weak or a placeholder: tokens are forgeable.")
        self._secret = secret
        self._ttl = timedelta(minutes=ttl_minutes)

    def issue(self, user_id: int) -> str:
        now = datetime.now(UTC)
        return jwt.encode(
            {"sub": str(user_id), "iat": now, "exp": now + self._ttl},
            self._secret,
            algorithm=ALGORITHM,
        )

    def user_id(self, token: str) -> int | None:
        """The user the token names, or None if it is not a token we issued.

        Every failure -- bad signature, expired, malformed, unexpected
        algorithm -- is one answer, so a caller cannot accidentally tell them
        apart for an attacker's benefit.
        """
        try:
            payload = jwt.decode(token, self._secret, algorithms=[ALGORITHM])
            return int(payload["sub"])
        except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
            return None
