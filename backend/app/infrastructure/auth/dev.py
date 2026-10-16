"""The development-only token verifier (CLAUDE.md 11).

Real verification is :class:`~app.infrastructure.auth.tokens.JwtTokens`; this
is what ``DEV_AUTH`` opts into instead, and nothing else should.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DevTokenVerifier:
    """Accepts any non-empty token. Implements ``TokenVerifier``.

    FOR LOCAL DEVELOPMENT ONLY -- it is not authentication. It exists so the
    camera, overlay and evaluation loop can be exercised without signing in.
    Every caller is the same local account, so attempts and progress still get
    recorded and can be tried out.
    """

    def __init__(self, dev_user_id: int) -> None:
        self._dev_user_id = dev_user_id
        logger.warning(
            "DEV_AUTH is on: the practice socket accepts any non-empty token "
            "as user %s. Never enable this on a deployed instance.",
            dev_user_id,
        )

    def user_id(self, token: str) -> int | None:
        return self._dev_user_id if token.strip() else None
