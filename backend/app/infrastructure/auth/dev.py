"""Token verifiers used until real JWT auth lands (CLAUDE.md 19, M3).

The default is :class:`ClosedTokenVerifier`: it rejects everything, so an
unfinished deployment is shut rather than open. Local development opts in to
:class:`DevTokenVerifier` explicitly through ``DEV_AUTH``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ClosedTokenVerifier:
    """Rejects every token. Implements ``TokenVerifier``.

    Wired whenever ``DEV_AUTH`` is off and real auth does not exist yet, so the
    practice socket cannot be used anonymously by accident.
    """

    def verify(self, token: str) -> bool:
        return False


class DevTokenVerifier:
    """Accepts any non-empty token. Implements ``TokenVerifier``.

    FOR LOCAL DEVELOPMENT ONLY -- it is not authentication. It exists so the
    camera, overlay and evaluation loop can be exercised before the users table
    exists, and it is replaced wholesale by JWT verification in M3.
    """

    def __init__(self) -> None:
        logger.warning(
            "DEV_AUTH is on: the practice socket accepts any non-empty token. "
            "Never enable this on a deployed instance."
        )

    def verify(self, token: str) -> bool:
        return bool(token.strip())
