"""Password hashing (CLAUDE.md 10).

bcrypt directly rather than through a wrapper: one algorithm, no configuration
to get wrong. The cost factor is bcrypt's default, which it raises over time.
Password policy (length) is the application's concern, in accounts.py.
"""

from __future__ import annotations

import bcrypt


class BcryptHasher:
    """Implements ``PasswordHasher``."""

    def hash(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")

    def verify(self, password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
        except ValueError:
            # A malformed hash, or a password bcrypt refuses (over 72 bytes),
            # must read as "wrong password" -- not as a crash that would tell
            # this case apart from any other.
            return False
