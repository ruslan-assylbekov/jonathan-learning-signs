"""Password hashing (CLAUDE.md 10).

bcrypt directly rather than through a wrapper: one algorithm, no configuration
to get wrong. The cost factor is bcrypt's default, which it raises over time.
"""

from __future__ import annotations

import bcrypt

#: bcrypt silently truncates at 72 bytes, so a longer password would make every
#: character past the limit meaningless. Refuse instead of pretending.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


class WeakPasswordError(ValueError):
    pass


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPasswordError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise WeakPasswordError(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes; bcrypt ignores the rest"
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        # A malformed hash in the database must read as "wrong password", not
        # as a crash that distinguishes this account from any other.
        return False
