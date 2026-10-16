"""Load hint templates from a JSON artifact into the letters table.

The one place templates reach the database, used both by
``ml/build_templates.py --apply`` on a development machine and on a server
that has neither the dataset nor the ml dependencies:

    python -m app.infrastructure.db.load_templates /srv/ml/artifacts/templates-v2.json

Every template is validated as a LetterTemplate before anything is written,
and all of them are written in one transaction, so a bad file leaves the
database as it was rather than half updated.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.gesture.features import FEATURE_VERSION
from app.domain.gesture.feedback import LetterTemplate
from app.infrastructure.db import models
from app.infrastructure.db.repositories import DEFAULT_ALPHABET


class TemplateFileError(ValueError):
    pass


def read(path: Path) -> dict[str, dict[str, Any]]:
    """Parse and validate a templates file without touching the database."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("feature_version") != FEATURE_VERSION:
        raise TemplateFileError(
            f"{path.name} was built for feature version {payload.get('feature_version')}, "
            f"the code measures angles as version {FEATURE_VERSION}. Rebuild the templates."
        )

    letters: dict[str, dict[str, Any]] = payload["letters"]
    for symbol, entry in letters.items():
        LetterTemplate(symbol=symbol, angles=entry["angles"], tolerances=entry["tolerances"])
    return letters


async def apply(
    letters: dict[str, dict[str, Any]],
    sessions: async_sessionmaker[AsyncSession],
    *,
    alphabet: str = DEFAULT_ALPHABET,
) -> int:
    """Write every template in one transaction; returns how many letters matched."""
    updated = 0
    async with sessions() as session:
        for symbol, entry in letters.items():
            result = await session.execute(
                update(models.Letter)
                .where(models.Letter.alphabet == alphabet, models.Letter.symbol == symbol)
                .values(template_angles=entry["angles"], finger_tolerances=entry["tolerances"])
                .returning(models.Letter.id)
            )
            updated += len(result.all())
        await session.commit()
    return updated


async def _main(path: Path) -> int:
    from app.infrastructure.db.session import engine, session_factory

    letters = read(path)
    updated = await apply(letters, session_factory)
    await engine.dispose()

    print(f"applied {updated} of {len(letters)} templates from {path.name}")
    if updated != len(letters):
        print("  some letters were not found: is the database migrated and seeded?")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m app.infrastructure.db.load_templates TEMPLATES.json")
    raise SystemExit(asyncio.run(_main(Path(sys.argv[1]))))
