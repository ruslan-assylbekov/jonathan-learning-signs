"""Letter catalog and template source against a real Postgres (CLAUDE.md 13).

Each test works inside its own throwaway ``alphabet`` value rather than the
seeded ``rsl`` rows, so a test run cannot corrupt real data and cleanup is one
delete. Skipped, loudly, when no database is reachable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.exc import SQLAlchemyError

from app.domain.gesture.features import FINGERS, JOINTS
from app.infrastructure.db import models
from app.infrastructure.db.repositories import (
    FALLBACK_TOLERANCE_DEG,
    DbLetterCatalog,
    DbTemplateSource,
)
from app.infrastructure.db.session import session_factory

pytestmark = pytest.mark.asyncio

TEMPLATE: dict[str, Any] = {finger: dict.fromkeys(JOINTS, 10.0) for finger in FINGERS}


@pytest.fixture
async def alphabet() -> AsyncIterator[str]:
    """An alphabet name nothing else uses, dropped afterwards."""
    name = f"t{uuid4().hex[:12]}"
    try:
        async with session_factory() as session:
            await session.execute(delete(models.Letter).where(models.Letter.alphabet == name))
            await session.commit()
    except SQLAlchemyError as error:
        pytest.skip(f"no database reachable: {error}")

    yield name

    async with session_factory() as session:
        await session.execute(delete(models.Letter).where(models.Letter.alphabet == name))
        await session.commit()


async def add_letter(alphabet: str, **columns: Any) -> int:
    async with session_factory() as session:
        letter_id = (
            await session.execute(
                insert(models.Letter)
                .values(alphabet=alphabet, **columns)
                .returning(models.Letter.id)
            )
        ).scalar_one()
        await session.commit()
    return int(letter_id)


async def test_only_static_letters_are_offered(alphabet: str) -> None:
    """A dynamic letter is out of MVP scope, and an undecided one is not yet
    knowingly practisable (CLAUDE.md 8.1)."""
    await add_letter(alphabet, symbol="S", is_static=True, sort_order=1)
    await add_letter(alphabet, symbol="D", is_static=False, sort_order=2)
    await add_letter(alphabet, symbol="U", is_static=None, sort_order=3)

    offered = await DbLetterCatalog(session_factory, alphabet=alphabet).all()

    assert [letter.symbol for letter in offered] == ["S"]


async def test_letters_come_back_in_sort_order(alphabet: str) -> None:
    await add_letter(alphabet, symbol="C", is_static=True, sort_order=3)
    await add_letter(alphabet, symbol="A", is_static=True, sort_order=1)
    await add_letter(alphabet, symbol="B", is_static=True, sort_order=2)

    offered = await DbLetterCatalog(session_factory, alphabet=alphabet).all()

    assert [letter.symbol for letter in offered] == ["A", "B", "C"]


async def test_a_letter_is_found_by_id(alphabet: str) -> None:
    letter_id = await add_letter(alphabet, symbol="A", is_static=True, sort_order=1)

    found = await DbLetterCatalog(session_factory, alphabet=alphabet).by_id(letter_id)

    assert found is not None
    assert (found.id, found.symbol) == (letter_id, "A")


async def test_an_unknown_id_is_none_not_an_error(alphabet: str) -> None:
    assert await DbLetterCatalog(session_factory, alphabet=alphabet).by_id(-1) is None


async def test_a_template_is_read_back_whole(alphabet: str) -> None:
    await add_letter(
        alphabet, symbol="A", is_static=True, template_angles=TEMPLATE, tolerance_deg=15.0
    )

    template = await DbTemplateSource(session_factory, alphabet=alphabet).template_for("A")

    assert template is not None
    assert template.tolerance_deg == 15.0
    assert template.angles["index"]["pip"] == 10.0


async def test_a_template_without_a_tolerance_gets_the_fallback(alphabet: str) -> None:
    await add_letter(alphabet, symbol="A", is_static=True, template_angles=TEMPLATE)

    template = await DbTemplateSource(session_factory, alphabet=alphabet).template_for("A")

    assert template is not None
    assert template.tolerance_deg == FALLBACK_TOLERANCE_DEG


async def test_a_letter_without_a_template_yields_none(alphabet: str) -> None:
    """The seeded letters are in this state until build_templates.py runs, and
    they must be practisable without hints rather than refused."""
    await add_letter(alphabet, symbol="A", is_static=True, template_angles=None)

    assert await DbTemplateSource(session_factory, alphabet=alphabet).template_for("A") is None


async def test_an_unknown_symbol_yields_none(alphabet: str) -> None:
    assert await DbTemplateSource(session_factory, alphabet=alphabet).template_for("?") is None


async def test_a_malformed_template_is_not_silently_ignored(alphabet: str) -> None:
    """A seeding bug must surface, not quietly disable hints for that letter."""
    incomplete = {finger: dict.fromkeys(JOINTS, 10.0) for finger in FINGERS if finger != "pinky"}
    await add_letter(alphabet, symbol="A", is_static=True, template_angles=incomplete)

    with pytest.raises(ValueError, match="pinky"):
        await DbTemplateSource(session_factory, alphabet=alphabet).template_for("A")
