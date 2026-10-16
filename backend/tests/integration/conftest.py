"""Shared database plumbing for the integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings


@pytest.fixture
def pooled_sessions() -> Iterator[async_sessionmaker[AsyncSession]]:
    """A session factory safe to use from TestClient's own event loop.

    TestClient runs the app in a loop of its own, while the rest of the suite
    shares pytest-asyncio's. A pooled connection opened in one loop and reused
    from the other fails on teardown with "Event loop is closed", so these
    tests get NullPool: every checkout opens and closes its own connection,
    inside whichever loop is running.
    """
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    yield async_sessionmaker(engine, expire_on_commit=False)
