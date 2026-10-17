"""How an attempt changes a learner's standing (domain, no database)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain.progress import MASTERY_ALPHA, Standing, next_standing

T0 = datetime(2026, 10, 1, tzinfo=UTC)
T1 = datetime(2026, 10, 2, tzinfo=UTC)


def test_a_first_attempt_starts_the_record() -> None:
    passed = next_standing(None, success=True, at=T0)
    failed = next_standing(None, success=False, at=T0)

    assert (passed.attempts_count, passed.success_rate, passed.mastery_score) == (1, 1.0, 1.0)
    assert (failed.success_rate, failed.mastery_score) == (0.0, 0.0)
    assert passed.last_practiced_at == T0


def test_success_rate_is_the_lifetime_average() -> None:
    standing = next_standing(None, success=True, at=T0)
    for success in (False, False, True):
        standing = next_standing(standing, success=success, at=T1)

    assert standing.attempts_count == 4
    assert standing.success_rate == pytest.approx(0.5)
    assert standing.last_practiced_at == T1


def test_mastery_weights_recent_attempts() -> None:
    """Five old failures, three new passes: mastery has recovered further than
    the lifetime rate, which is the point of using it to pick drills."""
    standing: Standing | None = None
    for success in [False] * 5 + [True] * 3:
        standing = next_standing(standing, success=success, at=T0)

    assert standing is not None
    assert standing.mastery_score == pytest.approx(1 - (1 - MASTERY_ALPHA) ** 3)
    assert standing.success_rate == pytest.approx(3 / 8)
    assert standing.mastery_score > standing.success_rate


def test_the_previous_standing_is_not_mutated() -> None:
    first = next_standing(None, success=True, at=T0)
    next_standing(first, success=False, at=T1)

    assert first.attempts_count == 1
