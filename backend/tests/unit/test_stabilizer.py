"""Hold logic (CLAUDE.md 6.4)."""

from __future__ import annotations

import pytest

from app.domain.gesture.stabilizer import NO_SIGN, Stabilizer

TARGET = "A"


def make(hold_frames: int = 3, threshold: float = 0.7) -> Stabilizer:
    return Stabilizer(TARGET, confidence_threshold=threshold, hold_frames=hold_frames)


def test_counts_consecutive_hits_and_reports_progress() -> None:
    stabilizer = make()

    first = stabilizer.update(TARGET, 0.9)
    second = stabilizer.update(TARGET, 0.9)

    assert (first.correct, first.consecutive_hits) == (True, 1)
    assert first.hold_progress == pytest.approx(1 / 3)
    assert second.hold_progress == pytest.approx(2 / 3)
    assert not first.succeeded and not second.succeeded


def test_succeeds_exactly_once_on_the_last_required_frame() -> None:
    stabilizer = make()

    stabilizer.update(TARGET, 0.9)
    stabilizer.update(TARGET, 0.9)
    third = stabilizer.update(TARGET, 0.9)
    fourth = stabilizer.update(TARGET, 0.9)

    assert third.succeeded
    assert third.hold_progress == pytest.approx(1.0)
    assert fourth.correct
    assert not fourth.succeeded, "success must fire once, not on every later frame"


def test_a_wrong_letter_resets_the_counter() -> None:
    stabilizer = make()

    stabilizer.update(TARGET, 0.9)
    stabilizer.update(TARGET, 0.9)
    missed = stabilizer.update("B", 0.9)

    assert not missed.correct
    assert missed.consecutive_hits == 0
    assert missed.hold_progress == pytest.approx(0.0)


def test_low_confidence_counts_as_a_miss() -> None:
    stabilizer = make(threshold=0.7)

    stabilizer.update(TARGET, 0.9)
    missed = stabilizer.update(TARGET, 0.69)

    assert not missed.correct
    assert missed.consecutive_hits == 0


def test_confidence_exactly_at_the_threshold_counts() -> None:
    assert make(threshold=0.7).update(TARGET, 0.7).correct


def test_no_sign_neither_advances_nor_resets() -> None:
    stabilizer = make()
    stabilizer.update(TARGET, 0.9)
    stabilizer.update(TARGET, 0.9)

    absent = stabilizer.update(NO_SIGN, 0.95)
    resumed = stabilizer.update(TARGET, 0.9)

    assert absent.is_no_sign
    assert not absent.correct
    assert absent.consecutive_hits == 2, "the user simply paused; keep the progress"
    assert resumed.succeeded


def test_reset_clears_progress_and_can_change_target() -> None:
    stabilizer = make()
    stabilizer.update(TARGET, 0.9)

    stabilizer.reset("B")
    state = stabilizer.update(TARGET, 0.9)

    assert stabilizer.target == "B"
    assert not state.correct
    assert state.consecutive_hits == 0


def test_reset_lets_a_finished_letter_succeed_again() -> None:
    stabilizer = make(hold_frames=1)
    assert stabilizer.update(TARGET, 0.9).succeeded

    stabilizer.reset()

    assert stabilizer.update(TARGET, 0.9).succeeded


@pytest.mark.parametrize(
    ("hold_frames", "threshold"),
    [(0, 0.7), (-1, 0.7), (3, -0.1), (3, 1.5)],
)
def test_rejects_nonsensical_configuration(hold_frames: int, threshold: float) -> None:
    with pytest.raises(ValueError):
        Stabilizer(TARGET, confidence_threshold=threshold, hold_frames=hold_frames)
