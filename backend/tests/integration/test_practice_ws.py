"""The /ws/practice conversation (CLAUDE.md 7).

Runs against the stub classifier, so these pin the protocol -- message order,
error codes, echoed fields, frame budget -- not recognition quality.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from app.config import settings
from app.infrastructure.ml.stub import STUB_TARGETS
from app.main import build_app
from tests import synthetic

OPEN_HAND_ID = STUB_TARGETS.index("demo:11111")
FIST_ID = STUB_TARGETS.index("demo:00000")

FULLY_BENT = dict.fromkeys(("thumb", "index", "middle", "ring", "pinky"), (80.0, 80.0, 80.0))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "dev_auth", True)
    with TestClient(build_app()) as test_client:
        yield test_client


@pytest.fixture
def socket(client: TestClient) -> Iterator[WebSocketTestSession]:
    with client.websocket_connect("/ws/practice") as session:
        yield session


def frame(landmarks: list[list[float]] | None = None, *, seq: int = 1) -> dict[str, Any]:
    return {
        "type": "frame",
        "seq": seq,
        "t_client": 1712345678.123,
        "handedness": "Right",
        "mirrored": False,
        "landmarks": landmarks if landmarks is not None else synthetic.hand(),
    }


def start(letter_id: int = OPEN_HAND_ID) -> dict[str, Any]:
    return {
        "type": "start",
        "mode": "learn",
        "letter_id": letter_id,
        "width": 1280,
        "height": 720,
    }


def authenticate(socket: WebSocketTestSession) -> None:
    socket.send_json({"type": "auth", "token": "dev"})


def test_frames_are_refused_before_authentication(socket: WebSocketTestSession) -> None:
    socket.send_json(frame())

    assert socket.receive_json()["code"] == "UNAUTHENTICATED"


def test_an_empty_token_is_rejected(socket: WebSocketTestSession) -> None:
    socket.send_json({"type": "auth", "token": "  "})

    assert socket.receive_json()["code"] == "UNAUTHENTICATED"


def test_a_good_token_is_acknowledged_silently(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start())
    socket.send_json(frame())

    assert socket.receive_json()["type"] == "eval", "auth should not emit a message of its own"


def test_the_socket_is_closed_when_dev_auth_is_off(client: TestClient) -> None:
    """Default configuration must not leave practice open to anyone (M3)."""
    settings.dev_auth = False
    with TestClient(build_app()) as closed:
        with closed.websocket_connect("/ws/practice") as session:
            session.send_json({"type": "auth", "token": "anything"})

            assert session.receive_json()["code"] == "UNAUTHENTICATED"


def test_frames_are_refused_before_start(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(frame())

    assert socket.receive_json()["code"] == "NOT_STARTED"


def test_an_unknown_letter_is_reported(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start(letter_id=9999))

    assert socket.receive_json()["code"] == "UNKNOWN_LETTER"


def test_stop_ends_the_session(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start())
    socket.send_json({"type": "stop"})
    socket.send_json(frame())

    assert socket.receive_json()["code"] == "NOT_STARTED"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        pytest.param({"type": "nonsense"}, "unknown type", id="unknown-type"),
        pytest.param({"type": "frame"}, "missing fields", id="incomplete"),
        pytest.param({**start(), "width": 0}, "frame size must be positive", id="zero-width"),
    ],
)
def test_malformed_messages_are_reported(
    socket: WebSocketTestSession, payload: dict[str, Any], reason: str
) -> None:
    authenticate(socket)
    socket.send_json(payload)

    assert socket.receive_json()["code"] == "INVALID_MESSAGE", reason


def test_a_frame_with_the_wrong_landmark_count_is_rejected(
    socket: WebSocketTestSession,
) -> None:
    authenticate(socket)
    socket.send_json(start())
    socket.send_json(frame(synthetic.hand()[:20]))

    answer = socket.receive_json()
    assert answer["code"] == "INVALID_MESSAGE"
    assert "landmarks" in answer["detail"]


def test_a_degenerate_hand_is_reported_as_an_invalid_frame(
    socket: WebSocketTestSession,
) -> None:
    """Shape is fine, geometry is not -- the domain catches this one."""
    authenticate(socket)
    socket.send_json(start())
    socket.send_json(frame([[0.5, 0.5, 0.0] for _ in range(21)]))

    assert socket.receive_json()["code"] == "INVALID_FRAME"


def test_an_invalid_frame_does_not_end_the_session(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start())
    socket.send_json(frame([[0.5, 0.5, 0.0] for _ in range(21)]))
    socket.receive_json()

    socket.send_json(frame(seq=2))

    assert socket.receive_json()["type"] == "eval"


def test_eval_echoes_the_fields_used_for_latency(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start())
    socket.send_json(frame(seq=77))

    answer = socket.receive_json()
    assert answer["seq"] == 77
    assert answer["t_client"] == 1712345678.123
    assert answer["server_ms"] >= 0.0


def test_an_extended_hand_matches_the_open_hand_target(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start(OPEN_HAND_ID))
    socket.send_json(frame())

    answer = socket.receive_json()
    assert answer["pred"] == "demo:11111"
    assert answer["correct"] is True
    assert answer["hints"] == []


def test_a_wrong_hand_earns_hints(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start(OPEN_HAND_ID))
    socket.send_json(frame(synthetic.hand(FULLY_BENT)))

    answer = socket.receive_json()
    assert answer["correct"] is False
    assert answer["hints"], "a closed hand against an open-hand target should be advised"
    assert all(hint["code"] == "SHOULD_BE_STRAIGHT" for hint in answer["hints"])
    assert len(answer["hints"]) <= 2


def test_holding_the_target_reports_progress_then_success(
    socket: WebSocketTestSession,
) -> None:
    authenticate(socket)
    socket.send_json(start(OPEN_HAND_ID))

    progress = []
    for seq in range(settings.hold_frames):
        socket.send_json(frame(seq=seq))
        progress.append(socket.receive_json()["hold_progress"])

    assert progress == sorted(progress), "progress must not go backwards"
    assert progress[-1] == pytest.approx(1.0)

    success = socket.receive_json()
    assert success["type"] == "success"
    assert success["letter_id"] == OPEN_HAND_ID
    assert success["time_to_success_ms"] > 0


def test_the_wrong_target_never_succeeds(socket: WebSocketTestSession) -> None:
    authenticate(socket)
    socket.send_json(start(FIST_ID))

    for seq in range(settings.hold_frames):
        socket.send_json(frame(seq=seq))
        answer = socket.receive_json()
        assert answer["type"] == "eval"
        assert answer["correct"] is False
        assert answer["hold_progress"] == pytest.approx(0.0)


def test_frames_over_the_budget_are_dropped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_fps_per_conn", 2)

    with TestClient(build_app()).websocket_connect("/ws/practice") as session:
        authenticate(session)
        session.send_json(start())
        for seq in range(2):
            session.send_json(frame(seq=seq))
            assert session.receive_json()["type"] == "eval"

        # Third frame inside the same second is over budget: dropped silently,
        # so the next thing on the wire is the answer to a later message.
        session.send_json(frame(seq=99))
        session.send_json({"type": "stop"})
        session.send_json(frame(seq=100))

        assert session.receive_json()["code"] == "NOT_STARTED"
