"""The /ws/practice handler (CLAUDE.md 7).

Transport only: parse a message, call one use case, serialize the result
(CLAUDE.md 5). No evaluation logic lives here.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.application.evaluate_gesture import PracticeSession
from app.application.record_attempt import AttemptRecord, Attempts
from app.domain.gesture.features import Handedness, InvalidLandmarksError, aspect_of
from app.domain.ports import GestureClassifier, LetterCatalog, TemplateSource, TokenVerifier
from app.presentation.ws.schemas import (
    AuthMessage,
    ClientMessage,
    ErrorOut,
    EvalOut,
    FrameMessage,
    StartMessage,
    StopMessage,
    SuccessOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_messages: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


@dataclass(frozen=True, slots=True)
class PracticeDeps:
    """Everything the handler needs, wired in main.py."""

    classifier: GestureClassifier
    templates: TemplateSource
    letters: LetterCatalog
    tokens: TokenVerifier
    attempts: Attempts
    confidence_threshold: float
    hold_frames: int
    max_fps: int


class FrameRateLimiter:
    """Drops frames above the per-connection ceiling (CLAUDE.md 7).

    A client is asked to send 10-15 Hz; this stops a faulty or hostile one from
    spending the server's CPU at camera frame rate.
    """

    def __init__(self, max_fps: int, *, window_seconds: float = 1.0) -> None:
        self._max = max_fps
        self._window = window_seconds
        self._recent: deque[float] = deque()

    def allow(self, now: float) -> bool:
        while self._recent and now - self._recent[0] > self._window:
            self._recent.popleft()
        if len(self._recent) >= self._max:
            return False
        self._recent.append(now)
        return True


@router.websocket("/ws/practice")
async def practice(websocket: WebSocket) -> None:
    deps: PracticeDeps = websocket.app.state.practice_deps
    await websocket.accept()
    try:
        await _converse(websocket, deps)
    except WebSocketDisconnect:
        logger.debug("practice socket closed by the client")


@dataclass
class Attendance:
    """What the current target's attempt will be made of."""

    letter_id: int
    mode: str
    started_at: float
    frames: int = 0
    confidence_total: float = 0.0

    def saw(self, confidence: float) -> None:
        self.frames += 1
        self.confidence_total += confidence

    def to_record(self, user_id: int, *, success: bool, elapsed_ms: float) -> AttemptRecord:
        return AttemptRecord(
            user_id=user_id,
            letter_id=self.letter_id,
            mode=self.mode,
            success=success,
            time_to_success_ms=elapsed_ms if success else None,
            avg_confidence=self.confidence_total / self.frames if self.frames else None,
            frames_count=self.frames,
        )


async def _converse(websocket: WebSocket, deps: PracticeDeps) -> None:
    limiter = FrameRateLimiter(deps.max_fps)
    user_id: int | None = None
    session: PracticeSession | None = None
    attendance: Attendance | None = None

    while True:
        payload = await websocket.receive_json()
        try:
            message = _messages.validate_python(payload)
        except ValidationError as error:
            await _fail(websocket, "INVALID_MESSAGE", _first_problem(error))
            continue

        if isinstance(message, AuthMessage):
            user_id = deps.tokens.user_id(message.token)
            if user_id is None:
                await _fail(websocket, "UNAUTHENTICATED", "token rejected")
            continue

        if user_id is None:
            await _fail(websocket, "UNAUTHENTICATED", "send an auth message first")
            continue

        if isinstance(message, StartMessage):
            letter = await deps.letters.by_id(message.letter_id)
            if letter is None:
                await _fail(websocket, "UNKNOWN_LETTER", f"no letter {message.letter_id}")
                continue
            session = await _new_session(message, letter.symbol, deps)
            attendance = Attendance(
                letter_id=letter.id, mode=message.mode, started_at=time.perf_counter()
            )
            continue

        if isinstance(message, StopMessage):
            # Giving up is a result too: without this, every unfinished letter
            # would silently vanish and progress would only ever see successes.
            if attendance is not None and attendance.frames:
                await _record(deps, attendance, user_id, success=False)
            session, attendance = None, None
            continue

        if session is None or attendance is None:
            await _fail(websocket, "NOT_STARTED", "send a start message first")
            continue

        await _handle_frame(websocket, message, session, deps, limiter, attendance, user_id)


async def _new_session(message: StartMessage, target: str, deps: PracticeDeps) -> PracticeSession:
    # Resolved here, once, rather than inside the per-frame path.
    template = await deps.templates.template_for(target)
    return PracticeSession(
        target=target,
        aspect=aspect_of(message.width, message.height),
        classifier=deps.classifier,
        template=template,
        confidence_threshold=deps.confidence_threshold,
        hold_frames=deps.hold_frames,
    )


async def _handle_frame(
    websocket: WebSocket,
    message: FrameMessage,
    session: PracticeSession,
    deps: PracticeDeps,
    limiter: FrameRateLimiter,
    attendance: Attendance,
    user_id: int,
) -> None:
    if not limiter.allow(time.monotonic()):
        return  # dropped on purpose; the client is over its frame budget

    began = time.perf_counter()
    try:
        evaluation = session.evaluate(
            message.landmarks,
            Handedness(message.handedness),
            mirrored=message.mirrored,
        )
    except InvalidLandmarksError as error:
        await _fail(websocket, "INVALID_FRAME", str(error))
        return
    elapsed_ms = (time.perf_counter() - began) * 1000
    attendance.saw(evaluation.prediction.confidence)

    await websocket.send_text(
        EvalOut.of(
            evaluation, seq=message.seq, t_client=message.t_client, server_ms=elapsed_ms
        ).model_dump_json()
    )

    if evaluation.succeeded:
        taken_ms = round((time.perf_counter() - attendance.started_at) * 1000, 1)
        await websocket.send_text(
            SuccessOut(
                letter_id=attendance.letter_id, time_to_success_ms=taken_ms
            ).model_dump_json()
        )
        await _record(deps, attendance, user_id, success=True, elapsed_ms=taken_ms)


async def _record(
    deps: PracticeDeps,
    attendance: Attendance,
    user_id: int,
    *,
    success: bool,
    elapsed_ms: float = 0.0,
) -> None:
    """Store the attempt, never at the cost of the session.

    A database hiccup must not drop the learner's connection mid-practice, so
    it is logged and the lesson continues.
    """
    try:
        await deps.attempts.record(
            attendance.to_record(user_id, success=success, elapsed_ms=elapsed_ms)
        )
    except SQLAlchemyError:
        logger.exception("could not record attempt for letter %s", attendance.letter_id)


async def _fail(websocket: WebSocket, code: str, detail: str) -> None:
    await websocket.send_text(ErrorOut(code=code, detail=detail).model_dump_json())


def _first_problem(error: ValidationError) -> str:
    """One readable reason, without echoing the whole landmark array back."""
    problems = error.errors()
    if not problems:
        return "malformed message"
    first = problems[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else str(first["msg"])
