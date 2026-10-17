"""The /ws/practice handler (CLAUDE.md 7).

Transport only: parse a message, call one use case, serialize the result
(CLAUDE.md 5). No evaluation logic lives here.
"""

from __future__ import annotations

import json
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
    conversation = _Conversation(websocket, deps)
    try:
        await conversation.run()
    except WebSocketDisconnect:
        logger.debug("practice socket closed by the client")
    finally:
        # Closing the tab mid-letter is giving up on it, like stop.
        await conversation.abandon()


@dataclass
class Attendance:
    """What the current target's attempt will be made of."""

    letter_id: int
    mode: str
    started_at: float
    frames: int = 0
    confidence_total: float = 0.0
    finished: bool = False
    """Set once the letter is recorded, so a later stop, switch or disconnect
    cannot record it a second time -- as a failure, after it was passed."""

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


class _Conversation:
    """One socket's state: who is connected and which letter they are on."""

    def __init__(self, websocket: WebSocket, deps: PracticeDeps) -> None:
        self._websocket = websocket
        self._deps = deps
        self._limiter = FrameRateLimiter(deps.max_fps)
        self._user_id: int | None = None
        self._session: PracticeSession | None = None
        self._attendance: Attendance | None = None

    async def run(self) -> None:
        while True:
            payload = await self._next_payload()
            if payload is None:
                continue
            try:
                message = _messages.validate_python(payload)
            except ValidationError as error:
                await _fail(self._websocket, "INVALID_MESSAGE", _first_problem(error))
                continue
            await self._handle(message)

    async def _next_payload(self) -> object | None:
        """The next parsed message, or None after reporting one that was not JSON.

        A malformed message must cost the sender an error, not the connection:
        receive_json would raise and drop the socket on a stray binary frame or
        a typo.
        """
        received = await self._websocket.receive()
        if received["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(received.get("code", 1000))
        text = received.get("text")
        if text is None:
            await _fail(self._websocket, "INVALID_MESSAGE", "send JSON text frames")
            return None
        try:
            parsed: object = json.loads(text)
            return parsed
        except json.JSONDecodeError:
            await _fail(self._websocket, "INVALID_MESSAGE", "not valid JSON")
            return None

    async def _handle(self, message: ClientMessage) -> None:
        if isinstance(message, AuthMessage):
            self._user_id = self._deps.tokens.user_id(message.token)
            if self._user_id is None:
                await _fail(self._websocket, "UNAUTHENTICATED", "token rejected")
            return

        if self._user_id is None:
            await _fail(self._websocket, "UNAUTHENTICATED", "send an auth message first")
            return

        if isinstance(message, StartMessage):
            letter = await self._deps.letters.by_id(message.letter_id)
            if letter is None:
                await _fail(self._websocket, "UNKNOWN_LETTER", f"no letter {message.letter_id}")
                return
            # Moving on from an unfinished letter is giving up on it.
            await self.abandon()
            self._session = await _new_session(message, letter.symbol, self._deps)
            self._attendance = Attendance(
                letter_id=letter.id, mode=message.mode, started_at=time.perf_counter()
            )
            return

        if isinstance(message, StopMessage):
            await self.abandon()
            self._session = None
            return

        if self._session is None or self._attendance is None:
            await _fail(self._websocket, "NOT_STARTED", "send a start message first")
            return

        await _handle_frame(
            self._websocket,
            message,
            self._session,
            self._deps,
            self._limiter,
            self._attendance,
            self._user_id,
        )

    async def abandon(self) -> None:
        """Record the current letter as failed, if it was tried and not finished.

        Giving up is a result too: without it only successes would reach
        progress, and a letter someone keeps abandoning would look untouched.
        A letter never shown a single frame is not an attempt.
        """
        attendance, self._attendance = self._attendance, None
        if (
            attendance is None
            or attendance.finished
            or not attendance.frames
            or self._user_id is None
        ):
            return
        attendance.finished = True
        await _record(self._deps, attendance, self._user_id, success=False)


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
        attendance.finished = True
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
