"""Wire format for /ws/practice (CLAUDE.md 7).

Pydantic lives only in this layer (CLAUDE.md 5). Any change here must land in
the same commit as the frontend's ``ws-client.ts`` and the integration tests.
"""

from __future__ import annotations

from typing import Annotated, Literal

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from app.domain.entities import Evaluation
from app.domain.gesture.features import NUM_LANDMARKS

Landmark = Annotated[list[float], Len(3, 3)]


class Incoming(BaseModel):
    """Base for client messages. Unknown fields are a bug, not a courtesy."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AuthMessage(Incoming):
    type: Literal["auth"]
    token: str


class StartMessage(Incoming):
    type: Literal["start"]
    mode: str
    letter_id: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FrameMessage(Incoming):
    type: Literal["frame"]
    seq: int
    t_client: float
    handedness: Literal["Left", "Right"]
    mirrored: bool
    landmarks: Annotated[list[Landmark], Len(NUM_LANDMARKS, NUM_LANDMARKS)]


class StopMessage(Incoming):
    type: Literal["stop"]


ClientMessage = Annotated[
    AuthMessage | StartMessage | FrameMessage | StopMessage,
    Field(discriminator="type"),
]


class HintOut(BaseModel):
    """A hint as the client sees it: the deviation is for ranking, not display."""

    finger: str
    code: str


class EvalOut(BaseModel):
    type: Literal["eval"] = "eval"
    seq: int
    pred: str
    confidence: float
    correct: bool
    hold_progress: float
    hints: list[HintOut]
    server_ms: float
    t_client: float

    @classmethod
    def of(cls, evaluation: Evaluation, *, seq: int, t_client: float, server_ms: float) -> EvalOut:
        return cls(
            seq=seq,
            pred=evaluation.prediction.label,
            confidence=round(evaluation.prediction.confidence, 4),
            correct=evaluation.correct,
            hold_progress=round(evaluation.hold_progress, 4),
            hints=[HintOut(finger=hint.finger, code=hint.code) for hint in evaluation.hints],
            server_ms=round(server_ms, 3),
            t_client=t_client,
        )


class SuccessOut(BaseModel):
    type: Literal["success"] = "success"
    letter_id: int
    time_to_success_ms: float


class ErrorOut(BaseModel):
    type: Literal["error"] = "error"
    code: str
    detail: str
