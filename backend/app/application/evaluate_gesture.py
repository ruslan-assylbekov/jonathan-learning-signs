"""Evaluating one practice session, frame by frame.

The use case owns the order of operations -- normalize, classify, stabilize,
advise -- and nothing else. Transport lives in ``presentation/ws``, the model
in ``infrastructure/ml``.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.entities import Evaluation
from app.domain.gesture.features import Handedness, extract_features
from app.domain.gesture.feedback import LetterTemplate, hints_for
from app.domain.gesture.stabilizer import Stabilizer
from app.domain.ports import GestureClassifier


class PracticeSession:
    """One learner working on one target, for the life of one connection.

    Holds the hold-counter, so it is per-connection state: created on ``start``
    and replaced when the target changes.

    The template is passed in already resolved rather than looked up per frame.
    At 15 frames a second a lookup per frame would mean a query per frame for a
    value that cannot change during the session.
    """

    def __init__(
        self,
        *,
        target: str,
        aspect: float,
        classifier: GestureClassifier,
        template: LetterTemplate | None,
        confidence_threshold: float,
        hold_frames: int,
    ) -> None:
        self._aspect = aspect
        self._classifier = classifier
        self._template = template
        self._stabilizer = Stabilizer(
            target, confidence_threshold=confidence_threshold, hold_frames=hold_frames
        )

    @property
    def target(self) -> str:
        return self._stabilizer.target

    def evaluate(
        self,
        landmarks: Sequence[Sequence[float]],
        handedness: Handedness,
        *,
        mirrored: bool,
    ) -> Evaluation:
        """Judge one frame.

        Raises:
            InvalidLandmarksError: malformed frame; the caller reports
                ``INVALID_FRAME`` and keeps the connection.
        """
        features = extract_features(landmarks, handedness, aspect=self._aspect, mirrored=mirrored)
        prediction = self._classifier.predict(features)
        hold = self._stabilizer.update(prediction.label, prediction.confidence)

        # No hand in frame means nothing to correct, and a correct hold needs no
        # advice either (CLAUDE.md 6.4, 6.5). A letter with no template yet is
        # practised without hints rather than against invented angles.
        hints = []
        if not hold.is_no_sign and not hold.correct and self._template is not None:
            hints = hints_for(features, self._template)

        return Evaluation(
            prediction=prediction,
            correct=hold.correct,
            hold_progress=hold.hold_progress,
            succeeded=hold.succeeded,
            is_no_sign=hold.is_no_sign,
            hints=hints,
        )
