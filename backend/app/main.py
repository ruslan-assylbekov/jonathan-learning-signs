"""Application wiring. Dependencies are chosen here and nowhere else (CLAUDE.md 5)."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.domain.gesture.features import FEATURE_VERSION
from app.domain.ports import GestureClassifier, LetterCatalog, TemplateSource, TokenVerifier
from app.infrastructure.auth.dev import ClosedTokenVerifier, DevTokenVerifier
from app.infrastructure.ml.stub import StubClassifier, StubLetterCatalog, StubTemplates
from app.presentation.api import letters
from app.presentation.ws import practice

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def build_app() -> FastAPI:
    app = FastAPI(title="Sign Assistant")

    classifier, catalog, templates = _gesture_stack()
    _check_feature_version(classifier)

    app.state.practice_deps = practice.PracticeDeps(
        classifier=classifier,
        templates=templates,
        letters=catalog,
        tokens=_token_verifier(),
        confidence_threshold=settings.confidence_threshold,
        hold_frames=settings.hold_frames,
        max_fps=settings.max_fps_per_conn,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(letters.router)
    app.include_router(practice.router)

    @app.get("/health")
    async def health() -> dict[str, str | int | bool]:
        return {
            "status": "ok",
            "feature_version": FEATURE_VERSION,
            "classifier": type(classifier).__name__,
            "stub": isinstance(classifier, StubClassifier),
        }

    return app


def _gesture_stack() -> tuple[GestureClassifier, LetterCatalog, TemplateSource]:
    """Pick the real stack or the stub, by whether a trained model is present.

    Deciding this from the artifact on disk rather than a flag means the app is
    runnable at every point in the project's life, and flips over by itself the
    moment ``model-v1`` is trained -- with nothing to remember to switch.

    The stub also brings its own letter catalog, because its classes are demo
    gestures rather than letters: pairing it with the real alphabet would offer
    33 targets that can never be matched.
    """
    if not settings.model_file.exists():
        logger.warning(
            "No model at %s: running on StubClassifier with demo targets. "
            "Nothing it reports is a result about sign language.",
            settings.model_file,
        )
        return StubClassifier(), StubLetterCatalog(), StubTemplates()

    raise NotImplementedError(
        f"A model is present at {settings.model_file} but no loader is wired yet "
        "(CLAUDE.md 19, M2). Implement infrastructure/ml loading before using it."
    )


def _check_feature_version(classifier: GestureClassifier) -> None:
    """Refuse to start on a feature/model mismatch (CLAUDE.md 6.3).

    The same 78 numbers would mean one thing to the extractor and another to
    the model -- which shows up as mysteriously poor accuracy, not as an error.
    """
    if classifier.feature_version != FEATURE_VERSION:
        raise RuntimeError(
            f"model expects feature version {classifier.feature_version}, "
            f"code produces {FEATURE_VERSION}. Retrain, or check out the matching model."
        )


def _token_verifier() -> TokenVerifier:
    """Closed by default: real JWT verification arrives in M3."""
    if settings.dev_auth:
        return DevTokenVerifier()
    logger.info("DEV_AUTH is off and JWT auth is not implemented yet: /ws/practice is closed")
    return ClosedTokenVerifier()


app = build_app()
