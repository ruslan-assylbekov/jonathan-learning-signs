"""Application wiring. Dependencies are chosen here and nowhere else (CLAUDE.md 5)."""

from __future__ import annotations

import dataclasses
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.accounts import Accounts
from app.application.record_attempt import Attempts
from app.application.user_progress import UserProgress
from app.config import settings
from app.domain.gesture.features import FEATURE_VERSION
from app.domain.ports import GestureClassifier, LetterCatalog, TemplateSource
from app.infrastructure.auth.dev import DevTokenVerifier
from app.infrastructure.auth.passwords import BcryptHasher
from app.infrastructure.auth.tokens import JwtTokens
from app.infrastructure.db import models
from app.infrastructure.db.repositories import (
    DbAttemptRepository,
    DbLetterCatalog,
    DbProgressRepository,
    DbTemplateSource,
    DbUserRepository,
)
from app.infrastructure.db.session import session_factory
from app.infrastructure.ml.model import JoblibClassifier
from app.infrastructure.ml.stub import (
    StubAttemptRepository,
    StubClassifier,
    StubLetterCatalog,
    StubTemplates,
)
from app.presentation.api import auth, errors, letters, progress
from app.presentation.ws import practice

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

DEV_EMAIL = "dev@localhost"


def build_app(sessions: async_sessionmaker[AsyncSession] | None = None) -> FastAPI:
    """Assemble the app.

    ``sessions`` is injectable so tests can supply an engine with a pool that
    suits them; the module-level factory is the default everywhere else.
    """
    sessions = sessions or session_factory
    app = FastAPI(title="Sign Assistant", lifespan=_lifespan)

    classifier, catalog, templates = _gesture_stack(sessions)
    _check_feature_version(classifier)
    stub = isinstance(classifier, StubClassifier)

    tokens = JwtTokens(
        settings.jwt_secret,
        ttl_minutes=settings.jwt_ttl_min,
        # A placeholder secret is refused outright unless this is a local
        # development run, where the whole point is to not need setup.
        allow_insecure_secret=settings.dev_auth,
    )

    app.state.sessions = sessions
    app.state.accounts = Accounts(DbUserRepository(sessions), BcryptHasher(), tokens)
    app.state.user_progress = UserProgress(DbProgressRepository(sessions))
    app.state.practice_deps = practice.PracticeDeps(
        classifier=classifier,
        templates=templates,
        letters=catalog,
        tokens=tokens,
        attempts=Attempts(StubAttemptRepository() if stub else DbAttemptRepository(sessions)),
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
    errors.install(app)
    app.include_router(auth.router)
    app.include_router(letters.router)
    app.include_router(progress.router)
    app.include_router(practice.router)

    @app.get("/health")
    async def health() -> dict[str, str | int | bool]:
        return {
            "status": "ok",
            "feature_version": FEATURE_VERSION,
            "classifier": type(classifier).__name__,
            "stub": stub,
        }

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Swap in the development verifier once its account exists.

    The token verifier is built before startup, but the dev account needs the
    database, so the substitution happens here rather than leaving the handler
    to deal with a user id that may not exist yet.
    """
    if settings.dev_auth:
        app.state.practice_deps = dataclasses.replace(
            app.state.practice_deps,
            tokens=DevTokenVerifier(await _ensure_dev_user(app.state.sessions)),
        )
    yield


async def _ensure_dev_user(sessions: async_sessionmaker[AsyncSession]) -> int:
    """The local account DEV_AUTH attributes everything to.

    It exists so attempts and progress are exercised in development too. Its
    password is random and discarded, because nothing should ever sign in as it.
    """
    async with sessions() as session:
        existing = (
            await session.execute(select(models.User).where(models.User.email == DEV_EMAIL))
        ).scalar_one_or_none()
        if existing is not None:
            return int(existing.id)

        user = models.User(
            email=DEV_EMAIL, password_hash=BcryptHasher().hash(secrets.token_urlsafe(32))
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        logger.warning("created the local %s account for DEV_AUTH", DEV_EMAIL)
        return int(user.id)


def _gesture_stack(
    sessions: async_sessionmaker[AsyncSession],
) -> tuple[GestureClassifier, LetterCatalog, TemplateSource]:
    """Pick the real stack or the stub, by whether a trained model is present.

    Deciding this from the artifact on disk rather than a flag means the app is
    runnable at every point in the project's life, and flips over by itself the
    moment a model is trained -- with nothing to remember to switch.

    The stub also brings its own letter catalog, because its classes are demo
    gestures rather than letters: pairing it with the real alphabet would offer
    targets that can never be matched. Likewise the real model gets the real
    catalog, which offers only the static letters it was trained on.
    """
    if not settings.model_file.exists():
        logger.warning(
            "No model at %s: running on StubClassifier with demo targets. "
            "Nothing it reports is a result about sign language.",
            settings.model_file,
        )
        return StubClassifier(), StubLetterCatalog(), StubTemplates()

    classifier = JoblibClassifier(settings.model_file)
    logger.info(
        "Loaded %s: %s, %d classes, feature version %d",
        settings.model_file.name,
        classifier.metadata.get("model_type"),
        len(classifier.classes),
        classifier.feature_version,
    )
    # The catalog offers only letters the model can predict: a target outside
    # its classes could be started and never recognised.
    catalog = DbLetterCatalog(sessions, classes=classifier.classes)
    return classifier, catalog, DbTemplateSource(sessions)


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


app = build_app()
