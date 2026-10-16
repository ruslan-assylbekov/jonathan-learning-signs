"""SQLAlchemy models for the schema in CLAUDE.md 9.

Infrastructure: the domain knows nothing about these. Every change goes through
a new Alembic migration -- applied migrations are never edited.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Letter(Base):
    """A practice target.

    ``template_angles`` and ``is_static`` are nullable because they come from
    the dataset, not from this file: templates are produced by
    ``ml/build_templates.py`` and the static subset is decided from the Bukva
    reference (CLAUDE.md 8.1). A letter with no template is practised without
    per-finger hints rather than against invented reference angles.
    """

    __tablename__ = "letters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(8))
    alphabet: Mapped[str] = mapped_column(String(16))
    reference_image_url: Mapped[str | None] = mapped_column(String(512))
    template_angles: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    tolerance_deg: Mapped[float | None] = mapped_column(Float)
    is_static: Mapped[bool | None] = mapped_column(Boolean)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("alphabet", "symbol", name="uq_letters_alphabet_symbol"),)


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    letter_id: Mapped[int] = mapped_column(ForeignKey("letters.id", ondelete="CASCADE"))
    mode: Mapped[str] = mapped_column(String(16))
    success: Mapped[bool] = mapped_column(Boolean)
    time_to_success_ms: Mapped[float | None] = mapped_column(Float)
    avg_confidence: Mapped[float | None] = mapped_column(Float)
    frames_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class Progress(Base):
    """Per-user, per-letter standing. Written in the same transaction as the
    attempt that changed it (CLAUDE.md 9)."""

    __tablename__ = "progress"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    letter_id: Mapped[int] = mapped_column(
        ForeignKey("letters.id", ondelete="CASCADE"), primary_key=True
    )
    attempts_count: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    mastery_score: Mapped[float] = mapped_column(Float, default=0.0)
    last_practiced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("success_rate >= 0 AND success_rate <= 1", name="ck_progress_rate"),
    )


class Sample(Base):
    """A landmark sample recorded through the data-collection mode.

    Landmarks only -- never frames or video (CLAUDE.md 8.3). ``feature_version``
    is stored so samples taken under older normalization can be excluded
    instead of silently mixed in.
    """

    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    letter_id: Mapped[int] = mapped_column(ForeignKey("letters.id", ondelete="CASCADE"))
    participant: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16))
    feature_version: Mapped[int] = mapped_column(Integer)
    landmarks: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
