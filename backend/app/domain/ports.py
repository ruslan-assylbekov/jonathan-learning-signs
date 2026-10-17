"""Interfaces the application layer depends on (CLAUDE.md 5).

Protocols, so infrastructure implements them without importing the domain and
tests can substitute fakes without inheritance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from app.domain.entities import Account, AttemptRecord, Letter, LetterProgress, Prediction
from app.domain.gesture.features import Features
from app.domain.gesture.feedback import LetterTemplate
from app.domain.progress import Standing


@runtime_checkable
class GestureClassifier(Protocol):
    """Maps a feature vector to a label."""

    @property
    def feature_version(self) -> int:
        """Feature version the model was trained on.

        The app refuses to start when this disagrees with the code
        (CLAUDE.md 6.3): the same numbers would otherwise mean something
        different to the model than to the extractor.
        """

    @property
    def classes(self) -> tuple[str, ...]:
        """Every label the model can emit. The letter catalog offers nothing
        outside this set, so no target can be one the model never predicts."""

    def predict(self, features: Features) -> Prediction: ...


@runtime_checkable
class TemplateSource(Protocol):
    """Reference poses used to generate hints."""

    async def template_for(self, label: str) -> LetterTemplate | None:
        """The template for ``label``, or None when there is none to compare
        against -- in which case the frame is evaluated without hints.

        Awaited once when a session starts, never per frame."""


@runtime_checkable
class LetterCatalog(Protocol):
    """The practice targets available, however they are stored."""

    async def all(self) -> list[Letter]: ...

    async def by_id(self, letter_id: int) -> Letter | None:
        """The letter with this id, or None -- the caller reports
        ``UNKNOWN_LETTER`` rather than guessing a target."""


@runtime_checkable
class TokenVerifier(Protocol):
    """Decides who, if anyone, is behind a connection."""

    def user_id(self, token: str) -> int | None:
        """The user the token names, or None if it names nobody.

        Returning the id rather than a yes/no is what lets attempts be
        attributed without the transport layer decoding tokens itself."""


@runtime_checkable
class TokenIssuer(Protocol):
    """Mints access tokens for a signed-in user."""

    def issue(self, user_id: int) -> str: ...


class EmailTakenError(ValueError):
    """Raised by UserRepository.create for an address already registered."""


@runtime_checkable
class UserRepository(Protocol):
    async def create(self, email: str, password_hash: str) -> int:
        """Store a new account and return its id.

        Raises:
            EmailTakenError: the address is already registered.
        """

    async def find_by_email(self, email: str) -> Account | None: ...


@runtime_checkable
class PasswordHasher(Protocol):
    """Deliberately slow, so these are synchronous: callers run them off the
    event loop rather than stall every open socket for each sign-in."""

    def hash(self, password: str) -> str: ...

    def verify(self, password: str, password_hash: str) -> bool: ...


@runtime_checkable
class AttemptRepository(Protocol):
    async def record(
        self,
        attempt: AttemptRecord,
        advance: Callable[[Standing | None], Standing],
    ) -> None:
        """Store the attempt and the standing ``advance`` derives from the
        previous one, in one transaction, with that previous standing locked
        so concurrent attempts cannot both read it (CLAUDE.md 9)."""


@runtime_checkable
class ProgressRepository(Protocol):
    async def for_user(self, user_id: int) -> list[LetterProgress]:
        """Every letter the user has attempted, most recently practised first."""
