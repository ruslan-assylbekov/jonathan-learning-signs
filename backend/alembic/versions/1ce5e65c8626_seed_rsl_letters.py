"""seed rsl letters

Seeds the 33 letters of the Russian alphabet as RSL dactyl targets.

The alphabet itself is a plain fact, so it belongs in a migration. What is
deliberately left NULL is everything that must come from the data rather than
from this file (CLAUDE.md 8.1, 20):

* ``is_static`` -- the Bukva authors disagree on whether 7 or 8 letters are
  dynamic, so it is decided from the reference and recorded in
  ``docs/alphabet.md``. ``DbLetterCatalog`` offers only letters marked static,
  so until that is filled in no letter is presented as practisable.
* ``template_angles`` and ``tolerance_deg`` -- produced by
  ``ml/build_templates.py`` from the extracted dataset. A letter without a
  template is practised without per-finger hints.

Whether Bukva's 33 classes line up exactly with these symbols is checked by
``make inspect``; a mismatch is fixed in a later migration, not by editing this
one.

Revision ID: 1ce5e65c8626
Revises: 67d137aec270
Create Date: 2026-10-16 17:37:08.108680

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "1ce5e65c8626"
down_revision: str | Sequence[str] | None = "67d137aec270"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ALPHABET = "rsl"
LETTERS = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

letters_table = sa.table(
    "letters",
    sa.column("symbol", sa.String),
    sa.column("alphabet", sa.String),
    sa.column("sort_order", sa.Integer),
)


def upgrade() -> None:
    op.bulk_insert(
        letters_table,
        [
            {"symbol": symbol, "alphabet": ALPHABET, "sort_order": order}
            for order, symbol in enumerate(LETTERS)
        ],
    )


def downgrade() -> None:
    op.execute(letters_table.delete().where(letters_table.c.alphabet == ALPHABET))
