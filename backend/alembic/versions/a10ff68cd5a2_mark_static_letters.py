"""mark static letters

Fills in ``letters.is_static``, which the seed migration deliberately left
NULL until the reference had been read.

Source: Figure 1 of Bukva (arXiv:2410.08675), whose caption says "Arrows
indicate the direction of movement for 8 dynamic signs". The eight signs drawn
with an arrow are Д, Ё, З, Й, Ц, Щ, Ъ and Ь; see ``docs/alphabet.md`` for the
quotes and for the caveat that the paper records another source counting seven.
This is to be cross-checked by measuring per-class landmark displacement once
extraction has run; if measurement disagrees, it wins and a later migration
corrects this one.

The dynamic letters stay in the table rather than being deleted: they are Tier
3 work (CLAUDE.md 2, item 13), and attempts already recorded against them must
keep their foreign key.

Revision ID: a10ff68cd5a2
Revises: 1ce5e65c8626
Create Date: 2026-10-16 18:02:11.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a10ff68cd5a2"
down_revision: str | Sequence[str] | None = "1ce5e65c8626"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ALPHABET = "rsl"
DYNAMIC = ("Д", "Ё", "З", "Й", "Ц", "Щ", "Ъ", "Ь")

letters = sa.table(
    "letters",
    sa.column("symbol", sa.String),
    sa.column("alphabet", sa.String),
    sa.column("is_static", sa.Boolean),
)


def upgrade() -> None:
    op.execute(
        letters.update()
        .where(letters.c.alphabet == ALPHABET)
        .values(is_static=letters.c.symbol.notin_(DYNAMIC))
    )


def downgrade() -> None:
    op.execute(letters.update().where(letters.c.alphabet == ALPHABET).values(is_static=None))
