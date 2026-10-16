"""per-finger hint tolerance

Replaces ``letters.tolerance_deg`` (one number per letter) with
``letters.finger_tolerances`` (one per finger, JSONB).

One tolerance per letter had to be applied to the worst of five fingers, so it
was sized for the noisiest -- ring and little finger vary about twice as much
as thumb and index -- and 11 of 25 letters hit the 60-degree ceiling. Per-finger
tolerances keep hints tight where the data is tight.

Dropping the old column loses nothing that cannot be rebuilt: both are derived
by ``ml/build_templates.py --apply``. Downgrading restores an empty column, not
the old values.

Revision ID: 62d774db2da2
Revises: a10ff68cd5a2
Create Date: 2026-10-17 10:12:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "62d774db2da2"
down_revision: str | Sequence[str] | None = "a10ff68cd5a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "letters",
        sa.Column("finger_tolerances", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    # Templates built against the old single tolerance are cleared rather than
    # left half-valid: the repository treats angles without tolerances as no
    # template, and build_templates.py --apply restores both together.
    op.execute("UPDATE letters SET template_angles = NULL")
    op.drop_column("letters", "tolerance_deg")


def downgrade() -> None:
    op.add_column("letters", sa.Column("tolerance_deg", sa.Float(), nullable=True))
    op.drop_column("letters", "finger_tolerances")
