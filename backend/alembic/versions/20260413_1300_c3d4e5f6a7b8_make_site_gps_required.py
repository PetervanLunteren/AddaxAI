"""make site gps required

Revision ID: c3d4e5f6a7b8
Revises: bfd49dc7b370
Create Date: 2026-04-13 13:00:00.000000

Tightens Site.latitude and Site.longitude from nullable to NOT NULL.
The frontend create form already requires GPS, but the backend schema
and DB column allowed null, leaving a gap. The map page assumes every
site has coordinates, so this migration closes the gap.

Pre-flight check: aborts loudly if any existing site has a null
coordinate, listing the offending site IDs. Per CONVENTIONS.md #1
("Crash early and loudly"), we never silently coerce data.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'bfd49dc7b370'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    # Pre-flight: refuse to migrate if any site has null GPS.
    null_rows = bind.execute(
        sa.text(
            "SELECT id, name FROM sites "
            "WHERE latitude IS NULL OR longitude IS NULL"
        )
    ).fetchall()
    if null_rows:
        details = ", ".join(f"{name}({sid})" for sid, name in null_rows)
        raise RuntimeError(
            f"Cannot make site GPS required: {len(null_rows)} site(s) have "
            f"null coordinates. Fix them first: {details}"
        )

    with op.batch_alter_table("sites") as batch_op:
        batch_op.alter_column("latitude", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("longitude", existing_type=sa.Float(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("sites") as batch_op:
        batch_op.alter_column("latitude", existing_type=sa.Float(), nullable=True)
        batch_op.alter_column("longitude", existing_type=sa.Float(), nullable=True)
