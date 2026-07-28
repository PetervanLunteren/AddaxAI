"""rename folder-run "review" step to "edit"

The folder-run verification step (and the research-projects page it
mirrors) was renamed from "Verification"/"Verify" to "Edit". The
folder-run URL slug + persisted ``folder_run_state.step`` value follow:
``"review"`` becomes ``"edit"``. The ``FolderRunStep`` Literal is
updated in both backend and frontend.

Any project whose ``folder_run_state.step`` still says ``"review"`` is
translated to ``"edit"`` so the next read does not blow up on an
unknown literal.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-05-20 10:00:00

"""
from collections.abc import Sequence

from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE projects "
        "SET folder_run_state = json_set(folder_run_state, '$.step', 'edit') "
        "WHERE folder_run_state IS NOT NULL "
        "  AND json_extract(folder_run_state, '$.step') = 'review'"
    )


def downgrade() -> None:
    pass
