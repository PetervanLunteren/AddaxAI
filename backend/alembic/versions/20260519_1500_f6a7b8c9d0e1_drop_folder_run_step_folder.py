"""drop folder-run "folder" step

The folder-run flow had a dedicated folder-picker step (URL ``/folder``)
that only hosted the folder selector + previous-run notice. That step
has been merged into the Setup step: the folder selector lives at the
top of Setup now, with progressive disclosure of the model form once
the folder scan is valid. The step is removed from the
``FolderRunStep`` Literal in both backend and frontend.

Any project whose ``folder_run_state.step`` still says ``"folder"``
from before this change is translated to ``"model"`` so the next read
does not blow up on an unknown literal. Setup is the natural landing
spot because the user had not yet committed any settings.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-19 15:00:00

"""
from collections.abc import Sequence

from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE projects "
        "SET folder_run_state = json_set(folder_run_state, '$.step', 'model') "
        "WHERE folder_run_state IS NOT NULL "
        "  AND json_extract(folder_run_state, '$.step') = 'folder'"
    )


def downgrade() -> None:
    pass
