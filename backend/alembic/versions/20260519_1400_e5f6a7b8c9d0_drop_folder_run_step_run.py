"""drop folder-run "run" step

The folder-run flow had a dedicated "Analysis" step (URL ``/run``) that
mostly hosted a "Press play" button before the JobProgressModal opened.
That step has been merged into the Setup step: clicking Start analysis
on Setup opens the modal directly. The step is removed from the
``FolderRunStep`` Literal in both backend and frontend.

Any project whose ``folder_run_state.step`` still says ``"run"`` from
before this change is translated to ``"model"`` so the next read does
not blow up on an unknown literal. Setup is the right landing spot
because the user has not yet kicked off analysis from the merged flow,
and the form will be pre-populated from saved settings.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-19 14:00:00

"""
from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLite's json_set returns the modified blob; we narrow the WHERE
    # to rows that actually have step='run' so we don't churn untouched
    # projects.
    op.execute(
        "UPDATE projects "
        "SET folder_run_state = json_set(folder_run_state, '$.step', 'model') "
        "WHERE folder_run_state IS NOT NULL "
        "  AND json_extract(folder_run_state, '$.step') = 'run'"
    )


def downgrade() -> None:
    # No clean reverse — we can't tell which 'model'-stepped runs were
    # originally 'run'-stepped. Leave the data alone on downgrade.
    pass
