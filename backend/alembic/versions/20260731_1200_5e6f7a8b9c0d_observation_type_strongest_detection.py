"""Recompute observation_type under the strongest-detection rule

``File.observation_type`` used to be the highest-*priority* detector
category among a file's passing detections (animal > human > vehicle),
so one animal box at 0.21 outranked thirty person boxes at 0.95. It is
now the raw category of the single strongest passing detection, ranked
verified first then confidence. See ``app/ml/observation_type.py``.

Two things therefore make every stored value potentially wrong, and the
column is denormalised so nothing recomputes it on its own:

1. ``person`` used to be stored as ``human``. The value is now the
   detector's own category, untranslated, so a file with a person is
   spelled ``person``. Only the Camtrap DP export translates back, since
   its ``observationType`` has a fixed controlled vocabulary.
2. A mixed file may resolve differently: a person box more confident
   than an animal box now wins.

This recomputes the column for every file from the detections already in
the database, at each file's own project ``counting_threshold``, which is
exactly what ``derive_observation_type`` does in Python.

Written as one correlated subquery rather than ``UPDATE ... FROM`` (which
needs SQLite 3.33) or a CTE, so it runs on every SQLite the app ships
with. It touches every ``files`` row once; on a large library that is
slow, and per DEVELOPERS.md a slow migration is not a failure.

Revision ID: 5e6f7a8b9c0d
Revises: 4d5e6f7a8b9c
Create Date: 2026-07-31 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e6f7a8b9c0d"
down_revision: str | None = "4d5e6f7a8b9c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# The deciding detection: passing (over its project's threshold, or
# verified), strongest first. The trailing `category` in the ORDER BY is
# only there to make an exact tie deterministic, matching the Python.
_STRONGEST_CATEGORY = """
    SELECT d.category
    FROM detections d
    WHERE d.file_id = files.id
      AND (
        d.verified = 1
        OR d.confidence >= (
            SELECT p.counting_threshold
            FROM deployments dep
            JOIN projects p ON p.id = dep.project_id
            WHERE dep.id = files.deployment_id
        )
      )
    ORDER BY d.verified DESC, d.confidence DESC, d.category ASC
    LIMIT 1
"""

# The rule this replaces, for the downgrade: highest-priority category
# among passing detections, with `person` spelled back to `human`.
_HIGHEST_PRIORITY_OBS = """
    SELECT CASE d.category WHEN 'person' THEN 'human' ELSE d.category END
    FROM detections d
    WHERE d.file_id = files.id
      AND d.category IN ('animal', 'person', 'vehicle')
      AND (
        d.verified = 1
        OR d.confidence >= (
            SELECT p.counting_threshold
            FROM deployments dep
            JOIN projects p ON p.id = dep.project_id
            WHERE dep.id = files.deployment_id
        )
      )
    ORDER BY CASE d.category
        WHEN 'animal' THEN 3 WHEN 'person' THEN 2 ELSE 1 END DESC
    LIMIT 1
"""


def _has_files_table(bind) -> bool:
    return "files" in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_files_table(bind):
        return
    op.execute(
        f"UPDATE files SET observation_type = "
        f"COALESCE(({_STRONGEST_CATEGORY}), 'blank')"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_files_table(bind):
        return
    op.execute(
        f"UPDATE files SET observation_type = "
        f"COALESCE(({_HIGHEST_PRIORITY_OBS}), 'blank')"
    )
