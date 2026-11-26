"""drop the independence interval from the reprocess hash

`Project.postprocessing_settings_hash` is the hash of the settings a
reprocess applies; a project whose current hash differs shows "reprocess
needed". The independence interval no longer groups anything that is
stored (the statistics apply it), so it left the hash. Without this step
every project with classifications would show the banner after the update.

Per project: where the stored hash equals the old formula on the current
row, write the new formula. Any other hash is left alone, so a project that
needed a reprocess still does, and one that did not still does not. Both
formulas are written out here because a migration does not import app code;
the new one must stay equal to `compute_postprocessing_settings_hash`.

Revision ID: c8e0a2b4d6f9
Revises: b7d9f1a3c5e7
Create Date: 2026-11-26 16:00:00

"""
import hashlib
import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8e0a2b4d6f9"
down_revision: str | None = "b7d9f1a3c5e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _settings_hash(row, *, with_interval: bool) -> str:
    # Raw rows hold booleans as 0/1 and JSON as text; the app hashes the
    # Python values, so convert to those first.
    settings = {
        "event_smoothing": bool(row.sequence_smoothing),
        "smoothing_strength": row.smoothing_strength,
        "taxonomic_rollup": bool(row.taxonomic_rollup),
        "excluded_classes": sorted(_json(row.excluded_classes) or []),
        "country_code": row.country_code,
        "state_code": row.state_code,
    }
    if with_interval:
        settings["independence_interval"] = row.independence_interval
    taxa_mapping = _json(row.taxa_mapping)
    if taxa_mapping:
        settings["taxa_mapping"] = taxa_mapping["rows"]
    canonical = json.dumps(settings, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _restamp(*, old_has_interval: bool) -> None:
    bind = op.get_bind()
    rows = bind.execute(sa.text(
        "SELECT id, postprocessing_settings_hash, sequence_smoothing, smoothing_strength, "
        "taxonomic_rollup, independence_interval, excluded_classes, country_code, "
        "state_code, taxa_mapping FROM projects "
        "WHERE postprocessing_settings_hash IS NOT NULL"
    )).all()
    for row in rows:
        if row.postprocessing_settings_hash == _settings_hash(
            row, with_interval=old_has_interval
        ):
            bind.execute(
                sa.text("UPDATE projects SET postprocessing_settings_hash = :h WHERE id = :i"),
                {"h": _settings_hash(row, with_interval=not old_has_interval), "i": row.id},
            )


def upgrade() -> None:
    _restamp(old_has_interval=True)


def downgrade() -> None:
    _restamp(old_has_interval=False)
