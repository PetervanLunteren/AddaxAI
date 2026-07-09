"""
Programmatic alembic helpers used at app startup.

`init_db()` calls these to reconcile the live SQLite schema with
alembic's migration history. The reconciliation is fingerprint-based:
`detect_schema_revision()` inspects the live schema and returns the
latest alembic revision whose state matches it. `init_db()` then
stamps that revision (if it disagrees with `alembic_version`) and
runs `upgrade_to_head()` to apply anything newer. This mirrors what
mature migration tools (Flyway baselining, Django `--fake`, Rails
`db:schema:load` + `migrate`) do: trust the introspected schema, not
a possibly-wrong stored version.

Alembic imports are local to function bodies so test/cold-path callers
that only need `_resolve_backend_dir()` don't pay the import cost.

## Adding a new migration

When you add a migration that changes the schema, update
`SCHEMA_FINGERPRINTS` below with one line for the new revision. Pick
a column or table the migration adds (or removes) as the signature.
The list must stay in chronological order (oldest first).

The fingerprint is what protects users whose DB is stamped at a wrong
revision (e.g. a historical stamp_head bug, a half-applied migration
on power loss, a hand-restored backup). Without it, alembic believes
the DB is at the recorded revision and never runs the migration that
would have added the column the model expects. With it, we detect the
mismatch on the next startup and re-stamp at the correct revision so
`upgrade_to_head()` can apply the missing migrations.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _Fingerprint:
    """One row in `SCHEMA_FINGERPRINTS`.

    Every alembic revision on disk has an entry, even ones whose
    schema effect is not detectable via column introspection. The
    entry expresses one of two intents:

    1. `table` and `column` set: the migration adds a column. The
       schema is at this revision or later when the column exists in
       the live DB. `column=None` with `table` set means the migration
       creates the table itself — "table exists" is the fingerprint.
    2. `table=None` and `column=None`: the migration's effect is not
       detectable from `PRAGMA table_info` (data-only DELETE, column
       nullability change, index rename, etc.). These entries do not
       participate in `detect_schema_revision`; they are listed so the
       "every revision has a fingerprint" test still passes and so
       reviewers see a per-revision rationale.

    `description` is human-readable context; it's logged when the
    fingerprint walk picks the revision and shown in test failures.
    """

    revision: str
    table: str | None
    column: str | None = None
    description: str = ""

    @property
    def is_detectable(self) -> bool:
        return self.table is not None


# Ordered oldest to newest. detect_schema_revision walks this list in
# reverse, skipping non-detectable entries, and returns the first row
# whose schema is satisfied in the live DB.
#
# Every alembic revision MUST have an entry here. For non-additive
# migrations (data-only DELETEs, nullability changes, index work),
# set table=None and column=None with a short description of why.
# `test_every_alembic_revision_has_a_fingerprint` enforces this.
#
# The fingerprint is what protects users whose DB is stamped at a
# wrong revision (a historical stamp_head bug, a half-applied
# migration on power loss, a hand-restored backup). Without it,
# alembic believes the DB is at the recorded revision and never runs
# the migration that would have added the column the model expects.
SCHEMA_FINGERPRINTS: tuple[_Fingerprint, ...] = (
    _Fingerprint(
        revision="9c173fff3bcd",
        table="projects",
        column=None,
        description="initial schema",
    ),
    _Fingerprint(
        revision="03e058c707df",
        table=None,
        column=None,
        description=(
            "add observations_max_detections to project — column was "
            "dropped by d4e5f6a7b8c9, so its presence is no longer a "
            "reliable revision marker (live schema at head no longer "
            "has it)"
        ),
    ),
    _Fingerprint(
        revision="2540e6edbee2",
        table="deployments",
        column="warnings",
        description="add deployment warnings column",
    ),
    _Fingerprint(
        revision="a1b2c3d4e5f6",
        table=None,
        column=None,
        description=(
            "drop frame file rows — data-only DELETE, no schema-level "
            "signal to fingerprint"
        ),
    ),
    _Fingerprint(
        revision="b2c3d4e5f6a7",
        table=None,
        column=None,
        description=(
            "detection bbox nullable — column nullability change, "
            "not fingerprinted via PRAGMA table_info"
        ),
    ),
    _Fingerprint(
        revision="c3d4e5f6a7b8",
        table="projects",
        column="mode",
        description="add project mode and folder_run_state",
    ),
    _Fingerprint(
        revision="d4e5f6a7b8c9",
        table=None,
        column=None,
        description=(
            "drop observations_max_detections — column removal, "
            "not fingerprintable via PRAGMA table_info presence"
        ),
    ),
    _Fingerprint(
        revision="e5f6a7b8c9d0",
        table=None,
        column=None,
        description=(
            "drop folder-run 'run' step — JSON data-only UPDATE, no "
            "schema-level signal to fingerprint"
        ),
    ),
    _Fingerprint(
        revision="f6a7b8c9d0e1",
        table=None,
        column=None,
        description=(
            "drop folder-run 'folder' step — JSON data-only UPDATE, no "
            "schema-level signal to fingerprint"
        ),
    ),
    _Fingerprint(
        revision="a7b8c9d0e1f2",
        table=None,
        column=None,
        description=(
            "rename folder-run 'review' step to 'edit' — JSON data-only "
            "UPDATE, no schema-level signal to fingerprint"
        ),
    ),
    _Fingerprint(
        revision="b8c9d0e1f2a3",
        table=None,
        column=None,
        description=(
            "nullable capture dates on files / events — column "
            "nullability change, not fingerprinted via PRAGMA table_info"
        ),
    ),
    _Fingerprint(
        revision="c9d0e1f2a3b4",
        table="detections",
        column="common_name",
        description=(
            "rename display_name -> scientific_name and add common_name "
            "on detections / label_taxonomy"
        ),
    ),
    _Fingerprint(
        revision="d0e1f2a3b4c5",
        table="detections",
        column="suggestion_dismissed",
        description="add suggestion_dismissed flag on detections",
    ),
    _Fingerprint(
        revision="e1f2a3b4c5d6",
        table=None,
        column=None,
        description=(
            "make projects.timezone nullable — column nullability change, "
            "not fingerprinted via PRAGMA table_info"
        ),
    ),
    _Fingerprint(
        revision="f2a3b4c5d6e7",
        table="events",
        column="verified",
        description=(
            "add event counts (event_observations.human_count) and the "
            "event sign-off (events.verified); detectable via events.verified"
        ),
    ),
    _Fingerprint(
        revision="a3b4c5d6e7f8",
        table="events",
        column="confirmed",
        description=(
            "rename events.verified -> events.confirmed (the Observations "
            "page 'Confirm' action); detectable via events.confirmed"
        ),
    ),
    _Fingerprint(
        revision="b4c5d6e7f8a9",
        table="files",
        column="frames_processed",
        description=(
            "add files.frames_processed (analysed frame numbers from "
            "MegaDetector's process_video) so the recognition JSON can "
            "emit the MD format 1.6 video fields"
        ),
    ),
    _Fingerprint(
        revision="c5d6e7f8a9b0",
        table="projects",
        column="classification_gate",
        description=(
            "add projects.classification_gate (detection confidence "
            "above which crops are classified and embedded; MD itself "
            "now runs untresholded)"
        ),
    ),
    _Fingerprint(
        revision="d6e7f8a9b0c1",
        table="deployments",
        column="classification_gate_used",
        description=(
            "add deployments.classification_gate_used (per-run audit "
            "of the gate an analysis ran with, for mixed-gate projects)"
        ),
    ),
    _Fingerprint(
        revision="e7f8a9b0c1d2",
        table=None,
        column=None,
        description=(
            "drop projects.taxonomic_rollup_threshold (rollup threshold "
            "is fixed policy, not per-project) — column removal, not "
            "fingerprintable via PRAGMA table_info presence"
        ),
    ),
    _Fingerprint(
        revision="f8a9b0c1d2e3",
        table="projects",
        column="counting_threshold",
        description=(
            "rename projects.detection_threshold -> counting_threshold "
            "(named for its purpose, matching DEFAULT_COUNTING_THRESHOLD "
            "and classification_gate)"
        ),
    ),
)


def _resolve_backend_dir() -> Path:
    """Locate the backend root containing alembic.ini.

    Works in both dev (running from `backend/`) and PyInstaller bundle
    (alembic.ini and alembic/ are placed at `_MEIPASS` by backend.spec).
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent


def _alembic_config():
    """Build an Alembic Config wired to our DB and migration script dir."""
    from alembic.config import Config

    backend_dir = _resolve_backend_dir()
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    return cfg


def get_head_revision() -> str:
    """Return the head revision id from the on-disk migration scripts."""
    from alembic.script import ScriptDirectory

    script_dir = ScriptDirectory.from_config(_alembic_config())
    head = script_dir.get_current_head()
    if head is None:
        raise RuntimeError("No alembic head revision found on disk")
    return head


def get_current_revision(engine: Engine) -> str | None:
    """Return the current alembic_version row, or None if the table is missing."""
    if not inspect(engine).has_table("alembic_version"):
        return None
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def needs_upgrade(engine: Engine) -> bool:
    """True if the live DB is at a revision other than head."""
    return get_current_revision(engine) != get_head_revision()


def stamp(revision: str) -> None:
    """Mark the existing schema as being at `revision` without running migrations.

    Used by `init_db()` to bring `alembic_version` in line with the
    schema fingerprint detected on disk. Callers must have a strong
    reason to pick `revision`: stamping at the wrong revision causes
    `upgrade_to_head()` to either skip a migration the schema needs
    (silent column-missing crashes downstream) or re-run a migration
    the schema already has (DuplicateColumnError mid-startup).
    """
    from alembic import command

    command.stamp(_alembic_config(), revision)


def upgrade_to_head() -> None:
    """Run `alembic upgrade head` against the configured database."""
    from alembic import command

    command.upgrade(_alembic_config(), "head")


def _fingerprint_satisfied(fp: "_Fingerprint", engine: Engine) -> bool:
    """Is the live DB at or beyond the schema state this fingerprint marks?

    Non-detectable entries always return True: they cannot be checked
    via introspection, so we conservatively trust alembic_version on
    them. The detectable entries check column or table presence.
    """
    if not fp.is_detectable:
        return True
    inspector = inspect(engine)
    if fp.table not in inspector.get_table_names():
        return False
    if fp.column is None:
        return True  # Bootstrap: table existing is the whole check
    return any(c["name"] == fp.column for c in inspector.get_columns(fp.table))


def detect_schema_revision(engine: Engine) -> str | None:
    """Return the latest detectable revision whose schema matches the live DB.

    Walks `SCHEMA_FINGERPRINTS` newest -> oldest, skipping non-detectable
    entries (data-only or nullability migrations), and returns the
    first row whose signature column / table is present. Returns
    `None` when no user tables exist (a fresh install).

    This is one piece of the reconciliation: it tells us the latest
    revision the schema is known to be at. It does not by itself say
    whether alembic_version is lying — `reconcile_alembic_version`
    combines this with the recorded revision to decide.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    def _has_column(table: str, column: str) -> bool:
        if table not in existing_tables:
            return False
        return any(c["name"] == column for c in inspector.get_columns(table))

    for fp in reversed(SCHEMA_FINGERPRINTS):
        if not fp.is_detectable:
            continue
        if fp.column is None:
            if fp.table in existing_tables:
                return fp.revision
        elif _has_column(fp.table, fp.column):
            return fp.revision
    return None


def _alembic_version_is_truthful(
    engine: Engine, recorded_revision: str
) -> bool:
    """Verify that every detectable migration up to `recorded_revision` is applied.

    Walks `SCHEMA_FINGERPRINTS` in order. For every detectable entry
    at or before `recorded_revision`, the live schema must satisfy
    its fingerprint. As soon as a detectable fingerprint is missing,
    `alembic_version` is lying — alembic claims to be past a migration
    whose schema effect is not present in the DB.

    Returns True when no detectable migration up to and including
    `recorded_revision` is unsatisfied. Returns False when at least
    one detectable migration is missing from the live schema.
    """
    for fp in SCHEMA_FINGERPRINTS:
        if not fp.is_detectable:
            if fp.revision == recorded_revision:
                return True
            continue
        if not _fingerprint_satisfied(fp, engine):
            return False
        if fp.revision == recorded_revision:
            return True
    # `recorded_revision` is not a known revision. Treat as a lie
    # so reconcile re-stamps to a known-good state.
    logger.warning(
        f"alembic_version is {recorded_revision!r}, which is not listed in "
        f"SCHEMA_FINGERPRINTS. Treating as inconsistent."
    )
    return False


def reconcile_alembic_version(engine: Engine) -> str | None:
    """Force `alembic_version` to reflect the introspected schema.

    Returns the revision `alembic_version` ends up at, or None for an
    empty DB. Idempotent. Four cases:

    - Fresh install (no user tables): returns None. Caller lets
      `upgrade_to_head()` run all migrations from base.
    - Legacy DB without `alembic_version`: stamps at the detected
      revision and returns it. `upgrade_to_head()` then applies the
      rest.
    - alembic-managed DB whose recorded revision is consistent with
      the live schema: returns the recorded revision unchanged. This
      includes the common "alembic claims a non-detectable revision
      newer than the latest detectable one" case — e.g. a healthy DB
      at the latest beta whose head is a nullability migration. We
      do not re-stamp in that case because re-running the intervening
      migrations is at best wasteful and at worst destructive.
    - alembic-managed DB whose recorded revision is past a detectable
      migration the schema does NOT have: re-stamps at the detected
      revision and logs a WARNING. This is the "Cara case": a stamp
      that lies about the schema. `upgrade_to_head()` then applies
      the migrations that the lie skipped.
    """
    detected = detect_schema_revision(engine)
    current = get_current_revision(engine)

    if detected is None:
        # Empty DB. upgrade_to_head() will run everything.
        return None

    if current is None:
        logger.info(
            f"Legacy DB without alembic_version: stamping at {detected} "
            f"based on schema fingerprint"
        )
        stamp(detected)
        return detected

    if _alembic_version_is_truthful(engine, current):
        return current

    # Hard safety gate: if the stamped revision is not registered in
    # SCHEMA_FINGERPRINTS at all, refuse to auto-resolve. Silently
    # re-stamping backward and re-running the migration chain against
    # an unknown future revision destroyed user data on 2026-05-27
    # (the missing fingerprint was for b8c9d0e1f2a3). Surface a clear
    # error so the developer fixes the cause (add the fingerprint)
    # instead of letting the DB drift through a destructive rebuild.
    if not any(fp.revision == current for fp in SCHEMA_FINGERPRINTS):
        raise RuntimeError(
            f"alembic_version is stamped at {current!r}, which is not "
            f"registered in SCHEMA_FINGERPRINTS. Refusing to auto-"
            f"resolve: re-stamping and re-running migrations against "
            f"an unknown revision risks destroying data. Add a "
            f"_Fingerprint entry for {current!r} in "
            f"app/db/migrations.py and restart."
        )

    logger.warning(
        f"alembic_version says {current} but the schema is missing changes "
        f"from a migration in that chain. Re-stamping at {detected} "
        f"so pending migrations can run to reach head."
    )
    stamp(detected)
    return detected
