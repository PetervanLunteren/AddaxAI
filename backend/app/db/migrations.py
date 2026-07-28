"""
Programmatic alembic helpers used at app startup.

The contract is deliberately boring: **`alembic_version` is ground
truth**. `init_db()` runs `alembic upgrade head` and then checks the
result against `Base.metadata`. We never guess which revision a schema
is at, and we never re-stamp a database backwards to replay the chain.

An earlier design did both, using a hand-maintained fingerprint table of
marker columns. Guessing a revision by introspection cannot work in the
general case (drops, renames and data backfills leave no trace), it
needed a new row for every migration forever, and the recovery it drove
re-ran one-time data migrations over data that had already moved on.
That destroyed user verification work on 2026-05-27.

Three things make us refuse to start, all raising `SchemaError` with a
message written for the end user:

1. User tables exist but there is no `alembic_version` row. Alembic has
   run on every launch since 2026-05-08 (commit 78dc9d9c, which replaced
   `Base.metadata.create_all` plus a hand-rolled column patcher), so
   such a database is from an early beta.
2. `alembic_version` does not hold exactly one revision that exists on
   disk. An unknown revision means the file came from a different build;
   more than one row means the version table itself is corrupt.
3. After upgrading, the schema is missing something the models declare.
   That means the stamp lied, and the only safe answer is to stop and
   let the user restore a backup or start fresh.

Alembic imports are local to function bodies so test/cold-path callers
that only need `_resolve_backend_dir()` don't pay the import cost.

## Adding a new migration

Nothing to register anywhere. Write the migration and make sure
`test_upgrade_from_base_matches_models` still passes.

Do make it tolerant of the schema already being in its target state. A
database whose stamp is legitimately behind (a restored older backup,
for instance) replays the chain forward, which is correct and expected,
so guard DDL with a presence check. See DEVELOPERS.md.
"""

import sys
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class SchemaError(RuntimeError):
    """Startup refusal whose message is meant for the end user.

    `main.py` writes it to the startup error file so the Electron error
    page can show it verbatim. Keep the wording plain and always say
    what the user can do next.
    """


# Every refusal ends the same way, because the two things a user can do
# are the two buttons on the startup error page.
_WHAT_NOW = (
    "Your data has not been changed. To continue, restore a backup or "
    "start fresh with an empty database."
)

_EARLY_BETA_MESSAGE = (
    "This database is from an early AddaxAI beta and cannot be upgraded "
    f"to this version.\n\n{_WHAT_NOW}"
)

_FOREIGN_DB_MESSAGE = (
    "This database was made by a different version of AddaxAI, so this "
    f"version cannot open it.\n\n{_WHAT_NOW}"
)

# The error page has room for a handful of lines, not a schema dump. The
# full list always goes to the log.
_MAX_LISTED_PROBLEMS = 5


def schema_mismatch_message(problems: list[str]) -> str:
    """Build the refusal shown when the live schema is missing things."""
    shown = problems[:_MAX_LISTED_PROBLEMS]
    listed = "\n".join(f"  {p}" for p in shown)
    remaining = len(problems) - len(shown)
    if remaining:
        listed += f"\n  and {remaining} more"
    return (
        "The database does not match what this version of AddaxAI "
        f"expects, so it cannot be opened safely.\n\n{listed}\n\n"
        f"{_WHAT_NOW}"
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


def _version_rows(engine: Engine) -> list[str]:
    """Every row in `alembic_version`, or `[]` when the table is absent."""
    if not inspect(engine).has_table("alembic_version"):
        return []
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version_num FROM alembic_version"))
        return list(result.scalars())


def get_current_revision(engine: Engine) -> str | None:
    """Return the stamped revision, or None if the table is missing/empty."""
    rows = _version_rows(engine)
    return rows[0] if rows else None


def needs_upgrade(engine: Engine) -> bool:
    """True if the live DB is at a revision other than head."""
    return get_current_revision(engine) != get_head_revision()


def upgrade_to_head() -> None:
    """Run `alembic upgrade head` against the configured database.

    A stamped revision that is not on disk (an app downgrade, a
    hand-edited version row) makes alembic raise `CommandError` while
    resolving the chain, before any migration runs. Translate that into
    a refusal the user can act on rather than letting a stack trace be
    the whole story.
    """
    from alembic import command
    from alembic.util.exc import CommandError

    try:
        command.upgrade(_alembic_config(), "head")
    except CommandError as e:
        logger.critical(f"Alembic could not upgrade this database: {e}")
        raise SchemaError(_FOREIGN_DB_MESSAGE) from e


def _has_user_tables(engine: Engine) -> bool:
    """True if any non-alembic table exists (i.e. the DB is not empty)."""
    return bool(set(inspect(engine).get_table_names()) - {"alembic_version"})


def ensure_upgradable(engine: Engine) -> None:
    """Refuse a database alembic cannot be trusted to migrate.

    A fresh install (no tables at all) passes: `upgrade_to_head()` then
    builds the whole schema from base.
    """
    rows = _version_rows(engine)

    if not rows:
        if _has_user_tables(engine):
            logger.critical(
                "Database has user tables but no alembic_version row, so it "
                "predates the runtime alembic wiring (2026-05-08). Refusing."
            )
            raise SchemaError(_EARLY_BETA_MESSAGE)
        return

    if len(rows) > 1:
        logger.critical(
            f"alembic_version holds {len(rows)} rows ({rows}), so the "
            f"recorded revision is ambiguous. Refusing."
        )
        raise SchemaError(_FOREIGN_DB_MESSAGE)


# compare_metadata reports both directions. We want one rule only:
# anything the models declare that the database lacks. The remove_* ops
# (a column the live DB has and the models don't) are harmless, and the
# modify_* ops cover nullability, column types and server defaults,
# where SQLite's loose typing makes a false alarm likelier than the
# skipped migration it would catch. A false alarm refuses a healthy
# user's launch, so the bar is high.
_ADDITIVE_OPS = frozenset(
    {"add_table", "add_column", "add_index", "add_constraint"}
)


def _describe_missing(op: tuple) -> str | None:
    """Render one additive compare_metadata op, or None if not additive."""
    kind = op[0]
    if kind not in _ADDITIVE_OPS:
        return None
    if kind == "add_table":
        return f"missing table {op[1].name}"
    if kind == "add_column":
        # ("add_column", schema, table_name, Column)
        return f"missing column {op[2]}.{op[3].name}"
    if kind == "add_index":
        return f"missing index {op[1].name} on {op[1].table.name}"
    # ("add_constraint", Constraint)
    name = getattr(op[1], "name", None)
    table = getattr(getattr(op[1], "table", None), "name", "?")
    return f"missing constraint {name or op[1]} on {table}"


def schema_problems(engine: Engine) -> list[str]:
    """Everything `Base.metadata` declares that the live schema lacks.

    An empty list means the schema matches. Three callers share it, and
    that is the point: the startup check, the CI guard that pins the
    migration chain to the models, and the diagnostic report all ask the
    same question of the same source of truth.

    Built on alembic's own `compare_metadata` (CONVENTIONS rule 13),
    filtered to the additive operations. That built-in does not report
    foreign key ON DELETE actions, so those get their own walk: the app
    relies on the database to cascade deletes (`passive_deletes=True`
    everywhere), which makes a lost ON DELETE CASCADE the one way this
    design can orphan or lose rows.
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from app.db.base import Base

    problems: list[str] = []

    try:
        with engine.connect() as conn:
            diffs = compare_metadata(
                MigrationContext.configure(conn), Base.metadata
            )
    except Exception as e:
        # compare_metadata reflects the whole schema, so it raises on a
        # schema too broken to read at all (a foreign key pointing at a
        # table that no longer exists, for instance). That is itself the
        # answer to the question being asked, so report it as a problem
        # rather than letting it escape as an unexplained crash.
        problems.append(f"the schema could not be read ({e})")
        diffs = []
    for diff in diffs:
        # A diff is one operation tuple, or a list of them for changes
        # alembic groups together.
        for op in diff if isinstance(diff, list) else [diff]:
            described = _describe_missing(op)
            if described:
                problems.append(described)

    inspector = inspect(engine)
    live_tables = set(inspector.get_table_names())
    for table_name, table in Base.metadata.tables.items():
        if table_name not in live_tables:
            continue  # already reported as a missing table
        live = {
            (col, (fk["options"].get("ondelete") or "").upper())
            for fk in inspector.get_foreign_keys(table_name)
            for col in fk["constrained_columns"]
        }
        for fk in table.foreign_keys:
            if (fk.parent.name, (fk.ondelete or "").upper()) not in live:
                problems.append(
                    f"foreign key {table_name}.{fk.parent.name} is missing "
                    f"ON DELETE {fk.ondelete or '(no action)'}"
                )
    return problems
