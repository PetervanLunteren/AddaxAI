"""
Programmatic alembic helpers used at app startup.

`init_db()` runs `upgrade_to_head()` on every launch. On a fresh install
that builds the schema from scratch; on an existing DB it applies any
pending migrations and is a no-op when already at head. Legacy beta-
tester databases that pre-date the runtime alembic wiring are detected
in `init_db()` and stamped at head with `stamp_head()` instead, because
their schema is already at head (built by Base.metadata.create_all and
the now-removed _migrate_missing_columns running on every prior boot).

Alembic imports are local to function bodies so test/cold-path callers
that only need `_resolve_backend_dir()` don't pay the import cost.
"""

import sys
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings


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


def stamp_head() -> None:
    """Mark the existing schema as being at head without running migrations."""
    from alembic import command

    command.stamp(_alembic_config(), "head")


def upgrade_to_head() -> None:
    """Run `alembic upgrade head` against the configured database."""
    from alembic import command

    command.upgrade(_alembic_config(), "head")
