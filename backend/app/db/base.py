"""
Database configuration and session management.

Following DEVELOPERS.md principles:
- Explicit configuration
- Type hints everywhere
- Crash early if database cannot be initialized
"""

import hashlib
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


def _seeded_hash(text_value: str | None, seed: int | None) -> int:
    """Deterministic hash of (text, seed) used for seeded random ordering.

    Python's built-in `hash()` is salted per process (PYTHONHASHSEED) so it
    cannot be used. md5 gives a stable, process-independent integer. Signed
    to fit SQLite's 64-bit signed INTEGER (unsigned would overflow).
    """
    if text_value is None or seed is None:
        return 0
    digest = hashlib.md5(f"{text_value}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


# Enable Write-Ahead Logging for SQLite (allows concurrent reads during writes)
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
    """
    Set SQLite performance and concurrency settings + register UDFs.

    foreign_keys: Enable foreign key constraints (SQLite doesn't enforce by default!)
    WAL mode: Allows concurrent reads during writes
    NORMAL synchronous: Safe with WAL, faster than FULL
    64MB cache: Better performance for large queries
    seeded_hash UDF: deterministic shuffle for the verify-tab Random sort.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # CRITICAL: Enable FK constraints
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.execute("PRAGMA optimize")  # Auto-ANALYZE when planner stats are stale
    cursor.close()
    dbapi_conn.create_function("seeded_hash", 2, _seeded_hash, deterministic=True)


def get_engine() -> Engine:
    """
    Create database engine.

    Crashes if database URL is invalid or database cannot be accessed.
    """
    settings = get_settings()

    engine = create_engine(
        settings.database_url,
        echo=settings.sql_echo,  # Verbose per-query logging; off by default
        future=True,  # Use SQLAlchemy 2.0 style
    )

    return engine


def get_session_factory() -> sessionmaker[Session]:
    """Create session factory for database operations."""
    engine = get_engine()
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        class_=Session,
    )


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints to get database session.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database via Alembic migrations.

    The reconciliation is fingerprint-based: introspect the live
    schema, decide which revision it actually matches, stamp that
    revision if `alembic_version` disagrees, then run upgrade_to_head
    to apply any pending migrations. Three cases collapse into one
    code path:

    - Fresh install: `reconcile_alembic_version` sees no user tables,
      returns None, leaves `alembic_version` empty. `upgrade_to_head`
      runs every migration from base.
    - Legacy DB without `alembic_version`: stamps at the detected
      revision. `upgrade_to_head` applies anything newer.
    - Existing DB with a stored revision that disagrees with the
      live schema (e.g. a historical stamp_head bug, a half-applied
      migration on power loss, a hand-restored backup): re-stamps at
      the detected revision and logs a WARNING. `upgrade_to_head`
      then applies the migrations the recorded revision skipped.

    Crashes if migrations fail. Schema integrity is non-negotiable;
    the rest of the app assumes the model and the DB agree.
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from app.db.migrations import (
        get_head_revision,
        reconcile_alembic_version,
        upgrade_to_head,
    )

    engine = get_engine()

    try:
        # Fail loudly if the alembic versions directory is missing.
        # Otherwise alembic.command.upgrade is a silent no-op and
        # init_db crashes later with a confusing "no such table" error.
        get_head_revision()

        reconcile_alembic_version(engine)

        logger.info("Running alembic upgrade head")
        upgrade_to_head()

        _seed_builtin_labels()
        with engine.connect() as conn:
            conn.execute(text("ANALYZE"))
            conn.commit()
        logger.info("Database initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize database: {e}") from e


def _seed_builtin_labels() -> None:
    """Seed builtin labels (person, vehicle) in label_taxonomy on startup."""
    from app.ml.taxonomy_db import ensure_builtin_labels

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        ensure_builtin_labels(db)
    finally:
        db.close()

