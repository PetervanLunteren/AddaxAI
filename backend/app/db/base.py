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
        # SQLite allows one writer at a time. Python's sqlite3 default gives
        # up after 5s, which turns any slow write (a large delete, a bulk
        # ingest) into "database is locked" errors on every other request
        # that happens to write during it. Wait instead of failing.
        connect_args={"timeout": 30},
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
    Bring the database up to date, or refuse to start.

    `alembic_version` is ground truth. We never guess which revision the
    schema is at and never re-stamp backwards to replay migrations: that
    replay re-ran one-time data migrations over already-migrated data
    and destroyed user verifications on 2026-05-27.

    Four steps:

    1. `ensure_upgradable` refuses a database alembic cannot be trusted
       to migrate (an early beta with no version row, an ambiguous
       version table).
    2. `alembic upgrade head` applies whatever is pending. On a fresh
       install that builds the whole schema from base.
    3. `schema_problems` checks the result against the models. Anything
       missing means the stamp lied, so we stop rather than guess.
    4. Seed builtin labels and refresh planner statistics.

    Raises `SchemaError` (message written for the end user) when the
    database cannot be used, and `RuntimeError` for anything else.
    Schema integrity is non-negotiable; the rest of the app assumes the
    models and the database agree.
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from app.db.migrations import (
        SchemaError,
        ensure_upgradable,
        get_head_revision,
        schema_mismatch_message,
        schema_problems,
        upgrade_to_head,
    )

    engine = get_engine()

    try:
        # Fail loudly if the alembic versions directory is missing.
        # Otherwise alembic.command.upgrade is a silent no-op and
        # init_db crashes later with a confusing "no such table" error.
        get_head_revision()

        ensure_upgradable(engine)

        logger.info("Running alembic upgrade head")
        upgrade_to_head()

        problems = schema_problems(engine)
        if problems:
            logger.critical(
                "Schema does not match the models after upgrading: "
                + "; ".join(problems)
            )
            raise SchemaError(schema_mismatch_message(problems))

        _seed_builtin_labels()
        with engine.connect() as conn:
            conn.execute(text("ANALYZE"))
            conn.commit()
        logger.info("Database initialized")
    except SchemaError:
        # Already carries a message written for the user. Wrapping it
        # would bury that under a developer-facing prefix.
        raise
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

