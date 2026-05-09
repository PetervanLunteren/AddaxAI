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

from sqlalchemy import create_engine, event, inspect, text
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
        echo=settings.debug,  # Log SQL queries in debug mode
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

    - Fresh DB or DB already managed by alembic: run upgrade head (no-op
      when already at head, applies pending migrations otherwise).
    - Legacy DB with the head schema but no alembic_version table
      (beta-tester DBs that predate runtime alembic wiring): stamp head.
      Their schema is already at head because the previous startup path
      ran Base.metadata.create_all + a hand-rolled column migrator on
      every launch.

    Crashes if migrations fail.
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from app.db.migrations import get_head_revision, stamp_head, upgrade_to_head

    engine = get_engine()

    try:
        inspector = inspect(engine)
        has_user_tables = inspector.has_table("projects")
        has_alembic = inspector.has_table("alembic_version")

        # Fail loudly if the alembic versions directory is missing.
        # Otherwise alembic.command.upgrade is a silent no-op and
        # init_db crashes later with a confusing "no such table" error.
        get_head_revision()

        if has_user_tables and not has_alembic:
            logger.info("Legacy DB without alembic_version: stamping head")
            stamp_head()
        else:
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

