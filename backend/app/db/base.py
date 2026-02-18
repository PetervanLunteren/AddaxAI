"""
Database configuration and session management.

Following DEVELOPERS.md principles:
- Explicit configuration
- Type hints everywhere
- Crash early if database cannot be initialized
"""

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


# Enable Write-Ahead Logging for SQLite (allows concurrent reads during writes)
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
    """
    Set SQLite performance and concurrency settings.

    foreign_keys: Enable foreign key constraints (SQLite doesn't enforce by default!)
    WAL mode: Allows concurrent reads during writes
    NORMAL synchronous: Safe with WAL, faster than FULL
    64MB cache: Better performance for large queries
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # CRITICAL: Enable FK constraints
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.close()


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
    Initialize database - create all tables.

    Crashes if database cannot be initialized.
    Called on application startup.
    """
    from app.models import audit_log, deployment, detection, event, file, job, project, site  # noqa: F401

    engine = get_engine()

    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        _migrate_missing_columns(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize database: {e}") from e


def _migrate_missing_columns(engine: Engine) -> None:
    """Add any missing columns to existing tables (lightweight migration)."""
    migrations: list[tuple[str, str, str]] = [
        # (table, column, SQL type + default)
        ("projects", "shortcut_labels", "JSON NOT NULL DEFAULT '{}'"),
    ]
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, col_type in migrations:
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing:
                logger.info(f"Migrating: adding {table}.{column}")
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))

