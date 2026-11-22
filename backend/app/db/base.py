"""
Database configuration and session management.

Following DEVELOPERS.md principles:
- Explicit configuration
- Type hints everywhere
- Crash early if database cannot be initialized
"""

import hashlib
import threading
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings
from app.core.logging_config import get_logger, log_duration

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

    **No `PRAGMA optimize` here.** There was one, and it did nothing except
    harm. It only analyses tables the connection has already queried, and
    this runs before the connection has issued a single statement, so on
    SQLite 3.45 it was a no-op every time. On 3.50 and later it does run,
    and then it rewrites the one statistic `refresh_query_statistics` exists
    to delete, undoing that fix on the very next connection. Statistics are
    refreshed after every migration and after every analysis, embedding and
    postprocessing job, which covers every bulk change to the data.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # CRITICAL: Enable FK constraints
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.close()
    dbapi_conn.create_function("seeded_hash", 2, _seeded_hash, deterministic=True)


# One engine and one session factory per process, so connections are
# pooled. A new engine per request meant a new SQLite connection per
# request: the pragmas set again, the schema parsed again and an empty
# 64 MB page cache every time. Rebuilt only when the database URL changes,
# which in the app never happens and in the tests happens per test
# (`tests/db/conftest.py` points the settings at a fresh directory).
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    """
    Return the process-wide database engine, built on first use.

    Crashes if database URL is invalid or database cannot be accessed.
    """
    global _engine, _session_factory
    settings = get_settings()
    with _engine_lock:
        if _engine is not None and str(_engine.url) == settings.database_url:
            return _engine
        if _engine is not None:
            _engine.dispose()
        _engine = create_engine(
            settings.database_url,
            echo=settings.sql_echo,  # Verbose per-query logging; off by default
            future=True,  # Use SQLAlchemy 2.0 style
            # SQLite allows one writer at a time. Python's sqlite3 default
            # gives up after 5s, which turns any slow write (a large
            # delete, a bulk ingest) into "database is locked" errors on
            # every other request that happens to write during it. Wait
            # instead of failing.
            connect_args={"timeout": 30},
            # No cap on open connections. Five stay pooled; beyond that a
            # session gets its own connection, closed when it ends. The
            # default cap of 15 made the 16th concurrent session wait 30 s
            # and fail, and slow requests a user has navigated away from
            # keep holding theirs.
            max_overflow=-1,
        )
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine,
            class_=Session,
        )
        return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the session factory bound to the process-wide engine."""
    get_engine()
    assert _session_factory is not None
    return _session_factory


# The indexes whose ANALYZE statistic has to be thrown away again.
#
# `files.source_video_id` is the only self-referencing foreign key in the
# schema (frame rows point at the video they came from) and it declares
# ON DELETE CASCADE, so SQLite consults `idx_files_source_video` once for
# every `files` row it deletes.
#
# ANALYZE writes one row per index whether anything uses it or not. For
# this one it writes "<n> <n>", because the column is NULL on every file
# that is not a video frame, so a lookup "matches" the whole table. SQLite
# 3.45, which the packaged build carries, reads that and stops using the
# index: the per-row cascade lookup becomes a full scan of all 380k index
# entries, once per deleted row. Deleting a project of 369,298 files went
# from 4 seconds to an estimated 6.5 hours that way, and deleting a
# deployment, deleting a folder run and re-running one all pay the same.
#
# Dropping the statistic costs nothing. No query searches by
# `source_video_id`: the two places that read it either project it
# (`crud/label_tree.py`) or join by `files.id` (`joinedload(File.source_video)`),
# and EXPLAIN QUERY PLAN gives both the same plan with the row and without
# it. A query that did search by it is *helped*: with the row present the
# planner answers `SCAN files`, without it `SEARCH files USING INDEX
# idx_files_source_video`.
#
# `detections.track_id` has the same shape: NULL on every box that is not
# from a video, so its row reads "<n> <n>" too, and 3.45 answers the
# verify lookup (`id IN (...) OR track_id IN (...)`, which reaches every
# box of a track) with `SCAN detections`. That was 2.2 s per statement on
# a 2.5M-box project, twice per verify. Without the row it is two index
# seeks.
_UNUSABLE_STAT_INDEXES = ("idx_files_source_video", "idx_detections_track")


def refresh_query_statistics(conn: Any) -> None:
    """
    Re-run ANALYZE, then drop the statistics that must not exist.

    Accepts a `Connection` or a `Session`. The caller commits, because
    every call site is already inside a transaction it owns.

    Never call bare `ANALYZE` instead of this, and never add a
    `PRAGMA optimize` anywhere: both write the rows straight back. See
    `_UNUSABLE_STAT_INDEXES` above for what that costs.
    """
    # Timed because it runs after every migration and every job, and its
    # cost grows with the whole database, not with the change.
    with log_duration(logger, "ANALYZE"):
        conn.execute(text("ANALYZE"))
    _drop_unusable_statistic(conn)


def _drop_unusable_statistic(conn: Any) -> None:
    """Delete the `_UNUSABLE_STAT_INDEXES` rows, and start every later
    session on a connection that has not read them."""
    for idx in _UNUSABLE_STAT_INDEXES:
        conn.execute(text("DELETE FROM sqlite_stat1 WHERE idx = :idx"), {"idx": idx})
    _replace_pooled_connections()


def _replace_pooled_connections() -> None:
    """Make every later session start on a new connection.

    SQLite reads the statistics once per connection, when it first loads
    the schema, and never again. Measured on 3.45.1: another connection
    keeps planning with the old statistics after an ANALYZE, and the
    connection that ran the ANALYZE keeps the deleted
    `idx_files_source_video` row in memory, so a project delete on it would
    be the full-scan cascade again. Disposing the engine closes the idle
    connections now and each busy one when its session ends.
    """
    if _engine is not None:
        _engine.dispose()


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
    4. Seed builtin labels, and refresh planner statistics when a
       migration ran or there are none. Otherwise the statistics from the
       last job still hold: `ANALYZE` reads the whole database (34 s on a
       4.6 GB one) and used to run before every launch could serve.

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
        needs_upgrade,
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
        migrating = needs_upgrade(engine)

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
            if migrating or not _has_query_statistics(conn):
                refresh_query_statistics(conn)
                conn.commit()
            else:
                # Cheap, and catches a row an older version left behind.
                _drop_unusable_statistic(conn)
                conn.commit()
                logger.info("Query statistics kept, no migration ran")
        logger.info("Database initialized")
    except SchemaError:
        # Already carries a message written for the user. Wrapping it
        # would bury that under a developer-facing prefix.
        raise
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize database: {e}") from e


def _has_query_statistics(conn: Any) -> bool:
    """True when `ANALYZE` has written statistics to this database."""
    has_table = conn.execute(
        text("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'")
    ).first()
    if has_table is None:
        return False
    return conn.execute(text("SELECT 1 FROM sqlite_stat1 LIMIT 1")).first() is not None


def _seed_builtin_labels() -> None:
    """Seed the classic detector classes (animal, person, vehicle) in
    label_taxonomy on startup, so the label picker offers them before
    anything was analysed. Other classes get their row on first use."""
    from app.ml.taxonomy_db import CLASSIC_DETECTOR_CLASSES, ensure_builtin_labels

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        ensure_builtin_labels(db, CLASSIC_DETECTOR_CLASSES)
    finally:
        db.close()

