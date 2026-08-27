"""
Alembic environment configuration.

Following DEVELOPERS.md principles:
- Explicit configuration (no defaults)
- Crash early if config missing
"""

import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.core.config import get_settings
from app.db.base import Base

# Import all models so Alembic can detect them
from app.models import (  # noqa: F401
    AuditLog,
    Deployment,
    Detection,
    DetectionEmbedding,
    Event,
    EventObservation,
    File,
    Job,
    LabelTaxonomy,
    Project,
    Site,
    event_files,
)

# Alembic Config object
config = context.config

# Only apply alembic.ini's logging config when nothing else has wired
# up the root logger yet, i.e. the dev `alembic` CLI. Inside the running
# app, setup_logging() has already attached a RotatingFileHandler to
# the root logger; fileConfig would replace it with alembic.ini's
# stderr-only console handler via [logger_root], leaving the rest of
# the process with no file log. disable_existing_loggers=False on its
# own does not prevent this, because the root logger is listed in
# [loggers] and gets reconfigured regardless of that flag.
if (
    config.config_file_name is not None
    and not logging.getLogger().handlers
):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Get database URL from application settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        raise RuntimeError("Database URL not configured")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Foreign keys OFF for the whole run, as alembic's SQLite docs
        # require. `set_sqlite_pragma` in db/base.py turns them on for
        # every engine, this one included, and batch mode rebuilds a
        # table with CREATE, copy, DROP TABLE, RENAME. With foreign keys
        # on, DROP TABLE runs an implicit DELETE FROM first and every
        # ON DELETE CASCADE fires: rebuilding `projects` emptied sites,
        # deployments, files and detections in the test that found this
        # (tests/db/test_migration_keeps_rows.py). On the raw DBAPI
        # connection: the pragma is a no-op inside a transaction, and
        # going through SQLAlchemy would open one.
        connection.connection.dbapi_connection.execute("PRAGMA foreign_keys=OFF")
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
