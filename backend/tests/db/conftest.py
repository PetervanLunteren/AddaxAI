"""Shared fixtures and helpers for the database tests.

The tests in this package are different from the rest of the suite:
they run the real alembic chain against a real SQLite file, and some of
them write rows into *historical* schemas. The ORM factories in
`tests/conftest.py` are no use for that, because they describe the
schema at head. So the helpers here work in raw SQL against whatever
the schema happens to be at the revision under test.
"""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import Connection, create_engine, text

from app.core.config import Settings


@pytest.fixture()
def isolated_db_settings(tmp_path: Path, monkeypatch):
    """Point get_settings() at a fresh empty user-data dir.

    Each test gets its own SQLite file so init_db() can run end-to-end
    without colliding with the developer's real `~/AddaxAI/addaxai.db`.
    """
    db_path = tmp_path / "addaxai.db"
    settings = Settings(
        user_data_dir=tmp_path,
        database_url=f"sqlite:///{db_path}",
    )

    def _get_settings() -> Settings:
        return settings

    monkeypatch.setattr("app.core.config.get_settings", _get_settings)
    monkeypatch.setattr("app.db.base.get_settings", _get_settings)
    monkeypatch.setattr("app.db.migrations.get_settings", _get_settings)

    yield settings


@pytest.fixture()
def engine(isolated_db_settings: Settings):
    """An engine on the test database, disposed after the test."""
    eng = create_engine(isolated_db_settings.database_url, future=True)
    yield eng
    eng.dispose()


def upgrade_to(revision: str) -> None:
    """Run the chain up to `revision` and stop there.

    Lets a test stand the database up in the exact state a migration was
    written to consume. Test-local on purpose: the app only ever
    upgrades to head, so there is no reason to expose a partial upgrade
    in `app.db.migrations`.
    """
    from alembic import command
    from app.db.migrations import _alembic_config

    command.upgrade(_alembic_config(), revision)


def _placeholder(column: str, declared_type: str) -> object:
    """A harmless value for a required column the test does not care about."""
    if column == "id":
        return str(uuid.uuid4())
    t = declared_type.upper()
    if "JSON" in t:
        return "{}"
    if "DATE" in t or "TIME" in t:
        return "2026-01-01 00:00:00"
    if "INT" in t or "BOOL" in t:
        return 0
    if any(k in t for k in ("REAL", "FLOA", "DOUB", "NUM")):
        return 0.0
    return ""


def insert_row(conn: Connection, table: str, **values: object) -> str:
    """Insert one row into `table`, filling in what the caller left out.

    Every column that is NOT NULL, has no default, and was not named by
    the caller gets a placeholder. That is what keeps these tests
    readable: `projects` alone has 16 required columns at the revisions
    they seed at, and the required set differs from one revision to the
    next, so spelling them all out would mean a revision-specific column
    list in every test.

    Foreign keys are deliberately *not* filled in. `PRAGMA foreign_keys`
    is on, so a missing parent fails loudly and the test has to seed a
    real one, which is the honest thing for a migration test to do.

    Returns the row's `id`, generated when the caller did not pass one.
    """
    columns = conn.execute(text(f"PRAGMA table_info({table})")).mappings().all()
    row = dict(values)
    for col in columns:
        name = col["name"]
        if name in row:
            continue
        if col["notnull"] and col["dflt_value"] is None:
            row[name] = _placeholder(name, col["type"])

    names = ", ".join(row)
    binds = ", ".join(f":{n}" for n in row)
    conn.execute(text(f"INSERT INTO {table} ({names}) VALUES ({binds})"), row)
    return str(row.get("id", ""))


def seed_deployment(conn: Connection) -> tuple[str, str]:
    """A project and a deployment: the parents most other rows need."""
    project_id = insert_row(conn, "projects", name="Test project")
    deployment_id = insert_row(conn, "deployments", project_id=project_id)
    return project_id, deployment_id
