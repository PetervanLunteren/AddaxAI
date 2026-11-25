"""Audit timestamps come back from the database as UTC, not naive."""

from datetime import UTC

from app.db.base import Base
from app.db.types import UtcDateTime
from tests.conftest import make_project


def test_every_utc_column_reads_back_as_utc():
    """SQLite drops the timezone, so a plain DateTime column returns a
    naive value and the API sends it without an offset. The browser then
    reads a UTC time as local: the recent-runs list was an hour off in
    Amsterdam and ten in Sydney."""
    columns = [
        column
        for table in Base.metadata.tables.values()
        for column in table.columns
        if column.name.endswith("_utc")
    ]
    assert columns
    assert [c for c in columns if not isinstance(c.type, UtcDateTime)] == []


def test_a_stored_time_is_utc_when_read_and_when_sent(client, db, tmp_path):
    from app.models import DeploymentQueue

    project = make_project(db, mode="folder_run", folder_run_state={"step": "setup"})
    db.add(DeploymentQueue(project_id=project.id, folder_path=str(tmp_path)))
    db.commit()
    db.expire_all()

    assert project.updated_at_utc.tzinfo is UTC
    sent = client.get("/api/folder-runs").json()[0]["updated_at_utc"]
    assert sent.endswith(("Z", "+00:00"))
