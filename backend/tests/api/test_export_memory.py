"""Exports keep flat memory, whatever the size of the project.

The table exports stream: rows are read from the database in batches and
written to the file as they come. Before, every table was collected in
full first, and a 3.2-million-box project ran the backend out of memory
(2026-10-14). This test exports the same project at two sizes and checks
that the peak does not grow with it.

It goes through the export endpoint rather than the builders, so the same
test runs unchanged against older code and shows the difference there.
Marked slow: it inserts and exports a few hundred thousand rows under
``tracemalloc``.
"""

from __future__ import annotations

import tempfile
import tracemalloc
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import insert

from app.models import Detection, File
from tests.conftest import make_deployment, make_project, make_site

pytestmark = pytest.mark.slow

BOXES_PER_FILE = 10


def _seed(db, n_files: int):
    project = make_project(db, timezone="UTC", counting_threshold=0.1)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id, folder_path="/data/cam")
    start = datetime(2024, 1, 1)
    files = []
    boxes = []
    for i in range(n_files):
        fid = str(uuid.uuid4())
        files.append(
            {
                "id": fid,
                "deployment_id": deployment.id,
                "file_path": f"/data/cam/{i:07d}.jpg",
                "file_type": "image",
                "file_format": "jpg",
                "captured_at_local": start + timedelta(seconds=i),
                "observation_type": "animal",
            }
        )
        for _ in range(BOXES_PER_FILE):
            boxes.append(
                {
                    "id": str(uuid.uuid4()),
                    "file_id": fid,
                    "category": "animal",
                    "confidence": 0.9,
                    "bbox_x": 0.1,
                    "bbox_y": 0.1,
                    "bbox_width": 0.2,
                    "bbox_height": 0.2,
                    "label": "red fox",
                    "label_confidence": 0.8,
                    "frame_number": None,
                }
            )
    db.execute(insert(File), files)
    db.execute(insert(Detection), boxes)
    db.commit()
    return project


def _export_peak_bytes(client, project_id: str) -> int:
    folder = tempfile.mkdtemp()
    tracemalloc.start()
    try:
        resp = client.post(
            f"/api/projects/{project_id}/export/tables",
            json={"output_dir": folder, "format": "csv"},
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert resp.status_code == 200, resp.text
    return peak


def test_export_memory_does_not_grow_with_the_project(client, db):
    small = _seed(db, 2_000)
    large = _seed(db, 8_000)

    peak_small = _export_peak_bytes(client, small.id)
    peak_large = _export_peak_bytes(client, large.id)

    # Four times the rows. Collected tables grow about four times too;
    # streamed ones stay within noise of the small export.
    assert peak_large < 1.5 * peak_small, (
        f"peak {peak_small / 1e6:.1f} MB for 20k boxes, "
        f"{peak_large / 1e6:.1f} MB for 80k boxes"
    )
