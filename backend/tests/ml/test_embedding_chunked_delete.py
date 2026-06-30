"""save_embeddings_to_db chunks its delete so a large id list doesn't blow
SQLite's bound-parameter limit ("too many SQL variables", Simon's crash).

The crash needs >32766 ids to reproduce on a modern SQLite, which is too many
for a fast test. Instead we shrink the chunk size and assert the chunked delete
still removes every pre-existing embedding across chunk boundaries.
"""

import uuid

import numpy as np

from app.ml import embedding_utils
from app.models.detection_embedding import DetectionEmbedding


def test_chunked_delete_removes_all_existing(db, tmp_path, monkeypatch):
    from tests.conftest import (
        make_deployment,
        make_file,
        make_job,
        make_project,
        make_site,
    )

    project = make_project(db, name="Emb")
    site = make_site(db, project_id=project.id, name="S")
    dep = make_deployment(db, project_id=project.id, site_id=site.id)
    f = make_file(db, deployment_id=dep.id)
    job = make_job(db)

    model_id = "DINOV2-VITB14"
    dim = 4
    n = 10  # > the shrunk chunk size below, so multiple chunks run

    from app.models import Detection

    det_ids: list[str] = []
    for _ in range(n):
        d = Detection(
            id=str(uuid.uuid4()),
            file_id=f.id,
            category="animal",
            confidence=0.9,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_width=0.2,
            bbox_height=0.2,
        )
        db.add(d)
        db.flush()
        det_ids.append(d.id)
        # Pre-existing embedding for the same (detection, model).
        db.add(
            DetectionEmbedding(
                id=str(uuid.uuid4()),
                detection_id=d.id,
                job_id=job.id,
                embedding_model_id=model_id,
                vector=np.zeros(dim, dtype=np.float16).tobytes(),
                dimension=dim,
                l2_norm=0.0,
            )
        )
    db.commit()
    assert db.query(DetectionEmbedding).count() == n

    # npz keyed by detection_id -> float16 vector, as the worker writes it.
    npz_path = tmp_path / "emb.npz"
    np.savez(
        npz_path,
        **{did: np.ones(dim, dtype=np.float16) for did in det_ids},
    )

    # Force several chunks so the loop is actually exercised.
    monkeypatch.setattr(embedding_utils, "_SQL_VAR_CHUNK", 3)

    inserted = embedding_utils.save_embeddings_to_db(
        npz_path, job.id, model_id, dim, db
    )

    assert inserted == n
    # Old n deleted across chunks, new n inserted -> exactly n, not 2n.
    assert db.query(DetectionEmbedding).count() == n
