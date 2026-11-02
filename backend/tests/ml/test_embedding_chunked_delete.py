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

    # Force several chunks so the loop is actually exercised. The chunk size
    # now lives in the shared iter_id_chunks default, so pin a tiny size on the
    # name embedding_utils calls.
    import functools

    from app.db.sql_params import iter_id_chunks

    monkeypatch.setattr(
        embedding_utils,
        "iter_id_chunks",
        functools.partial(iter_id_chunks, size=3),
    )

    inserted = embedding_utils.save_embeddings_to_db(
        npz_path, job.id, model_id, dim, db
    )

    assert inserted == n
    # Old n deleted across chunks, new n inserted -> exactly n, not 2n.
    assert db.query(DetectionEmbedding).count() == n


def test_build_embedding_input_applies_classification_gate(db, tmp_path):
    """Detections below the classification gate are not embedded (MD now
    runs untresholded, the gate bounds the per-crop model work), but a
    verified detection always embeds regardless of confidence."""
    from app.ml.embedding_utils import build_embedding_input
    from tests.conftest import make_deployment, make_detection, make_file, make_project

    project = make_project(db, name="emb-gate")
    dep = make_deployment(db, project_id=project.id)
    src = tmp_path / "IMG.jpg"
    src.write_bytes(b"x")
    file = make_file(
        db, deployment_id=dep.id, file_path=str(src), observation_type="animal"
    )
    common = dict(
        file_id=file.id, category="animal",
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )
    d_pass = make_detection(db, confidence=0.9, **common)
    make_detection(db, confidence=0.02, **common)
    d_verified = make_detection(db, confidence=0.02, verified=True, **common)

    result = build_embedding_input(dep.id, db, min_confidence=0.1)
    ids = {e["detection_id"] for e in result["detections"]}
    assert ids == {d_pass.id, d_verified.id}


def test_build_embedding_input_loads_only_a_tracks_card(db, tmp_path):
    """A clip's boxes come back from the database only on their track's
    representative frame: the card is the one box with a picture on disk
    (the track's crop), and the other frames never leave the database.
    A box without a track has no card and stays behind too. On a BRUV
    deployment of 238,941 boxes the old query loaded every one of them
    and the server was killed twice (2026-11-02); the input is 11,696
    cards."""
    from app.ml.embedding_utils import build_embedding_input
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
        make_track,
    )

    project = make_project(db, name="emb-cards")
    dep = make_deployment(db, project_id=project.id)
    # The crop is not written: the input lists the card anyway, because
    # the existence check moved to the embedder (one stat per card on an
    # external disk froze the backend; the embedder skips what it cannot
    # open and says so).
    crop = tmp_path / "track.jpg"
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/clip.mp4",
    )
    track = make_track(db, file_id=video.id, crop_path=str(crop))
    box = dict(
        file_id=video.id, category="animal", confidence=0.9,
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )
    make_detection(db, frame_number=30, track_id=track.id, **box)
    card = make_detection(db, frame_number=60, track_id=track.id, **box)
    make_detection(db, frame_number=90, track_id=track.id, **box)
    make_detection(db, frame_number=60, **box)  # no track: no card

    result = build_embedding_input(dep.id, db, min_confidence=0.1)

    assert [e["detection_id"] for e in result["detections"]] == [card.id]
    assert result["detections"][0]["image_path"] == str(crop)
