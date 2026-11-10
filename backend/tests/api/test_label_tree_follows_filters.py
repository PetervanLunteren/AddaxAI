"""The label filter tree counts what the page would show.

A tester saw "Macropus (20 detections)" in the tree beside an empty grid:
all twenty boxes were verified and the grid's Verified filter sat on
"Unverified", which the tree knew nothing about. The count beside a label
now follows every other filter of the page the tree is on, computed by
that page's own query, so the two cannot disagree. The label filter
itself is left out of the count (ticking a label must not change the
number beside it), and a label the filters leave nothing of stays in the
tree at 0, so it can still be picked under another filter.

One parity test per page: the tree's number for a leaf equals what the
page lists with that leaf ticked alone, under the same filters.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.crud import event as event_crud
from app.api.crud import file as file_crud
from app.api.crud.label_tree import build_label_filter_tree
from app.api.schemas.label import LabelFilters
from app.db.base import Base
from app.models import File
from app.models.event import event_files
from app.models.label_taxonomy import LabelTaxonomy
from app.services.label_service import count_detections_by_label
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
)

# do_sort imports its sibling `observation_sort` by bare name (it runs as
# a subprocess in production, where its own dir is on sys.path[0]).
_INFERENCE_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))

MODEL_ID = "EUR-DF-v1-3"


def _taxon(db, name: str) -> LabelTaxonomy:
    row = LabelTaxonomy(
        classification_model_id=MODEL_ID,
        name=name,
        level="species",
        scientific_name=name,
        taxon_class="mammalia",
        taxon_genus=name.split()[0],
        taxon_species=name.split()[-1],
    )
    db.add(row)
    db.flush()
    return row


def _leaf_counts(tree: dict) -> dict[str, int]:
    """Leaf id -> count, walking the built tree."""
    out: dict[str, int] = {}

    def walk(nodes):
        for n in nodes:
            if n.get("children"):
                walk(n["children"])
            else:
                out[n["id"]] = n["count"]

    walk(tree["tree"])
    return out


def _seed(db):
    """A project with two species, in two events, on files with a mix of
    verified boxes, verified files, flagged files and one weak box."""
    project = make_project(db, classification_model_id=MODEL_ID)
    dep = make_deployment(db, project_id=project.id)
    fox = _taxon(db, "vulpes vulpes")
    deer = _taxon(db, "cervus elaphus")
    files = []
    for i in range(6):
        ev = make_event_with_files(
            db, deployment_id=dep.id, event_start_local=datetime(2024, 6, 1 + i, 12)
        )
        link = db.execute(
            event_files.select().where(event_files.c.event_id == ev.id)
        ).first()
        files.append(db.get(File, link.file_id))
    # fox: 3 verified boxes on flagged files, 1 unverified; deer: 2 unverified,
    # one of them below the counting threshold (invisible unless verified).
    for f in files[:3]:
        f.flagged = True
        f.verified = True
        make_detection(db, file_id=f.id, label=fox.name, label_taxonomy_id=fox.id,
                       label_confidence=0.9, verified=True)
    make_detection(db, file_id=files[3].id, label=fox.name, label_taxonomy_id=fox.id,
                   label_confidence=0.9)
    make_detection(db, file_id=files[4].id, label=deer.name, label_taxonomy_id=deer.id,
                   label_confidence=0.9)
    make_detection(db, file_id=files[5].id, label=deer.name, label_taxonomy_id=deer.id,
                   label_confidence=0.9, confidence=0.05)
    db.commit()
    return project, fox, deer


def test_a_fully_verified_label_reads_zero_under_unverified_and_stays(db):
    """The reported case: 20 verified boxes, filter on Unverified."""
    project = make_project(db, classification_model_id=MODEL_ID)
    dep = make_deployment(db, project_id=project.id)
    roo = _taxon(db, "macropus species")
    for _ in range(20):
        f = make_file(db, deployment_id=dep.id)
        make_detection(db, file_id=f.id, label=roo.name, label_taxonomy_id=roo.id,
                       label_confidence=0.9, verified=True)
    db.commit()

    unverified = build_label_filter_tree(
        project.id, db, count_by="detection", verification="unverified"
    )
    everything = build_label_filter_tree(project.id, db, count_by="detection")

    assert roo.id in unverified["all_leaf_ids"]
    assert _leaf_counts(unverified)[roo.id] == 0
    assert _leaf_counts(everything)[roo.id] == 20


def test_file_counts_match_the_files_tab(db):
    project, fox, deer = _seed(db)
    # A detector range too: the Files tab keeps its floor at the project
    # threshold whatever the slider says, and the tree must do the same.
    filters = dict(
        verification="unverified", flagged="not_flagged", min_confidence=0.1
    )

    tree = build_label_filter_tree(project.id, db, count_by="file", **filters)

    for leaf, count in _leaf_counts(tree).items():
        total, _, _ = file_crud.get_labels_files(
            db, project.id, floor=project.counting_threshold, labels=[leaf], **filters
        )
        assert count == total, leaf
    assert _leaf_counts(tree) == {fox.id: 1, deer.id: 1}


def test_event_counts_match_the_counts_page(db):
    project, fox, deer = _seed(db)
    filters = dict(flagged="flagged")

    tree = build_label_filter_tree(project.id, db, count_by="event", **filters)

    for leaf, count in _leaf_counts(tree).items():
        assert count == event_crud.get_event_count_by_project(
            db, project.id, labels=[leaf],
            project_floor=project.counting_threshold, **filters
        ), leaf
    assert _leaf_counts(tree) == {fox.id: 3, deer.id: 0}


def test_show_only_empty_reads_zero_for_every_label(db):
    """The Counts page ignores the label filter under "show only empty"
    (an empty event has no species), so the honest count is 0, not the
    number of empty events."""
    project, fox, deer = _seed(db)
    tree = build_label_filter_tree(project.id, db, count_by="event", empty="show_only")
    assert _leaf_counts(tree) == {fox.id: 0, deer.id: 0}


@pytest.fixture
def file_db(tmp_path):
    """A file-based SQLite DB: the grid's sort worker opens the DB by path."""
    path = tmp_path / "grid.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield str(path), session
    finally:
        session.close()
        engine.dispose()


def test_detection_counts_match_the_labels_grid(file_db):
    from app.ml.inference.similarity_script import do_sort

    path, session = file_db
    project, fox, deer = _seed(session)
    filters = LabelFilters(verified=False, flagged="not_flagged")

    counts = count_detections_by_label(session, project.id, filters)

    for leaf in (fox.id, deer.id):
        grid = do_sort(
            path,
            project.id,
            {
                "filters": {
                    "labels": [leaf],
                    "verified": False,
                    "flagged": "not_flagged",
                    "project_floor": project.counting_threshold,
                },
                "sort": "events",
            },
        )
        assert counts.get(leaf, 0) == grid["total_matching"], leaf
    assert counts == {fox.id: 1, deer.id: 1}


def test_the_endpoint_takes_the_page_filters(client, db):
    project, fox, deer = _seed(db)

    resp = client.get(
        f"/api/events/label-tree?project_id={project.id}"
        "&count_by=detection&verification=verified"
    )

    assert resp.status_code == 200
    assert _leaf_counts(resp.json()) == {fox.id: 3, deer.id: 0}


def test_a_one_day_range_reaches_the_grid(file_db):
    """End to end through the grid's own SQL: the day's boxes are found
    with the range set to that single day."""
    from app.ml.inference.similarity_script import do_sort
    from app.services.label_service import _apply_project_threshold, _filters_to_dict

    path, session = file_db
    project, fox, deer = _seed(session)
    # The day as the Labels page sends it: a date, parsed into a datetime
    # at midnight by LabelFilters, serialised for the worker by the
    # service, the same path the request takes.
    filters = LabelFilters(date_from=datetime(2024, 6, 3), date_to=datetime(2024, 6, 3))

    grid = do_sort(
        path, project.id,
        {"filters": _filters_to_dict(_apply_project_threshold(filters, project.id, session)),
         "sort": "events"},
    )
    counts = count_detections_by_label(session, project.id, filters)

    assert grid["total_matching"] == 1
    assert counts == {fox.id: 1}
