"""
Tests for the classification performance endpoint
(confusion matrix + classification report).

Three layers:
1. CRUD tests on get_classification_performance with synthetic data,
   covering verified-only ground truth, taxonomy rollup, top-N collapse,
   detector-only projects, and the skipped counters.
2. HTTP endpoint shape via FastAPI TestClient.
"""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.api.crud import performance as performance_crud
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import make_deployment, make_detection, make_file, make_project, make_site


def _mk_taxonomy_row(
    db: Session,
    *,
    model_id: str,
    name: str,
    taxon_class: str | None = None,
    taxon_order: str | None = None,
    taxon_family: str | None = None,
    taxon_genus: str | None = None,
    taxon_species: str | None = None,
    level: str = "species",
    display_name: str | None = None,
) -> LabelTaxonomy:
    row = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id=model_id,
        name=name,
        taxon_class=taxon_class,
        taxon_order=taxon_order,
        taxon_family=taxon_family,
        taxon_genus=taxon_genus,
        taxon_species=taxon_species,
        level=level,
        display_name=display_name,
    )
    db.add(row)
    db.flush()
    return row


def _bootstrap_classified_project(db: Session, model_id: str = "CLS-TEST-v1"):
    """Project with a classifier and a two-species taxonomy (leopard,
    lynx) plus the builtin animal/person/vehicle rows the real JSON
    pipeline would populate. Returns (project, site, deployment, file)."""
    project = make_project(db, classification_model_id=model_id)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)

    _mk_taxonomy_row(
        db,
        model_id=model_id,
        name="leopard",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="felidae",
        taxon_genus="panthera",
        taxon_species="pardus",
        level="species",
        display_name="P. pardus",
    )
    _mk_taxonomy_row(
        db,
        model_id=model_id,
        name="lynx",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="felidae",
        taxon_genus="lynx",
        taxon_species="lynx",
        level="species",
        display_name="Lynx lynx",
    )
    _mk_taxonomy_row(
        db,
        model_id=model_id,
        name="deer",
        taxon_class="mammalia",
        taxon_order="artiodactyla",
        taxon_family="cervidae",
        taxon_genus="capreolus",
        taxon_species="capreolus",
        level="species",
        display_name="C. capreolus",
    )
    return project, site, deployment, f


# ---------------------------------------------------------------------------
# 1. CRUD layer
# ---------------------------------------------------------------------------


def test_species_matrix_counts_verified_pairs(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)

    # 3 true leopards: 2 predicted leopard, 1 predicted lynx (confusion)
    for predicted in ("leopard", "leopard", "lynx"):
        make_detection(
            db,
            file_id=f.id,
            label="leopard",
            original_label=predicted,
            verified=True,
        )
    # 1 true lynx correctly predicted
    make_detection(
        db,
        file_id=f.id,
        label="lynx",
        original_label="lynx",
        verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=None,
    )
    assert resp.has_classifier is True
    assert set(resp.classes) == {"leopard", "lynx"}
    i_leo = resp.classes.index("leopard")
    i_lynx = resp.classes.index("lynx")
    # leopard row: 2 TP, 1 confused as lynx
    assert resp.matrix[i_leo][i_leo] == 2
    assert resp.matrix[i_leo][i_lynx] == 1
    # lynx row: 1 TP
    assert resp.matrix[i_lynx][i_lynx] == 1
    assert resp.grand_total == 4
    assert resp.skipped_unverified == 0
    assert resp.skipped_no_prediction == 0


def test_matrix_rolls_up_to_family(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)

    # Two felids confused with each other still roll up to the same
    # family cell.
    make_detection(db, file_id=f.id, label="leopard", original_label="lynx", verified=True)
    make_detection(db, file_id=f.id, label="lynx", original_label="leopard", verified=True)
    # A deer correctly classified
    make_detection(db, file_id=f.id, label="deer", original_label="deer", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, rank="family", top_n=None,
    )
    assert set(resp.classes) == {"felidae", "cervidae"}
    i_fel = resp.classes.index("felidae")
    i_cerv = resp.classes.index("cervidae")
    assert resp.matrix[i_fel][i_fel] == 2  # both felid confusions collapse to the diagonal
    assert resp.matrix[i_cerv][i_cerv] == 1
    assert resp.grand_total == 3


def test_unverified_detections_are_counted_but_not_used(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=False)
    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=None,
    )
    assert resp.skipped_unverified == 1
    assert resp.grand_total == 1


def test_null_original_label_counts_as_skipped_no_prediction(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    # Verified animal with no original_label (pre-migration data)
    make_detection(db, file_id=f.id, label="leopard", original_label=None, verified=True)
    # A well-formed one alongside so we know the endpoint still produces a matrix
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=None,
    )
    assert resp.skipped_no_prediction == 1
    assert resp.grand_total == 1


def test_person_and_vehicle_detections_are_included(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    # Person / vehicle detections never get classified; both original and
    # current class should resolve to the category.
    make_detection(
        db,
        file_id=f.id,
        category="person",
        label=None,
        original_label=None,
        verified=True,
    )
    make_detection(
        db,
        file_id=f.id,
        category="vehicle",
        label=None,
        original_label=None,
        verified=True,
    )
    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=None,
    )
    assert "person" in resp.classes
    assert "vehicle" in resp.classes
    # both should land on their own diagonal
    i_p = resp.classes.index("person")
    i_v = resp.classes.index("vehicle")
    assert resp.matrix[i_p][i_p] == 1
    assert resp.matrix[i_v][i_v] == 1
    assert resp.skipped_no_prediction == 0


def test_detector_only_project_renders_category_matrix(db: Session) -> None:
    # No classifier configured → all detections have original_label NULL,
    # but person / vehicle categories and animal category still populate
    # a 3-class matrix.
    project = make_project(db, classification_model_id=None)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)

    # User verified an animal detection as-is (no relabel).
    make_detection(
        db, file_id=f.id, category="animal",
        label=None, original_label=None, verified=True,
    )
    # And a person + vehicle.
    make_detection(
        db, file_id=f.id, category="person",
        label=None, original_label=None, verified=True,
    )
    make_detection(
        db, file_id=f.id, category="vehicle",
        label=None, original_label=None, verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=None,
    )
    assert resp.has_classifier is False
    assert resp.classes == ["animal", "person", "vehicle"]
    # everything on the diagonal: category == category
    assert resp.matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert resp.skipped_no_prediction == 0


def test_precision_recall_f1_on_tiny_fixture(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    # 3 leopards (2 correctly, 1 confused as lynx)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="lynx", verified=True)
    # 2 lynx (both correct)
    make_detection(db, file_id=f.id, label="lynx", original_label="lynx", verified=True)
    make_detection(db, file_id=f.id, label="lynx", original_label="lynx", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=None,
    )
    # Match by class_name
    by_name = {m.class_name: m for m in resp.per_class}
    leo = by_name["leopard"]
    lynx = by_name["lynx"]

    # leopard: support=3, TP=2, col_total=2 → precision=1.0, recall=2/3
    assert leo.support == 3
    assert leo.precision == 1.0
    assert leo.recall == 2 / 3
    # f1 = 2 * 1 * (2/3) / (1 + 2/3) = 0.8
    assert abs(leo.f1 - 0.8) < 1e-9

    # lynx: support=2, TP=2, col_total=3 → precision=2/3, recall=1.0
    assert lynx.support == 2
    assert lynx.precision == 2 / 3
    assert lynx.recall == 1.0
    assert abs(lynx.f1 - 0.8) < 1e-9

    # macro F1 = (0.8 + 0.8) / 2
    assert abs(resp.macro_f1 - 0.8) < 1e-9
    # weighted F1 = (0.8 * 3 + 0.8 * 2) / 5 = 0.8
    assert abs(resp.weighted_f1 - 0.8) < 1e-9


def test_top_n_collapses_tail_into_other(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    # leopard (5), lynx (3), deer (2)
    for _ in range(5):
        make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    for _ in range(3):
        make_detection(db, file_id=f.id, label="lynx", original_label="lynx", verified=True)
    for _ in range(2):
        make_detection(db, file_id=f.id, label="deer", original_label="deer", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, rank="species", top_n=2,
    )
    assert resp.other_bucket_present is True
    assert "other" in resp.classes
    assert resp.grand_total == 10
    # leopard and lynx remain; deer folds into "other"
    assert "leopard" in resp.classes
    assert "lynx" in resp.classes
    assert "deer" not in resp.classes


def test_site_and_date_filters(db: Session) -> None:
    project = make_project(db, classification_model_id="CLS-TEST-v1")
    # Two sites, two deployments, different dates.
    site_a = make_site(db, project_id=project.id)
    site_b = make_site(db, project_id=project.id)
    dep_a = make_deployment(db, site_id=site_a.id)
    dep_b = make_deployment(db, site_id=site_b.id)

    f_a_early = make_file(
        db, deployment_id=dep_a.id, captured_at_local=datetime(2024, 1, 10),
    )
    f_a_late = make_file(
        db, deployment_id=dep_a.id, captured_at_local=datetime(2024, 6, 10),
    )
    f_b_early = make_file(
        db, deployment_id=dep_b.id, captured_at_local=datetime(2024, 1, 10),
    )

    _mk_taxonomy_row(
        db,
        model_id="CLS-TEST-v1",
        name="leopard",
        taxon_class="mammalia",
        taxon_family="felidae",
        taxon_genus="panthera",
    )

    make_detection(
        db, file_id=f_a_early.id,
        label="leopard", original_label="leopard", verified=True,
    )
    make_detection(
        db, file_id=f_a_late.id,
        label="leopard", original_label="leopard", verified=True,
    )
    make_detection(
        db, file_id=f_b_early.id,
        label="leopard", original_label="leopard", verified=True,
    )

    # Only site_a
    resp = performance_crud.get_classification_performance(
        db, project.id, site_ids=[site_a.id], rank="species", top_n=None,
    )
    assert resp.grand_total == 2

    # Date clip — only early files
    from datetime import date

    resp = performance_crud.get_classification_performance(
        db,
        project.id,
        date_from=date(2024, 1, 1),
        date_to=date(2024, 3, 1),
        rank="species",
        top_n=None,
    )
    assert resp.grand_total == 2


# ---------------------------------------------------------------------------
# 2. HTTP endpoint shape
# ---------------------------------------------------------------------------


def test_performance_endpoint_returns_expected_shape(db: Session, client) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    db.commit()

    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": project.id, "rank": "species", "top_n": "20"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rank"] == "species"
    assert body["classes"] == ["leopard"]
    assert body["matrix"] == [[1]]
    assert body["has_classifier"] is True
    assert body["per_class"][0]["class_name"] == "leopard"


def test_performance_endpoint_top_n_all(db: Session, client) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    db.commit()

    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": project.id, "top_n": "all"},
    )
    assert resp.status_code == 200
    assert resp.json()["top_n_applied"] is None


def test_performance_endpoint_bad_rank_returns_422(db: Session, client) -> None:
    project = make_project(db)
    db.commit()
    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": project.id, "rank": "kingdom"},
    )
    assert resp.status_code == 422


def test_performance_endpoint_bad_top_n_returns_422(db: Session, client) -> None:
    project = make_project(db)
    db.commit()
    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": project.id, "top_n": "nonsense"},
    )
    assert resp.status_code == 422


def test_performance_endpoint_missing_project_returns_404(db: Session, client) -> None:
    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": "no-such-project"},
    )
    assert resp.status_code == 404
