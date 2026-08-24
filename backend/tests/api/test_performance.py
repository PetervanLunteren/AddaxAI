"""
Tests for the classification performance endpoint
(confusion matrix + classification report).

1. CRUD tests on get_classification_performance with synthetic data
   covering ground truth vs prediction pairs, taxonomy rollup, the
   "Higher-level taxa" and "No taxonomy" buckets, detector-only
   projects, top-N collapse, and the skipped counters.
2. HTTP endpoint shape via FastAPI TestClient.
"""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.api.crud import performance as performance_crud
from app.ml.taxonomic_rank import HIGHER_LEVEL_TAXA, NO_TAXONOMY
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
    scientific_name: str | None = None,
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
        scientific_name=scientific_name,
    )
    db.add(row)
    db.flush()
    return row


def _bootstrap_classified_project(db: Session, model_id: str = "CLS-TEST-v1"):
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
        scientific_name="P. pardus",
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
        scientific_name="Lynx lynx",
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
        scientific_name="C. capreolus",
    )
    return project, site, deployment, f


# ---------------------------------------------------------------------------
# 1. CRUD layer
# ---------------------------------------------------------------------------


def test_species_matrix_counts_verified_pairs(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)

    # 3 true leopards: 2 predicted leopard, 1 predicted lynx
    for predicted in ("leopard", "leopard", "lynx"):
        make_detection(
            db, file_id=f.id,
            label="leopard", original_label=predicted, verified=True,
        )
    # 1 true lynx correctly predicted
    make_detection(
        db, file_id=f.id,
        label="lynx", original_label="lynx", verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    assert resp.has_classifier is True
    # Species rank shows the binomial built from the rank columns, not
    # the row's own scientific_name (which names the leaf and can sit
    # below species on a variant row).
    assert set(resp.classes) == {"P. pardus", "L. lynx"}
    i_leo = resp.classes.index("P. pardus")
    i_lynx = resp.classes.index("L. lynx")
    assert resp.matrix[i_leo][i_leo] == 2
    assert resp.matrix[i_leo][i_lynx] == 1
    assert resp.matrix[i_lynx][i_lynx] == 1
    assert resp.grand_total == 4
    assert resp.skipped_unverified == 0
    assert resp.skipped_no_prediction == 0


def test_most_specific_default_uses_raw_labels(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(
        db, file_id=f.id,
        label="leopard", scientific_name="P. pardus",
        original_label="leopard", verified=True,
    )

    # Default taxonomic_rank is "all"
    resp = performance_crud.get_classification_performance(db, project.id, top_n=None)
    # scientific_name wins in "all" mode, so the class is the pretty label
    assert resp.classes == ["P. pardus"]
    assert resp.grand_total == 1


def test_matrix_rolls_up_to_family(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label="lynx", verified=True)
    make_detection(db, file_id=f.id, label="lynx", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="deer", original_label="deer", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="family", top_n=None,
    )
    # Family names are stored lowercase in taxon_family but displayed
    # capitalised (convention; matches the dashboard after the rank
    # normalisation).
    assert set(resp.classes) == {"Felidae", "Cervidae"}
    i_fel = resp.classes.index("Felidae")
    i_cerv = resp.classes.index("Cervidae")
    assert resp.matrix[i_fel][i_fel] == 2
    assert resp.matrix[i_cerv][i_cerv] == 1
    assert resp.grand_total == 3


def test_rollup_row_at_species_buckets_into_higher_level_taxa(db: Session) -> None:
    # Postprocessing taxonomic rollup writes a family-level entry into
    # label_taxonomy (taxon_species is NULL). At rank=species the matrix
    # must bucket it into "Higher-level taxa", not show it as a pseudo-
    # species row — this was the user-visible issue that triggered the
    # whole alignment with the dashboard.
    project, _site, _dep, f = _bootstrap_classified_project(db)
    _mk_taxonomy_row(
        db,
        model_id="CLS-TEST-v1",
        name="Felidae",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="Felidae",
        level="family",
        scientific_name="Felidae",
    )
    # Current label is a family-level rollup; original is a species.
    make_detection(
        db, file_id=f.id,
        label="Felidae", original_label="leopard", verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    # Row (current) is the rollup → "Higher-level taxa"
    # Col (prediction) resolves via taxonomy → "P. pardus"
    assert HIGHER_LEVEL_TAXA in resp.classes
    assert "P. pardus" in resp.classes
    i_higher = resp.classes.index(HIGHER_LEVEL_TAXA)
    i_leo = resp.classes.index("P. pardus")
    assert resp.matrix[i_higher][i_leo] == 1
    assert resp.grand_total == 1


def test_label_with_no_taxonomy_row_buckets_into_no_taxonomy(db: Session) -> None:
    # A detection with a label that has no matching LabelTaxonomy row
    # (custom label, stale label from a previous model, etc.) lands in
    # the "No taxonomy" bucket at specific ranks.
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(
        db, file_id=f.id,
        label="mystery_label", original_label="mystery_label", verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    assert NO_TAXONOMY in resp.classes
    i = resp.classes.index(NO_TAXONOMY)
    assert resp.matrix[i][i] == 1


def test_unverified_detections_are_counted_but_not_used(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=False)
    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    assert resp.skipped_unverified == 1
    assert resp.grand_total == 1


def test_null_original_label_counts_as_skipped_no_prediction(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label=None, verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    assert resp.skipped_no_prediction == 1
    assert resp.grand_total == 1


def test_person_and_vehicle_detections_are_included(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(
        db, file_id=f.id, category="person",
        label=None, original_label=None, verified=True,
    )
    make_detection(
        db, file_id=f.id, category="vehicle",
        label=None, original_label=None, verified=True,
    )
    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    assert "person" in resp.classes
    assert "vehicle" in resp.classes
    i_p = resp.classes.index("person")
    i_v = resp.classes.index("vehicle")
    assert resp.matrix[i_p][i_p] == 1
    assert resp.matrix[i_v][i_v] == 1
    assert resp.skipped_no_prediction == 0


def test_detector_only_project_at_all_rank_shows_category_matrix(db: Session) -> None:
    # Detector-only: unclassified animal + person + vehicle.
    # At "all" (Most specific) the animal row renders as the category.
    project = make_project(db, classification_model_id=None)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)

    make_detection(
        db, file_id=f.id, category="animal",
        label=None, original_label=None, verified=True,
    )
    make_detection(
        db, file_id=f.id, category="person",
        label=None, original_label=None, verified=True,
    )
    make_detection(
        db, file_id=f.id, category="vehicle",
        label=None, original_label=None, verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="all", top_n=None,
    )
    assert resp.has_classifier is False
    assert resp.classes == ["animal", "person", "vehicle"]
    assert resp.matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert resp.skipped_no_prediction == 0


def test_detector_only_animal_at_species_rank_becomes_no_taxonomy(db: Session) -> None:
    # Same detector-only fixture but asking at rank=species.
    # Animals have no taxonomy at all → "No taxonomy"; person / vehicle
    # remain on their own category regardless of rank.
    project = make_project(db, classification_model_id=None)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)

    make_detection(
        db, file_id=f.id, category="animal",
        label=None, original_label=None, verified=True,
    )
    make_detection(
        db, file_id=f.id, category="person",
        label=None, original_label=None, verified=True,
    )

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    assert NO_TAXONOMY in resp.classes
    assert "person" in resp.classes


def test_precision_recall_f1_on_tiny_fixture(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    make_detection(db, file_id=f.id, label="leopard", original_label="lynx", verified=True)
    make_detection(db, file_id=f.id, label="lynx", original_label="lynx", verified=True)
    make_detection(db, file_id=f.id, label="lynx", original_label="lynx", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=None,
    )
    by_name = {m.class_name: m for m in resp.per_class}
    leo = by_name["P. pardus"]
    lynx = by_name["L. lynx"]

    assert leo.support == 3
    assert leo.precision == 1.0
    assert leo.recall == 2 / 3
    assert abs(leo.f1 - 0.8) < 1e-9

    assert lynx.support == 2
    assert lynx.precision == 2 / 3
    assert lynx.recall == 1.0
    assert abs(lynx.f1 - 0.8) < 1e-9

    assert abs(resp.macro_f1 - 0.8) < 1e-9
    assert abs(resp.weighted_f1 - 0.8) < 1e-9


def test_top_n_collapses_tail_into_other(db: Session) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    for _ in range(5):
        make_detection(db, file_id=f.id, label="leopard", original_label="leopard", verified=True)
    for _ in range(3):
        make_detection(db, file_id=f.id, label="lynx", original_label="lynx", verified=True)
    for _ in range(2):
        make_detection(db, file_id=f.id, label="deer", original_label="deer", verified=True)

    resp = performance_crud.get_classification_performance(
        db, project.id, taxonomic_rank="species", top_n=2,
    )
    assert resp.other_bucket_present is True
    assert "other" in resp.classes
    assert resp.grand_total == 10
    assert "P. pardus" in resp.classes
    assert "L. lynx" in resp.classes
    assert "C. capreolus" not in resp.classes


def test_site_and_date_filters(db: Session) -> None:
    project = make_project(db, classification_model_id="CLS-TEST-v1")
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

    resp = performance_crud.get_classification_performance(
        db, project.id, site_ids=[site_a.id], taxonomic_rank="species", top_n=None,
    )
    assert resp.grand_total == 2

    from datetime import date

    resp = performance_crud.get_classification_performance(
        db, project.id,
        date_from=date(2024, 1, 1),
        date_to=date(2024, 3, 1),
        taxonomic_rank="species",
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
        params={
            "project_id": project.id,
            "taxonomic_rank": "species",
            "top_n": "20",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["taxonomic_rank"] == "species"
    assert body["classes"] == ["P. pardus"]
    assert body["matrix"] == [[1]]
    assert body["has_classifier"] is True
    assert body["per_class"][0]["class_name"] == "P. pardus"


def test_performance_endpoint_default_rank_is_most_specific(db: Session, client) -> None:
    project, _site, _dep, f = _bootstrap_classified_project(db)
    make_detection(
        db, file_id=f.id,
        label="leopard", scientific_name="P. pardus",
        original_label="leopard", verified=True,
    )
    db.commit()

    # No taxonomic_rank passed — should default to "all".
    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": project.id, "top_n": "all"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["taxonomic_rank"] == "all"
    assert body["top_n_applied"] is None


def test_performance_endpoint_bad_rank_returns_422(db: Session, client) -> None:
    project = make_project(db)
    db.commit()
    resp = client.get(
        "/api/statistics/performance",
        params={"project_id": project.id, "taxonomic_rank": "kingdom"},
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


def test_off_best_frame_boxes_are_out_of_scope(db: Session) -> None:
    """A video's boxes live on every sampled frame, but only the best frame
    is written to disk, so the rest have no picture to open and can never be
    verified. Counting them inflated the denominator: the footer read "1
    verified detection of 220 ... 218 not yet verified" over a grid holding
    32. Same reasoning as the sub-threshold gate beside it."""
    project, _site, deployment, _f = _bootstrap_classified_project(db)
    video = make_file(
        db,
        deployment_id=deployment.id,
        file_type="video",
        file_format="mp4",
        best_frame_number=3,
    )
    # One verified box on the visible frame, two unreachable ones beside it.
    make_detection(
        db, file_id=video.id, confidence=0.9, label="leopard",
        original_label="leopard", frame_number=3, verified=True,
    )
    for frame in (7, 11):
        make_detection(
            db, file_id=video.id, confidence=0.9, label="lynx",
            original_label="lynx", frame_number=frame,
        )
    db.flush()

    result = performance_crud.get_classification_performance(db, project.id)
    # The two unreachable boxes must not be reported as work still to do.
    assert result.skipped_unverified == 0
    # The one box on the visible frame still counts.
    assert result.grand_total == 1
