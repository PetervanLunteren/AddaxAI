"""Tests for get_species_distribution (dashboard bars + species pickers).

Covers the wildlife_only filter behind the dashboard "Wildlife
detected" chart, and the date-range filter.
"""

from datetime import datetime

from app.api.crud import statistics as stats_crud
from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_project,
    make_site,
)


def _add_observation(
    db,
    *,
    event_id: str,
    label: str,
    max_n: int = 1,
    category: str = "animal",
) -> EventObservation:
    obs = EventObservation(
        event_id=event_id,
        label=label,
        category=category,
        max_n=max_n,
    )
    db.add(obs)
    db.flush()
    return obs


def _build_fixture(db):
    """One project, one deployment, four events across two months.

    Observations mix wildlife (leopard, lion) with every non-wildlife
    source the dashboard complaint covered: a person-category row (from
    the detector), classifier labels "human" and "vehicle", and a
    user-marked "false detection".
    """
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev1 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    ev2 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 5, 14, 0)
    )
    ev3 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 2, 3, 9, 0)
    )
    ev4 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 2, 7, 18, 0)
    )

    _add_observation(db, event_id=ev1.id, label="leopard", max_n=2)
    _add_observation(db, event_id=ev2.id, label="leopard", max_n=3)
    _add_observation(db, event_id=ev3.id, label="lion")
    # Detector category rows: label falls back to the category string.
    _add_observation(db, event_id=ev1.id, label="person", category="person")
    _add_observation(db, event_id=ev2.id, label="vehicle", category="vehicle")
    # Classifier classes that are real detections but not wildlife.
    _add_observation(db, event_id=ev3.id, label="human")
    _add_observation(db, event_id=ev4.id, label="vehicle")
    # A user-marked false detection (verified, so it passed every
    # threshold on its way into the observations).
    _add_observation(db, event_id=ev4.id, label="false detection")

    db.flush()
    return project


def _species_names(rows) -> set[str]:
    return {r.species for r in rows}


def test_unfiltered_returns_all_labels(db):
    """Without the flag the pickers keep every observed label."""
    project = _build_fixture(db)

    rows = stats_crud.get_species_distribution(db, project.id)

    assert _species_names(rows) == {
        "leopard",
        "lion",
        "person",
        "vehicle",
        "human",
        "false detection",
    }


def test_wildlife_only_drops_non_wildlife(db):
    """The flag drops detector person/vehicle categories, classifier
    human/vehicle labels, and false-detection markings in one pass."""
    project = _build_fixture(db)

    rows = stats_crud.get_species_distribution(db, project.id, wildlife_only=True)

    assert _species_names(rows) == {"leopard", "lion"}


def test_wildlife_only_label_match_is_case_insensitive(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    _add_observation(db, event_id=ev.id, label="Human")
    _add_observation(db, event_id=ev.id, label="False Detection")
    _add_observation(db, event_id=ev.id, label="leopard")

    rows = stats_crud.get_species_distribution(db, project.id, wildlife_only=True)

    assert _species_names(rows) == {"leopard"}


def test_wildlife_only_keeps_non_megadetector_categories(db):
    """A detector with its own vocabulary (shark/fish) is wildlife;
    only person/vehicle categories are excluded."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    _add_observation(db, event_id=ev.id, label="shark", category="shark")

    rows = stats_crud.get_species_distribution(db, project.id, wildlife_only=True)

    assert _species_names(rows) == {"shark"}


def test_event_count_mode_counts_events_per_label(db):
    project = _build_fixture(db)

    rows = stats_crud.get_species_distribution(db, project.id, wildlife_only=True)

    counts = {r.species: r.count for r in rows}
    assert counts == {"leopard": 2, "lion": 1}


def test_max_n_mode_sums_counts(db):
    project = _build_fixture(db)

    rows = stats_crud.get_species_distribution(
        db, project.id, count_mode="max_n", wildlife_only=True
    )

    counts = {r.species: r.count for r in rows}
    assert counts == {"leopard": 5, "lion": 1}


def test_date_range_filters_events(db):
    """date_from / date_to restrict which events are counted; date_to
    is inclusive of the whole end day."""
    project = _build_fixture(db)

    january = stats_crud.get_species_distribution(
        db, project.id, date_from="2024-01-01", date_to="2024-01-31"
    )
    assert {r.species: r.count for r in january} == {
        "leopard": 2,
        "person": 1,
        "vehicle": 1,
    }

    february = stats_crud.get_species_distribution(
        db, project.id, date_from="2024-02-01", date_to="2024-02-07"
    )
    assert {r.species: r.count for r in february} == {
        "lion": 1,
        "human": 1,
        "vehicle": 1,
        "false detection": 1,
    }


def _add_variant_taxonomy(db, model_id: str):
    """Two variant rows of one species, as populate_taxonomy_from_csv
    writes them for a variant model."""
    from app.models.label_taxonomy import LabelTaxonomy

    rows = {}
    for variant in ("adult", "juvenile"):
        row = LabelTaxonomy(
            classification_model_id=model_id,
            name=f"red fox {variant}",
            taxon_class="mammalia",
            taxon_order="carnivora",
            taxon_family="canidae",
            taxon_genus="vulpes",
            taxon_species="vulpes",
            taxon_variant=variant,
            level="variant",
            common_name=f"Red fox {variant}",
            scientific_name=f"V. vulpes ({variant})",
        )
        db.add(row)
        db.flush()
        rows[variant] = row
    return rows


def test_variants_stay_separate_at_most_specific(db):
    """Adult and juvenile fox are distinct classes; the qualified
    scientific name keeps them apart in "Most specific" views."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    taxonomy = _add_variant_taxonomy(db, "ARC-v1")

    ev1 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    ev2 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 3, 9, 0)
    )
    for ev, variant in ((ev1, "adult"), (ev2, "juvenile")):
        obs = _add_observation(db, event_id=ev.id, label=f"red fox {variant}")
        obs.label_taxonomy_id = taxonomy[variant].id
    db.flush()

    rows = stats_crud.get_species_distribution(db, project.id)
    by_species = {r.species: r for r in rows}
    assert set(by_species) == {"V. vulpes (adult)", "V. vulpes (juvenile)"}
    assert by_species["V. vulpes (adult)"].common_name == "Red fox adult"


def test_variants_merge_into_the_binomial_at_species_rank(db):
    """At species rank both variants land in one "V. vulpes" bucket.

    The bucket's common name is the binomial too: a variant row has no
    species-level common name, so the bucket falls back to Latin, the
    same way genus and family buckets do."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    taxonomy = _add_variant_taxonomy(db, "ARC-v1")

    ev1 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    ev2 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 3, 9, 0)
    )
    for ev, variant in ((ev1, "adult"), (ev2, "juvenile")):
        obs = _add_observation(db, event_id=ev.id, label=f"red fox {variant}")
        obs.label_taxonomy_id = taxonomy[variant].id
    db.flush()

    rows = stats_crud.get_species_distribution(
        db, project.id, taxonomic_rank="species"
    )
    assert len(rows) == 1
    bucket = rows[0]
    assert bucket.species == "V. vulpes"
    assert bucket.common_name == "V. vulpes"
    assert bucket.count == 2
    assert set(bucket.label_taxonomy_ids) == {
        taxonomy["adult"].id,
        taxonomy["juvenile"].id,
    }
