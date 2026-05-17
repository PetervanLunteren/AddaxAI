"""Tests for the taxonomic-tree placement and species exclusion
filter on ``separate_into_folders``.

The base routing tests (animal-no-label fallback, person / vehicle /
blank routing, mode = copy / move / symlink, collisions) live in
``test_separate_folders.py``. This file pins the behaviour that
depends on the project's LabelTaxonomy chain: full nested paths,
truncation at the deepest known rank, multi-species placement, and
how the exclusion filter interacts with both.
"""

from pathlib import Path

from app.ml.postprocessing_outputs.separate_folders import (
    UNRANKED_FOLDER,
    separate_into_folders,
)
from app.models import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _make_source(tmp_path: Path, name: str) -> str:
    src = tmp_path / "source" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"x")
    return str(src)


def _add_taxonomy(
    db,
    *,
    model_id: str,
    name: str,
    level: str = "species",
    taxon_class: str | None = None,
    taxon_order: str | None = None,
    taxon_family: str | None = None,
    taxon_genus: str | None = None,
    taxon_species: str | None = None,
) -> LabelTaxonomy:
    row = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        level=level,
        taxon_class=taxon_class,
        taxon_order=taxon_order,
        taxon_family=taxon_family,
        taxon_genus=taxon_genus,
        taxon_species=taxon_species,
    )
    db.add(row)
    db.flush()
    return row


def test_writes_full_five_level_nested_path(db, tmp_path):
    """A species with the full taxonomy chain produces a 5-level
    nested folder structure: Class/Order/Family/Genus/species."""
    project = make_project(
        db,
        name="sep-tree-full",
        detection_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="dog",
        level="species",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Canidae",
        taxon_genus="Canis",
        taxon_species="Canis lupus familiaris",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target)

    assert result.copied_count == 1
    expected = (
        target
        / "Mammalia"
        / "Carnivora"
        / "Canidae"
        / "Canis"
        / "dog"
        / "IMG_001.jpg"
    )
    assert expected.is_file()


def test_truncates_at_deepest_known_rank(db, tmp_path):
    """A family-level label (taxon_genus + taxon_species NULL) stops
    the path at the family level — no spurious deeper folders."""
    project = make_project(
        db,
        name="sep-tree-trunc",
        detection_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="Canidae",
        level="family",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Canidae",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="Canidae")

    target = tmp_path / "out"
    separate_into_folders(db, project.id, target)

    assert (
        target / "Mammalia" / "Carnivora" / "Canidae" / "IMG_001.jpg"
    ).is_file()
    assert not (target / "Mammalia" / "Carnivora" / "Canidae" / "Canis").exists()


def test_multi_species_two_leaves(db, tmp_path):
    """A file with dog + lion lands in two distinct leaf folders
    under different family branches."""
    project = make_project(
        db,
        name="sep-tree-multi",
        detection_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="dog",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Canidae",
        taxon_genus="Canis",
        taxon_species="Canis lupus familiaris",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="lion",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Felidae",
        taxon_genus="Panthera",
        taxon_species="Panthera leo",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.85, label="lion")

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target)

    assert result.copied_count == 2
    assert result.multi_placement_count == 1
    assert (
        target / "Mammalia" / "Carnivora" / "Canidae" / "Canis" / "dog" / "IMG_001.jpg"
    ).is_file()
    assert (
        target / "Mammalia" / "Carnivora" / "Felidae" / "Panthera" / "lion" / "IMG_001.jpg"
    ).is_file()


def test_unmapped_label_falls_back_to_other(db, tmp_path):
    """A label with no taxonomy row lands under Other/<label>/."""
    project = make_project(
        db,
        name="sep-tree-unmapped",
        detection_threshold=0.5,
        classification_model_id="test-model",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="mystery")

    target = tmp_path / "out"
    separate_into_folders(db, project.id, target)

    assert (target / UNRANKED_FOLDER / "mystery" / "IMG_001.jpg").is_file()


# ---------------------------------------------------------------------
# Species exclusion filter
# ---------------------------------------------------------------------


def test_excluded_label_ids_drops_animal_file(db, tmp_path):
    """Animal file whose only label is excluded → skipped_excluded."""
    project = make_project(db, name="sep-excl", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    target = tmp_path / "out"
    result = separate_into_folders(
        db,
        project.id,
        target,
        excluded_label_ids=frozenset({"dog"}),
    )

    assert result.skipped_excluded == 1
    assert result.copied_count == 0
    assert not (target / "Other").exists()


def test_excluded_label_ids_partial_keeps_file_in_remaining_folders(
    db, tmp_path
):
    """File with dog + wolf, exclude wolf → file in Other/dog/ only,
    single placement, not multi."""
    project = make_project(
        db, name="sep-partial-excl", detection_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.85, label="wolf")

    target = tmp_path / "out"
    result = separate_into_folders(
        db,
        project.id,
        target,
        excluded_label_ids=frozenset({"wolf"}),
    )

    assert result.copied_count == 1
    assert result.multi_placement_count == 0
    assert (target / "Other" / "dog" / "IMG_001.jpg").is_file()
    assert not (target / "Other" / "wolf").exists()


def test_flat_mode_places_single_segment_per_species(db, tmp_path):
    """``group_by="flat"`` produces one folder per species label at
    the root of separated/, not the nested taxonomic chain."""
    project = make_project(
        db,
        name="sep-flat",
        detection_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="dog",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Canidae",
        taxon_genus="Canis",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    target = tmp_path / "out"
    separate_into_folders(db, project.id, target, group_by="flat")

    assert (target / "dog" / "IMG_001.jpg").is_file()
    assert not (target / "Mammalia").exists()


def test_flat_mode_multi_species_two_leaves(db, tmp_path):
    project = make_project(db, name="sep-flat-multi", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.85, label="wolf")

    target = tmp_path / "out"
    result = separate_into_folders(
        db, project.id, target, group_by="flat"
    )

    assert result.copied_count == 2
    assert result.multi_placement_count == 1
    assert (target / "dog" / "IMG_001.jpg").is_file()
    assert (target / "wolf" / "IMG_001.jpg").is_file()


def test_excluded_label_ids_does_not_affect_non_animal_files(db, tmp_path):
    """Person / vehicle / blank files are never dropped by the species
    filter — they have no species labels to match against."""
    project = make_project(db, name="sep-non-animal")
    dep = make_deployment(db, project_id=project.id)
    for name, otype in [
        ("PERSON.jpg", "human"),
        ("CAR.jpg", "vehicle"),
        ("BLANK.jpg", "blank"),
    ]:
        src = _make_source(tmp_path, name)
        make_file(
            db,
            deployment_id=dep.id,
            file_path=src,
            observation_type=otype,
        )

    target = tmp_path / "out"
    result = separate_into_folders(
        db,
        project.id,
        target,
        excluded_label_ids=frozenset({"dog", "wolf", "cat"}),
    )

    assert result.copied_count == 3
    assert result.skipped_excluded == 0
