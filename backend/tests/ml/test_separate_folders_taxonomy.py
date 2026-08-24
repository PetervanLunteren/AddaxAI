"""Tests for the taxonomic-tree placement and species exclusion
filter on ``separate_into_folders``.

The base routing tests (animal-no-label fallback, person / vehicle /
blank routing, copy / move modes, collisions) live in
``test_separate_folders.py``. This file pins the behaviour that
depends on the project's LabelTaxonomy chain: full nested paths,
truncation at the deepest known rank, main-species placement, and
how the exclusion filter interacts with both.
"""

from pathlib import Path

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.separate_folders import (
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


def _ctx(output_root: Path) -> OutputContext:
    return OutputContext(output_root=output_root)


def _add_taxonomy(
    db,
    *,
    model_id: str,
    name: str,
    level: str = "species",
    scientific_name: str | None = None,
    taxon_class: str | None = None,
    taxon_order: str | None = None,
    taxon_family: str | None = None,
    taxon_genus: str | None = None,
    taxon_species: str | None = None,
    taxon_variant: str | None = None,
) -> LabelTaxonomy:
    row = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        level=level,
        scientific_name=scientific_name,
        taxon_class=taxon_class,
        taxon_order=taxon_order,
        taxon_family=taxon_family,
        taxon_genus=taxon_genus,
        taxon_species=taxon_species,
        taxon_variant=taxon_variant,
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
        counting_threshold=0.5,
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
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 1
    expected = (
        target
        / "mammalia"
        / "carnivora"
        / "canidae"
        / "canis"
        / "dog"
        / "IMG_001.jpg"
    )
    assert expected.is_file()


def test_scientific_mode_uses_scientific_name_leaf(db, tmp_path):
    """With name_mode="scientific" the leaf folder is the scientific name
    (the abbreviated binomial), while the ancestor ranks stay Latin. The
    common-name leaf is not used."""
    project = make_project(
        db,
        name="sep-tree-sci",
        counting_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="grey wolf",
        level="species",
        scientific_name="C. lupus",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Canidae",
        taxon_genus="Canis",
        taxon_species="Canis lupus",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="grey wolf")

    target = tmp_path / "out"
    separate_into_folders(
        db, project.id, _ctx(target), name_mode="scientific",
        media_threshold=0.5,
    )

    assert (
        target / "mammalia" / "carnivora" / "canidae" / "canis" / "c_lupus"
        / "IMG_001.jpg"
    ).is_file()
    # The common-name leaf must NOT be used in scientific mode.
    assert not (
        target / "mammalia" / "carnivora" / "canidae" / "canis" / "grey_wolf"
    ).exists()


def test_variant_label_gets_a_species_segment(db, tmp_path):
    """A variant label sits one rank below species, so its path gains a
    species segment: Class/Order/Family/Genus/species/leaf."""
    project = make_project(
        db,
        name="sep-tree-variant",
        counting_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="red fox adult",
        level="variant",
        scientific_name="V. vulpes (adult)",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="canidae",
        taxon_genus="vulpes",
        taxon_species="vulpes",
        taxon_variant="adult",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="red fox adult")

    target = tmp_path / "out"
    result = separate_into_folders(
        db, project.id, _ctx(target), media_threshold=0.5
    )

    assert result.copied_count == 1
    expected = (
        target
        / "mammalia"
        / "carnivora"
        / "canidae"
        / "vulpes"
        / "vulpes"
        / "red_fox_adult"
        / "IMG_001.jpg"
    )
    assert expected.is_file()


def test_scientific_mode_falls_back_to_label_without_scientific_name(db, tmp_path):
    """Scientific mode with no scientific_name on the row falls back to
    the common-name label leaf."""
    project = make_project(
        db,
        name="sep-tree-sci-fallback",
        counting_threshold=0.5,
        classification_model_id="test-model",
    )
    _add_taxonomy(
        db,
        model_id="test-model",
        name="dog",
        level="species",
        scientific_name=None,
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
    separate_into_folders(
        db, project.id, _ctx(target), name_mode="scientific",
        media_threshold=0.5,
    )

    assert (
        target / "mammalia" / "carnivora" / "canidae" / "canis" / "dog"
        / "IMG_001.jpg"
    ).is_file()


def test_truncates_at_deepest_known_rank(db, tmp_path):
    """A family-level label (taxon_genus + taxon_species NULL) stops
    the path at the family level — no spurious deeper folders."""
    project = make_project(
        db,
        name="sep-tree-trunc",
        counting_threshold=0.5,
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
    separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert (
        target / "mammalia" / "carnivora" / "canidae" / "IMG_001.jpg"
    ).is_file()
    assert not (target / "mammalia" / "carnivora" / "canidae" / "canis").exists()


def test_multi_species_lands_in_main_species_leaf(db, tmp_path):
    """A dog + lion file lands once, in its main species' (dog) leaf,
    not lion's branch."""
    project = make_project(
        db,
        name="sep-tree-multi",
        counting_threshold=0.5,
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
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 1
    assert (
        target / "mammalia" / "carnivora" / "canidae" / "canis" / "dog" / "IMG_001.jpg"
    ).is_file()
    assert not (target / "mammalia" / "carnivora" / "felidae").exists()


def test_unmapped_label_falls_back_to_other(db, tmp_path):
    """A label with no taxonomy row lands under Other/<label>/."""
    project = make_project(
        db,
        name="sep-tree-unmapped",
        counting_threshold=0.5,
        classification_model_id="test-model",
    )
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="mystery")

    target = tmp_path / "out"
    separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    # _slug lowercases every segment, so the unranked folder is "other".
    # Asserting the lowercase form keeps the test correct on case-sensitive
    # filesystems (Linux CI), where "Other" != "other".
    assert (target / "other" / "mystery" / "IMG_001.jpg").is_file()


# ---------------------------------------------------------------------
# Species exclusion filter
# ---------------------------------------------------------------------


def test_excluded_label_ids_drops_animal_file(db, tmp_path):
    """Animal file whose only label is excluded → skipped_excluded."""
    project = make_project(db, name="sep-excl", counting_threshold=0.5)
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
        _ctx(target),
        excluded_label_ids=frozenset({"dog"}),
        media_threshold=0.5,
    )

    assert result.skipped_excluded == 1
    assert result.copied_count == 0
    assert not (target / "other").exists()


def test_excluded_label_ids_partial_keeps_file_in_remaining_folders(
    db, tmp_path
):
    """File with dog + wolf, exclude wolf → file in Other/dog/ only,
    single placement, not multi."""
    project = make_project(
        db, name="sep-partial-excl", counting_threshold=0.5
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
        _ctx(target),
        excluded_label_ids=frozenset({"wolf"}),
        media_threshold=0.5,
    )

    assert result.copied_count == 1
    assert (target / "other" / "dog" / "IMG_001.jpg").is_file()
    assert not (target / "other" / "wolf").exists()


def test_flat_mode_places_single_segment_per_species(db, tmp_path):
    """``group_by="flat"`` produces one folder per species label at
    the root of separated/, not the nested taxonomic chain."""
    project = make_project(
        db,
        name="sep-flat",
        counting_threshold=0.5,
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
    separate_into_folders(db, project.id, _ctx(target), group_by="flat", media_threshold=0.5)

    assert (target / "dog" / "IMG_001.jpg").is_file()
    assert not (target / "mammalia").exists()


def test_flat_mode_multi_species_main_only(db, tmp_path):
    project = make_project(db, name="sep-flat-multi", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.85, label="wolf")

    target = tmp_path / "out"
    result = separate_into_folders(
        db, project.id, _ctx(target), group_by="flat"
    , media_threshold=0.5)

    assert result.copied_count == 1
    assert (target / "dog" / "IMG_001.jpg").is_file()
    assert not (target / "wolf").exists()


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
        _ctx(target),
        excluded_label_ids=frozenset({"dog", "wolf", "cat"}),
        media_threshold=0.5,
    )

    assert result.copied_count == 3
    assert result.skipped_excluded == 0
