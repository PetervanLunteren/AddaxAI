"""Tests for the Save-step output preview computation.

The preview surfaces what the postprocess run will produce as a nested
folder tree (``by_media_tree``: the species / observation folder combined
with the preserved source subfolder in the chosen order). These tests pin
file counts, byte aggregation, single main-species placement, the
combined species + source-subfolder layout under both folder orders, the
in-scope counters under the species exclusion filter, and the non-animal
observation-type fallback.
"""

from app.ml.postprocessing_outputs.output_preview import (
    build_output_preview,
)
from app.models import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _animal_file(db, deployment_id, *, size=None, file_type="image"):
    kw = {"observation_type": "animal", "file_type": file_type}
    if size is not None:
        kw["size_bytes"] = size
    return make_file(db, deployment_id=deployment_id, **kw)


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
) -> LabelTaxonomy:
    row = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        level=level,
        taxon_class=taxon_class,
        taxon_order=taxon_order,
        taxon_family=taxon_family,
        taxon_genus=taxon_genus,
    )
    db.add(row)
    db.flush()
    return row


def test_empty_project_returns_zero_counts(db):
    project = make_project(db, name="prev-empty")

    preview = build_output_preview(db, project.id, media_confidence=0.5)

    assert preview.total_files == 0
    assert preview.image_count == 0
    assert preview.video_count == 0
    assert preview.total_bytes == 0
    assert preview.files_with_known_size == 0
    assert dict(preview.by_media_tree) == {}


def test_image_video_split(db):
    project = make_project(db, name="prev-split")
    dep = make_deployment(db, project_id=project.id)
    _animal_file(db, dep.id, file_type="image")
    _animal_file(db, dep.id, file_type="image")
    _animal_file(db, dep.id, file_type="video")

    preview = build_output_preview(db, project.id, media_confidence=0.5)

    assert preview.total_files == 3
    assert preview.image_count == 2
    assert preview.video_count == 1


def test_size_aggregation_skips_null(db):
    project = make_project(db, name="prev-size")
    dep = make_deployment(db, project_id=project.id)
    _animal_file(db, dep.id, size=1000)
    _animal_file(db, dep.id, size=2000)
    _animal_file(db, dep.id, size=None)

    preview = build_output_preview(db, project.id, media_confidence=0.5)

    assert preview.total_bytes == 3000
    assert preview.files_with_known_size == 2
    assert preview.total_files == 3


def test_unmapped_label_falls_back_to_other(db):
    """No taxonomy row → tree leaf is Other/<label>."""
    project = make_project(
        db,
        name="prev-other",
        counting_threshold=0.5,
        classification_model_id="test-model",
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.9, label="mystery")

    preview = build_output_preview(db, project.id, group_by="taxonomic", media_confidence=0.5)

    assert preview.by_media_tree == {"other/mystery": 1}


def test_full_taxonomy_yields_full_nested_path(db):
    project = make_project(
        db,
        name="prev-full-tree",
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
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    preview = build_output_preview(db, project.id, group_by="taxonomic", media_confidence=0.5)

    assert preview.by_media_tree == {
        "mammalia/carnivora/canidae/canis/dog": 1
    }


def test_multi_species_counts_main_species_only(db):
    """A dog + wolf file is one placement, in its main species' (dog)
    leaf — never both."""
    project = make_project(
        db, name="prev-multi", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.85, label="wolf")

    preview = build_output_preview(db, project.id, group_by="taxonomic", media_confidence=0.5)

    assert preview.by_media_tree == {"other/dog": 1}


def test_low_confidence_detection_is_ignored(db):
    project = make_project(
        db, name="prev-thresh", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.2, label="wolf")

    preview = build_output_preview(db, project.id, group_by="taxonomic", media_confidence=0.5)

    # Wolf is below threshold and unverified — should not place.
    assert preview.by_media_tree == {"other/dog": 1}


def test_verified_below_threshold_still_placed(db):
    project = make_project(
        db, name="prev-verified", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(
        db, file_id=f.id, confidence=0.2, label="dog", verified=True
    )

    preview = build_output_preview(db, project.id, group_by="taxonomic", media_confidence=0.5)

    assert preview.by_media_tree == {"other/dog": 1}


def test_animal_without_passing_label_falls_back(db):
    project = make_project(
        db, name="prev-fallback", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    # A passing animal detection with no species label.
    make_detection(db, file_id=f.id, confidence=0.9, label=None)

    preview = build_output_preview(db, project.id, media_confidence=0.5)

    # Falls back to the single-segment "animal" leaf.
    assert preview.by_media_tree.get("animal") == 1


def test_below_media_confidence_file_reads_as_blank(db):
    """A file whose every detection sits below the media confidence is
    effectively empty for the media outputs, whatever the stored
    observation_type says (that column is derived at the project
    threshold). Lowering the media confidence brings it back."""
    project = make_project(db, name="prev-blank-derived")
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.2, label="cat")

    preview = build_output_preview(db, project.id, media_confidence=0.5)
    assert dict(preview.by_media_tree) == {"blank": 1}

    preview_low = build_output_preview(
        db, project.id, media_confidence=0.1
    )
    assert preview_low.by_media_tree.get("blank") is None


def test_non_animal_observation_types_bucket_to_fixed_folders(db):
    project = make_project(db, name="prev-non-animal")
    dep = make_deployment(db, project_id=project.id)
    f_human = make_file(db, deployment_id=dep.id, observation_type="human")
    make_detection(db, file_id=f_human.id, category="person", confidence=0.9)
    f_vehicle = make_file(
        db, deployment_id=dep.id, observation_type="vehicle"
    )
    make_detection(
        db, file_id=f_vehicle.id, category="vehicle", confidence=0.9
    )
    make_file(db, deployment_id=dep.id, observation_type="blank")
    make_file(db, deployment_id=dep.id, observation_type="blank")

    preview = build_output_preview(db, project.id, media_confidence=0.5)

    assert dict(preview.by_media_tree) == {
        "person": 1,
        "vehicle": 1,
        "blank": 2,
    }


# ---------------------------------------------------------------------
# Combined species + source-subfolder layout (folder order)
# ---------------------------------------------------------------------


def test_source_subfolder_nested_under_species(db, tmp_path):
    """Species-first: the preserved source subfolder sits under the
    species folder, so the preview shows the full combined path."""
    project = make_project(
        db, name="prev-subdir", counting_threshold=0.5
    )
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    src = source / "cam01" / "IMG_1.jpg"
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(src),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    preview = build_output_preview(db, project.id, group_by="flat", media_confidence=0.5)

    assert preview.by_media_tree == {"dog/cam01": 1}
    assert preview.root_files == []


def test_species_last_puts_source_subfolder_on_top(db, tmp_path):
    """Species-last flips the order: source subfolder on top, species
    inside it (the camtrapR station/species layout)."""
    project = make_project(
        db, name="prev-species-last", counting_threshold=0.5
    )
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    src = source / "cam01" / "IMG_1.jpg"
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(src),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    preview = build_output_preview(
        db, project.id, group_by="flat", species_last=True
    , media_confidence=0.5)

    assert preview.by_media_tree == {"cam01/dog": 1}


def test_none_mode_mirrors_source_tree_and_lists_root_files(db, tmp_path):
    """``group_by="none"`` drops the species folder: subfolders feed the
    tree, loose source-root files feed the capped root-file list."""
    project = make_project(
        db, name="prev-none", counting_threshold=0.5
    )
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    nested = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(source / "cam01" / "IMG_1.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=nested.id, confidence=0.9, label="dog")
    loose = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(source / "IMG_2.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=loose.id, confidence=0.9, label="dog")

    preview = build_output_preview(db, project.id, group_by="none", media_confidence=0.5)

    assert preview.by_media_tree == {"cam01": 1}
    assert preview.root_files == ["IMG_2.jpg"]


# ---------------------------------------------------------------------
# Species exclusion filter
# ---------------------------------------------------------------------


def test_excluded_label_ids_drops_file_from_tree(db):
    project = make_project(
        db, name="prev-excl", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")

    preview = build_output_preview(
        db, project.id, excluded_label_ids=frozenset({"dog"})
    , media_confidence=0.5)

    assert preview.dropped_by_filter == 1
    assert preview.in_scope_files == 0
    assert dict(preview.by_media_tree) == {}


def test_excluded_label_ids_partial_inclusion(db):
    """File with dog + wolf, exclude wolf → in scope, single
    placement under the dog leaf only."""
    project = make_project(
        db, name="prev-partial", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(db, file_id=f.id, confidence=0.9, label="dog")
    make_detection(db, file_id=f.id, confidence=0.85, label="wolf")

    preview = build_output_preview(
        db,
        project.id,
        excluded_label_ids=frozenset({"wolf"}),
        group_by="taxonomic",
        media_confidence=0.5,
    )

    assert preview.dropped_by_filter == 0
    assert preview.in_scope_files == 1
    assert preview.by_media_tree == {"other/dog": 1}


def test_excluded_label_ids_does_not_affect_non_animal_files(db):
    project = make_project(db, name="prev-excl-non-animal")
    dep = make_deployment(db, project_id=project.id)
    f_human = make_file(db, deployment_id=dep.id, observation_type="human")
    make_detection(db, file_id=f_human.id, category="person", confidence=0.9)
    make_file(db, deployment_id=dep.id, observation_type="blank")

    preview = build_output_preview(
        db, project.id, excluded_label_ids=frozenset({"dog"})
    , media_confidence=0.5)

    assert preview.dropped_by_filter == 0
    assert preview.in_scope_files == 2
    assert dict(preview.by_media_tree) == {"person": 1, "blank": 1}


def test_excluded_filter_matches_taxonomy_id(db):
    """Exclusion set holding a LabelTaxonomy.id UUID drops the
    matching detection."""
    project = make_project(
        db,
        name="prev-excl-by-id",
        counting_threshold=0.5,
        classification_model_id="test-model",
    )
    taxon = _add_taxonomy(
        db,
        model_id="test-model",
        name="dog",
        taxon_genus="Canis",
        taxon_family="Canidae",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
    )
    dep = make_deployment(db, project_id=project.id)
    f = _animal_file(db, dep.id)
    make_detection(
        db,
        file_id=f.id,
        confidence=0.95,
        label="dog",
        label_taxonomy_id=taxon.id,
    )

    preview = build_output_preview(
        db, project.id, excluded_label_ids=frozenset({taxon.id})
    , media_confidence=0.5)

    assert preview.dropped_by_filter == 1
    assert preview.in_scope_files == 0
    assert dict(preview.by_media_tree) == {}
