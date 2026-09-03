"""Tests for the separate_folders postprocess output module.

Covers the placement rules (animal labels with / without taxonomy,
animal-no-label fallback, non-animal observation types), multi-species
placement, file-placement modes (copy / move), collision handling, the
missing-source-file short-circuit, and the OutputContext recording
contract downstream modules depend on.
"""

from pathlib import Path

import pytest

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.separate_folders import (
    separate_into_folders,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _make_source(tmp_path: Path, name: str, content: bytes = b"x") -> str:
    """Write a small placeholder file and return its absolute path."""
    src = tmp_path / "source" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(content)
    return str(src)


def _ctx(output_root: Path) -> OutputContext:
    return OutputContext(output_root=output_root)


def test_separate_routes_animal_to_other_when_no_taxonomy(db, tmp_path):
    """An animal label with no LabelTaxonomy row lands under
    ``Other/<label>/`` — the taxonomic-tree fallback for unmapped
    labels."""
    project = make_project(db, name="sep-animal", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_001.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label="dog"
    )

    target = tmp_path / "out"
    ctx = _ctx(target)
    result = separate_into_folders(db, project.id, ctx, media_threshold=0.5)

    assert result.copied_count == 1
    assert (target / "other" / "dog" / "IMG_001.jpg").is_file()
    # Context records the placement so downstream modules can find it.
    assert ctx.resolved_for(file.id) == [
        target / "other" / "dog" / "IMG_001.jpg"
    ]


def test_separate_animal_without_label_falls_back_to_animal_folder(db, tmp_path):
    project = make_project(db, name="sep-animal-nolbl", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_002.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label=None
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert dict(result.by_label) == {"animal": 1}
    assert (target / "animal" / "IMG_002.jpg").is_file()


def test_separate_threshold_filters_low_confidence_detections(db, tmp_path):
    """A file whose every detection sits below the media confidence is
    effectively empty for the media outputs: it routes to the blank
    folder regardless of the stored observation_type (derived at the
    project threshold, which the media outputs no longer trust)."""
    project = make_project(db, name="sep-thresh", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_003.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    # Below the media confidence and not verified, must be ignored.
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.2,
        label="cat",
        verified=False,
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert dict(result.by_label) == {"blank": 1}

    # Lowering the media confidence brings the detection into scope and
    # the same file routes to its species folder instead.
    target2 = tmp_path / "out2"
    result2 = separate_into_folders(
        db, project.id, _ctx(target2), media_threshold=0.1
    )
    assert dict(result2.by_label) == {"other/cat": 1}


def test_separate_routes_human_to_person_folder(db, tmp_path):
    project = make_project(db, name="sep-human")
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_004.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="human",
    )
    make_detection(
        db, file_id=file.id, category="person", confidence=0.9
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert dict(result.by_label) == {"person": 1}
    assert (target / "person" / "IMG_004.jpg").is_file()


def test_separate_routes_blank_to_blank_folder(db, tmp_path):
    project = make_project(db, name="sep-blank")
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_005.jpg")
    make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="blank",
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert dict(result.by_label) == {"blank": 1}
    assert (target / "blank" / "IMG_005.jpg").is_file()


def test_separate_renames_on_collision(db, tmp_path):
    project = make_project(db, name="sep-collide")
    dep = make_deployment(db, project_id=project.id)

    # Two source files with the same basename in different sub-folders.
    src1 = _make_source(tmp_path / "a", "IMG_006.jpg", b"first")
    src2 = _make_source(tmp_path / "b", "IMG_006.jpg", b"second")
    make_file(
        db,
        deployment_id=dep.id,
        file_path=src1,
        observation_type="blank",
    )
    make_file(
        db,
        deployment_id=dep.id,
        file_path=src2,
        observation_type="blank",
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 2
    assert result.renamed_count == 1
    blank_dir = target / "blank"
    files = sorted(p.name for p in blank_dir.iterdir())
    assert files == ["IMG_006.jpg", "IMG_006_2.jpg"]


def test_separate_skips_missing_source(db, tmp_path):
    project = make_project(db, name="sep-missing")
    dep = make_deployment(db, project_id=project.id)
    make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "nope" / "IMG_X.jpg"),
        observation_type="blank",
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 0
    assert result.skipped_missing_source == 1
    assert any("nope" in err for err in result.errors)


def test_separate_preserves_original(db, tmp_path):
    project = make_project(db, name="sep-preserve")
    dep = make_deployment(db, project_id=project.id)
    src_path = _make_source(tmp_path, "IMG_007.jpg", b"original-bytes")
    make_file(
        db,
        deployment_id=dep.id,
        file_path=src_path,
        observation_type="blank",
    )

    target = tmp_path / "out"
    separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    # Source is still where it was; we never move.
    assert Path(src_path).is_file()
    assert Path(src_path).read_bytes() == b"original-bytes"
    # And the destination has the same bytes.
    assert (target / "blank" / "IMG_007.jpg").read_bytes() == b"original-bytes"


def test_separate_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        separate_into_folders(
            db, "does-not-exist", _ctx(tmp_path / "out")
        , media_threshold=0.5)


# ---------------------------------------------------------------------
# Single-destination placement
# ---------------------------------------------------------------------


def test_multi_species_lands_in_main_species_folder(db, tmp_path):
    """A dog + wolf file lands once, in its most confident species'
    folder (dog) — never in both."""
    project = make_project(db, name="multi-main", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M01.jpg", b"multi-species")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.95, label="dog"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.80, label="wolf"
    )

    target = tmp_path / "out"
    ctx = _ctx(target)
    result = separate_into_folders(db, project.id, ctx, media_threshold=0.5)

    assert result.copied_count == 1
    assert result.written_count == 1
    assert (target / "other" / "dog" / "IMG_M01.jpg").is_file()
    assert not (target / "other" / "wolf").exists()
    assert ctx.resolved_for(file.id) == [
        target / "other" / "dog" / "IMG_M01.jpg"
    ]


def test_multi_species_repeated_labels_dedupe(db, tmp_path):
    """A file with three `dog` detections places once in Other/dog/."""
    project = make_project(db, name="multi-dedupe", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M02.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    for conf in (0.9, 0.85, 0.7):
        make_detection(
            db,
            file_id=file.id,
            category="animal",
            confidence=conf,
            label="dog",
        )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 1
    assert (target / "other" / "dog" / "IMG_M02.jpg").is_file()


def test_main_species_is_highest_confidence(db, tmp_path):
    """A low-confidence wolf doesn't change the folder; the file lands
    in its main species (dog)."""
    project = make_project(db, name="multi-thresh", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M03.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label="dog"
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.2,
        label="wolf",
        verified=False,
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 1
    assert (target / "other" / "dog" / "IMG_M03.jpg").is_file()
    assert not (target / "other" / "wolf").exists()


# ── The strongest detection names the folder ─────────────────────────


def test_person_subject_beats_a_weaker_labelled_animal(db, tmp_path):
    """The case that started this change. A clip whose strongest box is a
    person files under ``person/`` even though a weaker animal box
    carries a species. Reading the best *label* instead of the best
    *detection* is what filed a person in camouflage under
    ``chimpanzee/``, off one false-positive box the classifier guessed at
    29%, while the still beside it was correctly labelled Person."""
    project = make_project(db, name="sep-mixed", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "MIXED.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="person"
    )
    make_detection(
        db, file_id=file.id, category="person", confidence=0.95
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.68,
        label="chimpanzee",
    )

    target = tmp_path / "out"
    ctx = _ctx(target)
    result = separate_into_folders(db, project.id, ctx, media_threshold=0.2)

    assert dict(result.by_label) == {"person": 1}
    assert (target / "person" / "MIXED.jpg").is_file()
    assert not (target / "other" / "chimpanzee").exists()


def test_unlabelled_novel_category_names_its_own_folder(db, tmp_path):
    """A detector with no classifier behind it still produces useful
    folders, because the category is the folder. Every such file used to
    collapse into one generic ``animal/``."""
    project = make_project(db, name="sep-marine", counting_threshold=0.2)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "REEF.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="shark"
    )
    make_detection(db, file_id=file.id, category="shark", confidence=0.9)

    target = tmp_path / "out"
    ctx = _ctx(target)
    result = separate_into_folders(db, project.id, ctx, media_threshold=0.2)

    assert dict(result.by_label) == {"shark": 1}
    assert (target / "shark" / "REEF.jpg").is_file()
