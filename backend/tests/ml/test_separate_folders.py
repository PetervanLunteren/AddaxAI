"""Tests for the separate_folders postprocess output module.

Covers the taxonomic-tree placement rules (animal labels with /
without taxonomy, animal-no-label fallback, non-animal observation
types), multi-species placement, file-placement modes (copy / move
/ symlink), collision handling, and the missing-source-file
short-circuit.
"""

from pathlib import Path

import pytest

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


def test_separate_routes_animal_to_other_when_no_taxonomy(db, tmp_path):
    """An animal label with no LabelTaxonomy row lands under
    ``Other/<label>/`` — the taxonomic-tree fallback for unmapped
    labels."""
    project = make_project(db, name="sep-animal", detection_threshold=0.5)
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
    result = separate_into_folders(db, project.id, target)

    assert result.copied_count == 1
    assert (target / "Other" / "dog" / "IMG_001.jpg").is_file()


def test_separate_animal_without_label_falls_back_to_animal_folder(db, tmp_path):
    project = make_project(db, name="sep-animal-nolbl", detection_threshold=0.5)
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
    result = separate_into_folders(db, project.id, target)

    assert dict(result.by_label) == {"animal": 1}
    assert (target / "animal" / "IMG_002.jpg").is_file()


def test_separate_threshold_filters_low_confidence_detections(db, tmp_path):
    project = make_project(db, name="sep-thresh", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_003.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    # Below threshold and not verified, must be ignored.
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.2,
        label="cat",
        verified=False,
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target)

    # No detection passes threshold → falls back to animal/ folder.
    assert dict(result.by_label) == {"animal": 1}


def test_separate_routes_human_to_person_folder(db, tmp_path):
    project = make_project(db, name="sep-human")
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_004.jpg")
    make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="human",
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target)

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
    result = separate_into_folders(db, project.id, target)

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
    result = separate_into_folders(db, project.id, target)

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
    result = separate_into_folders(db, project.id, target)

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
    separate_into_folders(db, project.id, target)

    # Source is still where it was; we never move.
    assert Path(src_path).is_file()
    assert Path(src_path).read_bytes() == b"original-bytes"
    # And the destination has the same bytes.
    assert (target / "blank" / "IMG_007.jpg").read_bytes() == b"original-bytes"


def test_separate_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        separate_into_folders(db, "does-not-exist", tmp_path / "out")


def test_move_relocates_file_and_rewrites_db(db, tmp_path):
    """Move mode is the destructive variant: source is gone, the DB's
    File.file_path now points at the destination so the verify UI
    keeps working."""
    project = make_project(db, name="sep-move")
    dep = make_deployment(db, project_id=project.id)
    src_path = _make_source(tmp_path, "IMG_M01.jpg", b"original-bytes")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src_path,
        observation_type="blank",
    )
    file_id = file.id

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target, mode="move")

    # File moved on disk: source gone, destination present.
    assert not Path(src_path).exists()
    moved_to = target / "blank" / "IMG_M01.jpg"
    assert moved_to.is_file()
    assert moved_to.read_bytes() == b"original-bytes"

    # Result counters reflect a move.
    assert result.moved_count == 1
    assert result.copied_count == 0
    assert result.written_count == 1

    # DB has been rewritten so the verify UI keeps working post-move.
    db.expire_all()
    from app.models import File as FileModel

    refreshed = db.get(FileModel, file_id)
    assert refreshed.file_path == str(moved_to)


def test_symlink_creates_link_pointing_at_source(db, tmp_path):
    """Symlink mode leaves the source in place and creates a link
    pointing at it. DB stays unchanged because the original path is
    still authoritative."""
    project = make_project(db, name="sep-link")
    dep = make_deployment(db, project_id=project.id)
    src_path = _make_source(tmp_path, "IMG_L01.jpg", b"link-target")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src_path,
        observation_type="blank",
    )
    file_id = file.id

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target, mode="symlink")

    link_path = target / "blank" / "IMG_L01.jpg"
    assert link_path.is_symlink(), "expected a symlink at the destination"
    # Reading through the link returns the original bytes.
    assert link_path.read_bytes() == b"link-target"

    # Source untouched.
    assert Path(src_path).exists()

    # DB untouched.
    db.expire_all()
    from app.models import File as FileModel

    refreshed = db.get(FileModel, file_id)
    assert refreshed.file_path == src_path

    assert result.linked_count == 1
    assert result.moved_count == 0
    assert result.copied_count == 0


# ---------------------------------------------------------------------
# Multi-species placement
# ---------------------------------------------------------------------


def test_multi_species_lands_in_each_leaf_folder(db, tmp_path):
    """A file with `dog` and `wolf` detections (neither mapped to
    taxonomy) lands in BOTH Other/dog/ and Other/wolf/."""
    project = make_project(db, name="multi-copy", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M01.jpg", b"multi-species")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.95, label="dog"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.80, label="wolf"
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target)

    assert (target / "Other" / "dog" / "IMG_M01.jpg").is_file()
    assert (target / "Other" / "wolf" / "IMG_M01.jpg").is_file()
    assert result.copied_count == 2
    assert result.written_count == 2
    assert result.multi_placement_count == 1
    assert (target / "Other" / "dog" / "IMG_M01.jpg").read_bytes() == b"multi-species"
    assert (target / "Other" / "wolf" / "IMG_M01.jpg").read_bytes() == b"multi-species"


def test_multi_species_repeated_labels_dedupe(db, tmp_path):
    """A file with three `dog` detections places once in Other/dog/,
    not three times."""
    project = make_project(db, name="multi-dedupe", detection_threshold=0.5)
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
    result = separate_into_folders(db, project.id, target)

    assert result.copied_count == 1
    assert result.multi_placement_count == 0
    assert (target / "Other" / "dog" / "IMG_M02.jpg").is_file()


def test_multi_species_threshold_filters(db, tmp_path):
    """A low-confidence wolf detection below threshold does NOT
    add a wolf leaf; the file only lands in Other/dog/."""
    project = make_project(db, name="multi-thresh", detection_threshold=0.5)
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
    result = separate_into_folders(db, project.id, target)

    assert (target / "Other" / "dog" / "IMG_M03.jpg").is_file()
    assert not (target / "Other" / "wolf").exists()
    assert result.multi_placement_count == 0


def test_multi_species_verified_below_threshold_still_placed(db, tmp_path):
    """A verified low-confidence wolf detection still contributes a
    wolf leaf (verified-override rule applies)."""
    project = make_project(db, name="multi-verified", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M04.jpg")
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
        verified=True,
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target)

    assert (target / "Other" / "dog" / "IMG_M04.jpg").is_file()
    assert (target / "Other" / "wolf" / "IMG_M04.jpg").is_file()
    assert result.multi_placement_count == 1


def test_multi_species_move_moves_primary_copies_others(db, tmp_path):
    """Move mode places the source at the primary (top-confidence)
    leaf and copies into the others. DB rewrites to the primary
    path."""
    project = make_project(db, name="multi-move", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M05.jpg", b"mover")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    file_id = file.id
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.95, label="dog"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.80, label="wolf"
    )

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, target, mode="move")

    # Source is gone.
    assert not Path(src).exists()
    # Primary (dog) has the moved file; secondary (wolf) has a copy.
    dog_path = target / "Other" / "dog" / "IMG_M05.jpg"
    wolf_path = target / "Other" / "wolf" / "IMG_M05.jpg"
    assert dog_path.is_file()
    assert wolf_path.is_file()
    assert dog_path.read_bytes() == b"mover"
    assert wolf_path.read_bytes() == b"mover"

    # Counters: one move (primary), one copy (extra).
    assert result.moved_count == 1
    assert result.copied_count == 1
    assert result.written_count == 2
    assert result.multi_placement_count == 1

    # DB rewrites to the primary (moved) destination.
    db.expire_all()
    from app.models import File as FileModel

    refreshed = db.get(FileModel, file_id)
    assert refreshed.file_path == str(dog_path)


def test_multi_species_symlink_links_each_folder(db, tmp_path):
    """Symlink mode produces a symlink in every matching leaf,
    each pointing at the original source."""
    project = make_project(db, name="multi-link", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_M06.jpg", b"link-target")
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
    result = separate_into_folders(db, project.id, target, mode="symlink")

    dog_link = target / "Other" / "dog" / "IMG_M06.jpg"
    wolf_link = target / "Other" / "wolf" / "IMG_M06.jpg"
    assert dog_link.is_symlink()
    assert wolf_link.is_symlink()
    # Both links resolve to the original source path.
    assert dog_link.resolve() == Path(src).resolve()
    assert wolf_link.resolve() == Path(src).resolve()
    # Source is untouched.
    assert Path(src).exists()
    assert Path(src).read_bytes() == b"link-target"

    assert result.linked_count == 2
    assert result.multi_placement_count == 1
