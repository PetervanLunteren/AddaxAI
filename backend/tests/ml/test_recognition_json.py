"""Tests for the recognition_json postprocess output module.

Pins the canonical AddaxAI / Timelapse JSON shape so a folder run's
output is interchangeable with what the Timelapse runner writes and
what existing downstream tooling expects.
"""

import json
import uuid
from pathlib import Path

import pytest

from app.ml.postprocessing_outputs.recognition_json import (
    RECOGNITION_JSON_FILENAME,
    write_recognition_json,
)
from app.models import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _make_taxonomy(db, **kw) -> LabelTaxonomy:
    defaults = dict(
        id=str(uuid.uuid4()),
        classification_model_id="TEST-MODEL",
        name="fox",
        display_name="Vulpes vulpes",
        level="species",
    )
    defaults.update(kw)
    tax = LabelTaxonomy(**defaults)
    db.add(tax)
    db.flush()
    return tax


def _load_json(target_dir: Path) -> dict:
    """Read the recognition file from a finished run."""
    path = target_dir / RECOGNITION_JSON_FILENAME
    assert path.is_file(), f"recognition file missing at {path}"
    with open(path) as f:
        return json.load(f)


def test_output_has_canonical_top_level_keys(db, tmp_path):
    project = make_project(db, name="rj-keys")
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG_001.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)

    payload = _load_json(target)
    assert set(payload.keys()) >= {
        "images",
        "detection_categories",
        "classification_categories",
        "info",
    }
    # MD category mapping matches the rest of the codebase.
    assert payload["detection_categories"] == {
        "1": "animal",
        "2": "person",
        "3": "vehicle",
    }
    # info.addaxai is the canonical metadata block.
    assert "addaxai" in payload["info"]
    info = payload["info"]["addaxai"]
    assert info["detection_model"] == project.detection_model_id
    assert info["deployment_id"] == dep.id
    assert "classification_completion_time" in info


def test_detection_serialisation_matches_canonical_shape(db, tmp_path):
    project = make_project(db, name="rj-shape")
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG_001.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.873,
        label="dog",
        label_confidence=0.91,
        bbox_x=0.10,
        bbox_y=0.20,
        bbox_width=0.30,
        bbox_height=0.40,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    assert len(payload["images"]) == 1
    img = payload["images"][0]
    assert img["file"] == "IMG_001.jpg"  # path relative to deployment folder
    assert len(img["detections"]) == 1
    det = img["detections"][0]

    # Category serialised as the MD numeric id string.
    assert det["category"] == "1"
    assert det["conf"] == pytest.approx(0.873)
    assert det["bbox"] == [
        pytest.approx(0.10),
        pytest.approx(0.20),
        pytest.approx(0.30),
        pytest.approx(0.40),
    ]
    # Classifications are a list of [id_str, confidence] pairs.
    assert det["classifications"] == [["1", pytest.approx(0.91)]]


def test_classification_categories_map_built_from_labels(db, tmp_path):
    project = make_project(db, name="rj-classes")
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src"),
    )
    file_a = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "a.jpg"),
        observation_type="animal",
    )
    file_b = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "b.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file_a.id,
        category="animal",
        confidence=0.9,
        label="dog",
        label_confidence=0.8,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
    )
    make_detection(
        db,
        file_id=file_b.id,
        category="animal",
        confidence=0.9,
        label="cat",
        label_confidence=0.7,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    # Both labels show up with monotonic string ids.
    classes = payload["classification_categories"]
    assert set(classes.values()) == {"dog", "cat"}
    assert set(classes.keys()) == {"1", "2"}


def test_classification_category_descriptions_carry_taxonomy(db, tmp_path):
    """The 7-token taxonomy strings are rebuilt from label_taxonomy, keyed
    by the same classification category id, matching results mode."""
    project = make_project(db, name="rj-tax")
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src"),
    )
    tax = _make_taxonomy(
        db,
        name="fox",
        taxon_class="Mammalia",
        taxon_order="Carnivora",
        taxon_family="Canidae",
        taxon_genus="Vulpes",
        taxon_species="vulpes",
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="fox",
        label_confidence=0.8,
        label_taxonomy_id=tax.id,
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.3, bbox_height=0.3,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    classes = payload["classification_categories"]
    fox_id = next(cid for cid, name in classes.items() if name == "fox")
    descriptions = payload["classification_category_descriptions"]
    # 7-token, all lowercase: name;class;order;family;genus;species;name
    assert descriptions[fox_id] == (
        "fox;mammalia;carnivora;canidae;vulpes;vulpes;fox"
    )


def test_custom_label_without_ranks_has_no_description(db, tmp_path):
    """A label whose taxonomy carries no ranks (a user-invented label) gets
    no description entry, and the key is omitted when nothing has taxonomy."""
    project = make_project(db, name="rj-custom")
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src"),
    )
    tax = _make_taxonomy(db, name="fake-bird", level="unknown", is_custom=True)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="fake-bird",
        label_confidence=1.0,
        label_taxonomy_id=tax.id,
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.3, bbox_height=0.3,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    assert "fake-bird" in payload["classification_categories"].values()
    assert "classification_category_descriptions" not in payload


def test_detection_without_label_omits_classifications(db, tmp_path):
    project = make_project(db, name="rj-no-cls")
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label=None,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    det = payload["images"][0]["detections"][0]
    assert "classifications" not in det


def test_verified_flag_per_detection(db, tmp_path):
    """Each detection carries its human-verified state, so the folder JSON
    captures review status from the DB (not present in raw results-mode
    output, but a deliberate addaxai extension)."""
    project = make_project(db, name="rj-verified")
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.95,
        verified=True, bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.90,
        verified=False, bbox_x=0.5, bbox_y=0.5, bbox_width=0.2, bbox_height=0.2,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    dets = payload["images"][0]["detections"]
    # Ordered by confidence desc: the 0.95 (verified) detection comes first.
    assert [d["verified"] for d in dets] == [True, False]


def test_video_frame_number_preserved(db, tmp_path):
    project = make_project(db, name="rj-video")
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "VID.mp4"),
        file_type="video",
        file_format="mp4",
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
        frame_number=42,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    det = payload["images"][0]["detections"][0]
    assert det["frame_number"] == 42


def test_event_level_observation_dropped(db, tmp_path):
    """A detection with no bbox cannot be serialised in the canonical
    shape (there's no place for it in the schema), so it is dropped."""
    project = make_project(db, name="rj-event")
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        bbox_x=None,
        bbox_y=None,
        bbox_width=None,
        bbox_height=None,
    )

    target = tmp_path / "out"
    write_recognition_json(db, project.id, target)
    payload = _load_json(target)

    assert len(payload["images"]) == 1
    assert payload["images"][0]["detections"] == []


def test_filename_is_canonical(db, tmp_path):
    project = make_project(db, name="rj-filename")
    make_deployment(db, project_id=project.id, folder_path=str(tmp_path / "src"))

    target = tmp_path / "out"
    result = write_recognition_json(db, project.id, target)

    # Same filename as the Timelapse runner uses, so existing
    # downstream scripts find it.
    assert result.output_path.endswith("recognition.json")
    assert (target / "recognition.json").is_file()


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_recognition_json(db, "no-such-id", tmp_path / "out")
