"""Tests for app.ml.postprocessing utilities."""

import json
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.ml.postprocessing import (
    build_smoother_input,
    compute_postprocessing_settings_hash,
    run_postprocessing_for_deployment,
)
from tests.conftest import make_deployment, make_file, make_project, make_site


def _make_project_mock(**overrides):
    defaults = dict(
        event_smoothing=True,
        smoothing_strength="normal",
        taxonomic_rollup=True,
        independence_interval=1800,
        excluded_classes=[],
        country_code=None,
        state_code=None,
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def test_hash_deterministic():
    p = _make_project_mock()
    h1 = compute_postprocessing_settings_hash(p)
    h2 = compute_postprocessing_settings_hash(p)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_hash_changes_on_smoothing_toggle():
    p1 = _make_project_mock(event_smoothing=True)
    p2 = _make_project_mock(event_smoothing=False)
    assert compute_postprocessing_settings_hash(p1) != compute_postprocessing_settings_hash(p2)


def test_hash_changes_on_interval_change():
    p1 = _make_project_mock(independence_interval=1800)
    p2 = _make_project_mock(independence_interval=3600)
    assert compute_postprocessing_settings_hash(p1) != compute_postprocessing_settings_hash(p2)


def test_hash_excluded_classes_order_independent():
    p1 = _make_project_mock(excluded_classes=["a", "b"])
    p2 = _make_project_mock(excluded_classes=["b", "a"])
    assert compute_postprocessing_settings_hash(p1) == compute_postprocessing_settings_hash(p2)


def test_hash_changes_on_country_code():
    p1 = _make_project_mock(country_code="KEN")
    p2 = _make_project_mock(country_code="NLD")
    assert compute_postprocessing_settings_hash(p1) != compute_postprocessing_settings_hash(p2)


def test_smoother_input_empty_deployment(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    result = build_smoother_input(d.id, 1800, db)
    assert result == []


def test_smoother_input_single_file(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 1, 1, 12, 0))
    result = build_smoother_input(d.id, 1800, db)
    assert len(result) == 1
    # seq_id is MegaDetector's CCT contract — kept in the emitted rows.
    assert "seq_id" in result[0]
    assert "file_name" in result[0]


def test_smoother_input_groups_within_interval(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    f1 = make_file(
        db, deployment_id=d.id,
        file_path="/fake/folder/img_001.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 0, 0),
    )
    f2 = make_file(
        db, deployment_id=d.id,
        file_path="/fake/folder/img_002.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 10, 0),
    )
    _ = (f1, f2)
    result = build_smoother_input(d.id, 1800, db)
    assert len(result) == 2
    # Both share seq_id (10 min gap < 30 min interval, same folder).
    assert result[0]["seq_id"] == result[1]["seq_id"]


def test_smoother_input_splits_on_gap(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(
        db, deployment_id=d.id,
        file_path="/fake/folder/img_001.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 0, 0),
    )
    make_file(
        db, deployment_id=d.id,
        file_path="/fake/folder/img_002.jpg",
        captured_at_local=datetime(2024, 1, 1, 13, 0, 0),
    )
    result = build_smoother_input(d.id, 1800, db)
    assert len(result) == 2
    # 60 min gap > 30 min interval → different seq_id.
    assert result[0]["seq_id"] != result[1]["seq_id"]


def test_smoother_input_splits_on_folder_boundary(db):
    """A backlog deployment with files in two folders must give two
    distinct seq_ids, even when timestamps fall inside the interval."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/data/backlog")
    make_file(
        db, deployment_id=d.id,
        file_path="/data/backlog/card_a/img_001.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 0, 0),
    )
    make_file(
        db, deployment_id=d.id,
        file_path="/data/backlog/card_b/img_001.jpg",
        captured_at_local=datetime(2024, 1, 1, 12, 5, 0),
    )
    result = build_smoother_input(d.id, 1800, db)
    assert len(result) == 2
    # Different folder → different seq_id, matching Event behaviour.
    assert result[0]["seq_id"] != result[1]["seq_id"]


# ---------------------------------------------------------------------------
# Smoothing strength tests
# ---------------------------------------------------------------------------


def test_hash_changes_on_smoothing_strength():
    """Each smoothing strength should produce a different settings hash."""
    hashes = set()
    for strength in ("mild", "normal", "aggressive"):
        p = _make_project_mock(smoothing_strength=strength)
        hashes.add(compute_postprocessing_settings_hash(p))
    assert len(hashes) == 3


def test_smoothing_strength_passed_to_subprocess(db):
    """run_postprocessing_for_deployment passes smoothing_strength to the subprocess options."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 1, 1, 12, 0, 0))

    # Write a minimal results JSON
    results = {
        "images": [],
        "classification_categories": {},
        "classification_category_descriptions": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        json_path = f.name

    project_mock = _make_project_mock(
        smoothing_strength="aggressive",
        classification_model_id=None,
        excluded_classes=[],
        taxonomic_rollup=False,
        detection_threshold=0.5,
    )

    captured_opts = {}

    def fake_popen_group(args, **kwargs):
        # Read the options JSON that was passed to the subprocess.
        # args layout: [python, script, input, opts, output]
        opts_path = args[3]
        with open(opts_path) as f:
            captured_opts.update(json.load(f))
        # Write empty output so the caller doesn't crash on json.load.
        with open(args[4], "w") as f:
            json.dump(results, f)
        # Mock a Popen-like process: the production code calls
        # process.communicate() and reads .returncode.
        process = MagicMock()
        process.communicate.return_value = ("", "")
        process.returncode = 0
        return process

    with (
        patch("app.ml.postprocessing.popen_group", side_effect=fake_popen_group),
        patch("app.ml.postprocessing._get_ml_python_path", return_value="/fake/python"),
    ):
        run_postprocessing_for_deployment(
            d.id, json_path, "/fake/folder", project_mock, db
        )

    assert captured_opts["smoothing_strength"] == "aggressive"


def test_smoothing_presets_aggressiveness_ordering():
    """Verify that preset parameters get monotonically more aggressive.

    More aggressive means:
    - lower confidence threshold (more classifications visible)
    - fewer detections needed to overwrite (easier to overwrite)
    - more non-dominant classes tolerated
    """
    # Import presets from the script — this dict is pure data, but the module
    # imports megadetector at the top level so we parse it directly.
    import ast
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent.parent.parent
    script_path = script_dir / "app" / "ml" / "smoothing_script.py"
    tree = ast.parse(script_path.read_text())

    # Find the SMOOTHING_PRESETS assignment
    presets = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SMOOTHING_PRESETS":
                    presets = ast.literal_eval(node.value)
    assert presets is not None, "Could not find SMOOTHING_PRESETS in smoothing_script.py"

    mild = presets["mild"]
    normal = presets["normal"]
    aggressive = presets["aggressive"]

    # Confidence threshold: mild > normal > aggressive
    k = "classification_confidence_threshold"
    assert mild[k] > normal[k]
    assert normal[k] > aggressive[k]

    # min_detections_to_overwrite_other: mild > normal > aggressive
    k = "min_detections_to_overwrite_other"
    assert mild[k] > normal[k]
    assert normal[k] > aggressive[k]

    # min_detections_to_overwrite_secondary: mild > normal > aggressive
    k = "min_detections_to_overwrite_secondary"
    assert mild[k] > normal[k]
    assert normal[k] > aggressive[k]

    # max_detections_nondominant_class: mild <= normal <= aggressive
    k = "max_detections_nondominant_class"
    assert mild[k] <= normal[k]
    assert normal[k] <= aggressive[k]


def test_smoothing_presets_cover_all_strengths():
    """All three preset names must be present."""
    import ast
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent.parent.parent
    script_path = script_dir / "app" / "ml" / "smoothing_script.py"
    tree = ast.parse(script_path.read_text())

    presets = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SMOOTHING_PRESETS":
                    presets = ast.literal_eval(node.value)

    assert presets is not None
    assert set(presets.keys()) == {"mild", "normal", "aggressive"}

    # Each preset should have the same parameter keys
    keys = set(presets["normal"].keys())
    assert set(presets["mild"].keys()) == keys
    assert set(presets["aggressive"].keys()) == keys


def test_smoothing_failure_returns_rollup_results(db):
    """When smoothing subprocess crashes, rollup-only results are returned."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/fake/folder")
    make_file(db, deployment_id=d.id, captured_at_local=datetime(2024, 1, 1, 12, 0, 0))

    results = {
        "images": [
            {
                "file": "img_001.jpg",
                "detections": [
                    {
                        "category": "1",
                        "conf": 0.9,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                        "classifications": [["1", 0.85]],
                    }
                ],
            }
        ],
        "classification_categories": {"1": "lion"},
        "classification_category_descriptions": {
            "1": "lion;mammalia;carnivora;felidae;panthera;leo;lion"
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        json_path = f.name

    project_mock = _make_project_mock(
        classification_model_id=None,
        excluded_classes=[],
        taxonomic_rollup=False,
        detection_threshold=0.5,
    )

    def crashing_subprocess(args, **kwargs):
        return MagicMock(
            returncode=1,
            stderr="KeyError: '999'",
            stdout="",
        )

    with (
        patch("app.ml.postprocessing.subprocess.run", side_effect=crashing_subprocess),
        patch("app.ml.postprocessing._get_ml_python_path", return_value="/fake/python"),
    ):
        result = run_postprocessing_for_deployment(
            d.id, json_path, "/fake/folder", project_mock, db
        )

    # Should return the pre-smoothing results, not crash
    assert "images" in result
    assert len(result["images"]) == 1
    det = result["images"][0]["detections"][0]
    assert det["classifications"][0][0] == "1"
