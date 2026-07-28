"""Manifest loading is resilient: one bad manifest never hides the rest.

Regression for the beta-tester report where every detection manifest was
missing a field the schema then required, so load_manifests raised and the
whole model catalog vanished, surfacing downstream as a misleading
"Detection model 'MD5A-0-0' not found".
"""

import json

from app.ml.manifest_manager import ManifestManager


def _write_manifest(model_dir, data):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "manifest.json").write_text(json.dumps(data))


def _valid_manifest(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "friendly_name": f"Friendly {model_id}",
        "env": "addaxai-base",
        "model_fname": f"{model_id}.pt",
        "description": "A model.",
        "developer": "Someone",
        "info_url": "https://example.com",
        "min_app_version": "0.1.0",
    }


def test_invalid_manifest_is_skipped_not_fatal(tmp_path):
    det = tmp_path / "det"
    _write_manifest(det / "GOOD-1", _valid_manifest("GOOD-1"))
    # Missing required fields (model_id etc.) — must be skipped, not raise.
    _write_manifest(det / "BAD-1", {"friendly_name": "broken"})

    manifests = ManifestManager(models_dir=tmp_path).load_manifests()

    assert "GOOD-1" in manifests
    assert "BAD-1" not in manifests


def test_malformed_json_is_skipped(tmp_path):
    det = tmp_path / "det"
    _write_manifest(det / "GOOD-2", _valid_manifest("GOOD-2"))
    (det / "BROKEN").mkdir(parents=True)
    (det / "BROKEN" / "manifest.json").write_text("{ not valid json")

    manifests = ManifestManager(models_dir=tmp_path).load_manifests()

    assert list(manifests.keys()) == ["GOOD-2"]


def test_all_valid_still_load(tmp_path):
    det = tmp_path / "det"
    _write_manifest(det / "A", _valid_manifest("A"))
    _write_manifest(det / "B", _valid_manifest("B"))

    manifests = ManifestManager(models_dir=tmp_path).load_manifests()

    assert set(manifests.keys()) == {"A", "B"}
    assert manifests["A"].model_category == "detection"
