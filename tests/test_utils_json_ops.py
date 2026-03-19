"""Tests for addaxai.utils.json_ops — JSON recognition file manipulation."""

import json
import os
import pytest

from addaxai.utils.json_ops import (
    append_to_json,
    change_hitl_var_in_json,
    check_json_paths,
    fetch_label_map_from_json,
    get_hitl_var_in_json,
    make_json_absolute,
    make_json_relative,
    merge_jsons,
)


def _write_json(path, data):
    """Helper: write a dict as JSON to a file."""
    with open(path, "w") as f:
        json.dump(data, f)


def _read_json(path):
    """Helper: read and return JSON from a file."""
    with open(path) as f:
        return json.load(f)


# --- fetch_label_map_from_json ---

def test_fetch_label_map_from_json(tmp_path):
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "detection_categories": {"1": "animal", "2": "person"},
        "images": [],
    })
    result = fetch_label_map_from_json(str(f))
    assert result == {"1": "animal", "2": "person"}


# --- check_json_paths ---

def test_check_json_paths_absolute(tmp_path):
    base = str(tmp_path)
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "images": [{"file": os.path.join(base, "photo.jpg")}],
    })
    assert check_json_paths(str(f), base) == "absolute"


def test_check_json_paths_relative(tmp_path):
    base = str(tmp_path)
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "images": [{"file": "subfolder/photo.jpg"}],
    })
    assert check_json_paths(str(f), base) == "relative"


# --- make_json_relative ---

def test_make_json_relative_converts_absolute(tmp_path):
    base = str(tmp_path)
    abs_path = os.path.join(base, "sub", "photo.jpg")
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "images": [{"file": abs_path}],
    })
    make_json_relative(str(f), base)
    data = _read_json(str(f))
    # Should now be relative (no leading base path)
    assert not os.path.isabs(data["images"][0]["file"])
    assert "photo.jpg" in data["images"][0]["file"]


def test_make_json_relative_noop_if_already_relative(tmp_path):
    base = str(tmp_path)
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "images": [{"file": "sub/photo.jpg"}],
    })
    make_json_relative(str(f), base)
    data = _read_json(str(f))
    assert data["images"][0]["file"] == "sub/photo.jpg"


# --- make_json_absolute ---

def test_make_json_absolute_converts_relative(tmp_path):
    base = str(tmp_path)
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "images": [{"file": "sub/photo.jpg"}],
    })
    make_json_absolute(str(f), base)
    data = _read_json(str(f))
    expected = os.path.normpath(os.path.join(base, "sub", "photo.jpg"))
    assert data["images"][0]["file"] == expected


def test_make_json_absolute_noop_if_already_absolute(tmp_path):
    base = str(tmp_path)
    abs_path = os.path.join(base, "sub", "photo.jpg")
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "images": [{"file": abs_path}],
    })
    make_json_absolute(str(f), base)
    data = _read_json(str(f))
    assert os.path.normpath(data["images"][0]["file"]) == os.path.normpath(abs_path)


# --- append_to_json ---

def test_append_to_json_adds_info(tmp_path):
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "info": {"detector": "megadetector"},
        "images": [],
    })
    append_to_json(str(f), {"classifier": "mewc"})
    data = _read_json(str(f))
    assert data["info"]["detector"] == "megadetector"
    assert data["info"]["classifier"] == "mewc"


# --- change_hitl_var_in_json / get_hitl_var_in_json ---

def test_change_hitl_var_in_json(tmp_path):
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "info": {"addaxai_metadata": {"hitl_status": "never-started"}},
        "images": [],
    })
    change_hitl_var_in_json(str(f), "in-progress")
    data = _read_json(str(f))
    assert data["info"]["addaxai_metadata"]["hitl_status"] == "in-progress"


def test_get_hitl_var_in_json_returns_status(tmp_path):
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "info": {"addaxai_metadata": {"hitl_status": "completed"}},
        "images": [],
    })
    assert get_hitl_var_in_json(str(f)) == "completed"


def test_get_hitl_var_in_json_default_never_started(tmp_path):
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "info": {"addaxai_metadata": {}},
        "images": [],
    })
    assert get_hitl_var_in_json(str(f)) == "never-started"


def test_get_hitl_var_in_json_backward_compat_ecoassist(tmp_path):
    """Old files use 'ecoassist_metadata' instead of 'addaxai_metadata'."""
    f = tmp_path / "rec.json"
    _write_json(str(f), {
        "info": {"ecoassist_metadata": {"hitl_status": "done"}},
        "images": [],
    })
    assert get_hitl_var_in_json(str(f)) == "done"


# --- merge_jsons ---

def test_merge_jsons_both_files(tmp_path):
    img = tmp_path / "img.json"
    vid = tmp_path / "vid.json"
    out = tmp_path / "merged.json"

    _write_json(str(img), {
        "images": [{"file": "a.jpg"}],
        "detection_categories": {"1": "animal"},
        "info": {"detector": "md"},
        "classification_categories": {"0": "deer"},
    })
    _write_json(str(vid), {
        "images": [{"file": "b.mp4"}],
        "detection_categories": {"1": "animal"},
        "info": {"detector": "md"},
    })

    merge_jsons(str(img), str(vid), str(out))
    data = _read_json(str(out))
    assert len(data["images"]) == 2
    assert data["detection_categories"] == {"1": "animal"}
    assert data["classification_categories"] == {"0": "deer"}


def test_merge_jsons_image_only(tmp_path):
    img = tmp_path / "img.json"
    out = tmp_path / "merged.json"

    _write_json(str(img), {
        "images": [{"file": "a.jpg"}],
        "detection_categories": {"1": "animal"},
        "info": {"detector": "md"},
    })

    merge_jsons(str(img), None, str(out))
    data = _read_json(str(out))
    assert len(data["images"]) == 1


def test_merge_jsons_video_only(tmp_path):
    vid = tmp_path / "vid.json"
    out = tmp_path / "merged.json"

    _write_json(str(vid), {
        "images": [{"file": "b.mp4"}],
        "detection_categories": {"1": "animal"},
        "info": {"detector": "md"},
    })

    merge_jsons(None, str(vid), str(out))
    data = _read_json(str(out))
    assert len(data["images"]) == 1
