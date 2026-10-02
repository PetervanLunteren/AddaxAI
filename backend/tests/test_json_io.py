"""Tests for `app.utils.json_io.write_json_verified` and the reader side.

The failure these pin is a storage layer that corrupts a file the OS
reported as written (full or failing drive). Fault injection happens by
patching `json.dump` to write broken bytes, which is exactly what a
corrupt landing looks like to the verify step.
"""

import json
from pathlib import Path

import pytest

from app.ml.json_pipeline import load_results_json
from app.utils import json_io
from app.utils.json_io import write_json_verified


def test_writes_readable_json_and_leaves_no_partial(tmp_path: Path) -> None:
    target = tmp_path / "results.json"
    write_json_verified(target, {"images": [1, 2, 3]})
    with open(target) as f:
        assert json.load(f) == {"images": [1, 2, 3]}
    assert not (tmp_path / "results.json.partial").exists()


def test_a_truncated_first_write_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "results.json"
    real_dump = json.dump
    calls = {"n": 0}

    def flaky_dump(obj, f, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            f.write('{"images": [1, 2')
            return
        real_dump(obj, f, **kwargs)

    monkeypatch.setattr(json_io.json, "dump", flaky_dump)
    write_json_verified(target, {"images": [1, 2, 3]})

    assert calls["n"] == 2
    with open(target) as f:
        assert json.load(f) == {"images": [1, 2, 3]}
    assert not (tmp_path / "results.json.partial").exists()


def test_a_persistently_unreadable_write_raises_naming_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "results.json"

    def broken_dump(obj, f, **kwargs):
        f.write("{broken")

    monkeypatch.setattr(json_io.json, "dump", broken_dump)
    with pytest.raises(OSError, match="results.json"):
        write_json_verified(target, {"images": []})

    # The real name was never written, and no half file is left behind.
    assert not target.exists()
    assert not (tmp_path / "results.json.partial").exists()


def test_a_corrupt_results_file_reads_as_an_actionable_error(tmp_path: Path) -> None:
    bad = tmp_path / "detection_image.json"
    bad.write_text('{"images": [1, 2')
    with pytest.raises(RuntimeError, match="full or failing"):
        load_results_json(bad)


def test_a_binary_garbage_results_file_reads_as_the_same_error(tmp_path: Path) -> None:
    """Corruption is not always valid UTF-8; the message must not depend
    on which way the bytes are broken."""
    bad = tmp_path / "detection_image.json"
    bad.write_bytes(bytes(range(256)) * 10)
    with pytest.raises(RuntimeError, match="full or failing"):
        load_results_json(bad)


def test_a_non_serializable_object_leaves_no_partial_behind(tmp_path: Path) -> None:
    """A TypeError is the caller's bug and propagates as itself, but it
    must not leave a junk .partial file on the user's drive."""
    target = tmp_path / "results.json"
    with pytest.raises(TypeError):
        write_json_verified(target, {"x": object()})
    assert not target.exists()
    assert not (tmp_path / "results.json.partial").exists()
