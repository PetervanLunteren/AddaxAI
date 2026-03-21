"""Tests for addaxai.core.config — settings load/save functions.

These test against temporary JSON files so they don't touch real app config.
"""

import json
import os
import pytest

from addaxai.core.config import load_global_vars, load_model_vars_for, write_global_vars


def test_load_global_vars_reads_json(tmp_path):
    """load_global_vars should read and return the contents of global_vars.json."""
    addaxai_dir = tmp_path / "AddaxAI"
    addaxai_dir.mkdir()
    vars_file = addaxai_dir / "global_vars.json"
    vars_file.write_text(json.dumps({"lang_idx": 0, "var_thresh": 0.3}))

    result = load_global_vars(str(tmp_path))
    assert result == {"lang_idx": 0, "var_thresh": 0.3}


def test_load_global_vars_missing_file(tmp_path):
    """load_global_vars should raise FileNotFoundError if file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_global_vars(str(tmp_path))


def test_write_global_vars_updates_existing_keys(tmp_path):
    """write_global_vars should update existing keys and leave others unchanged."""
    addaxai_dir = tmp_path / "AddaxAI"
    addaxai_dir.mkdir()
    vars_file = addaxai_dir / "global_vars.json"
    vars_file.write_text(json.dumps({"lang_idx": 0, "var_thresh": 0.3}))

    write_global_vars(str(tmp_path), {"var_thresh": 0.5})

    result = load_global_vars(str(tmp_path))
    assert result["var_thresh"] == 0.5
    assert result["lang_idx"] == 0  # unchanged


def test_write_global_vars_ignores_unknown_keys(tmp_path, caplog):
    """write_global_vars should warn and skip keys not in the existing file."""
    import logging
    addaxai_dir = tmp_path / "AddaxAI"
    addaxai_dir.mkdir()
    vars_file = addaxai_dir / "global_vars.json"
    vars_file.write_text(json.dumps({"lang_idx": 0}))

    with caplog.at_level(logging.WARNING, logger="addaxai.core.config"):
        write_global_vars(str(tmp_path), {"nonexistent_key": 42})

    result = load_global_vars(str(tmp_path))
    assert "nonexistent_key" not in result
    assert "nonexistent_key" in caplog.text


def test_write_global_vars_no_changes(tmp_path):
    """write_global_vars with None should rewrite file without changes."""
    addaxai_dir = tmp_path / "AddaxAI"
    addaxai_dir.mkdir()
    vars_file = addaxai_dir / "global_vars.json"
    original = {"lang_idx": 0, "var_thresh": 0.3}
    vars_file.write_text(json.dumps(original))

    write_global_vars(str(tmp_path))

    result = load_global_vars(str(tmp_path))
    assert result == original


def test_load_model_vars_for_reads_json(tmp_path):
    """load_model_vars_for should read a model's variables.json."""
    model_dir = tmp_path / "models" / "cls" / "MyModel"
    model_dir.mkdir(parents=True)
    vars_file = model_dir / "variables.json"
    vars_file.write_text(json.dumps({
        "all_classes": ["deer", "bear"],
        "selected_classes": ["deer"],
    }))

    result = load_model_vars_for(str(tmp_path), "cls", "MyModel")
    assert result["all_classes"] == ["deer", "bear"]
    assert result["selected_classes"] == ["deer"]


def test_load_model_vars_for_missing_returns_empty(tmp_path):
    """load_model_vars_for should return {} if the file doesn't exist."""
    result = load_model_vars_for(str(tmp_path), "cls", "NonExistent")
    assert result == {}
