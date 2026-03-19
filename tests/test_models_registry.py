"""Tests for addaxai.models.registry — model discovery and setup utilities."""

import json
import os
import platform
import pytest
from unittest.mock import patch, MagicMock

from addaxai.models.registry import (
    fetch_known_models,
    set_up_unknown_model,
    distribute_individual_model_jsons,
    is_first_startup,
    remove_first_startup_file,
    environment_needs_downloading,
)


# --- fetch_known_models ---

def test_fetch_known_models_lists_subdirs(tmp_path):
    (tmp_path / "modelA").mkdir()
    (tmp_path / "modelB").mkdir()
    (tmp_path / "not_a_dir.txt").write_text("file")

    result = fetch_known_models(str(tmp_path))
    assert result == ["modelA", "modelB"]


def test_fetch_known_models_sorted(tmp_path):
    for name in ["zebra", "alpha", "middle"]:
        (tmp_path / name).mkdir()

    result = fetch_known_models(str(tmp_path))
    assert result == ["alpha", "middle", "zebra"]


def test_fetch_known_models_empty(tmp_path):
    assert fetch_known_models(str(tmp_path)) == []


# --- set_up_unknown_model ---

def test_set_up_unknown_model_creates_dir_and_json(tmp_path):
    model_dict = {"model_fname": "best.pt", "description": "test model"}
    set_up_unknown_model(
        title="my_model",
        model_dict=model_dict,
        model_type="det",
        base_path=str(tmp_path),
    )

    model_dir = tmp_path / "models" / "det" / "my_model"
    assert model_dir.is_dir()

    var_file = model_dir / "variables.json"
    assert var_file.exists()
    data = json.loads(var_file.read_text())
    assert data["model_fname"] == "best.pt"


def test_set_up_unknown_model_downloads_taxonomy_csv(tmp_path):
    model_dict = {
        "model_fname": "best.pt",
        "taxon_mapping_csv": "https://example.com/taxon.csv",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"species,common_name\nPanthera leo,Lion"

    with patch("addaxai.models.registry.requests.get", return_value=mock_response) as mock_get:
        set_up_unknown_model(
            title="taxon_model",
            model_dict=model_dict,
            model_type="cls",
            base_path=str(tmp_path),
        )
        mock_get.assert_called_once_with("https://example.com/taxon.csv", timeout=1)

    csv_path = tmp_path / "models" / "cls" / "taxon_model" / "taxon-mapping.csv"
    assert csv_path.exists()
    assert b"Panthera leo" in csv_path.read_bytes()


def test_set_up_unknown_model_skips_existing_csv(tmp_path):
    model_dir = tmp_path / "models" / "cls" / "existing_model"
    model_dir.mkdir(parents=True)
    csv_path = model_dir / "taxon-mapping.csv"
    csv_path.write_text("already here")

    model_dict = {
        "model_fname": "best.pt",
        "taxon_mapping_csv": "https://example.com/taxon.csv",
    }

    with patch("addaxai.models.registry.requests.get") as mock_get:
        set_up_unknown_model(
            title="existing_model",
            model_dict=model_dict,
            model_type="cls",
            base_path=str(tmp_path),
        )
        mock_get.assert_not_called()


def test_set_up_unknown_model_no_taxonomy(tmp_path):
    model_dict = {"model_fname": "best.pt"}
    set_up_unknown_model(
        title="simple_model",
        model_dict=model_dict,
        model_type="det",
        base_path=str(tmp_path),
    )

    model_dir = tmp_path / "models" / "det" / "simple_model"
    assert not (model_dir / "taxon-mapping.csv").exists()


def test_set_up_unknown_model_handles_download_failure(tmp_path):
    model_dict = {
        "model_fname": "best.pt",
        "taxon_mapping_csv": "https://example.com/taxon.csv",
    }

    with patch("addaxai.models.registry.requests.get", side_effect=Exception("timeout")):
        # Should not raise
        set_up_unknown_model(
            title="fail_model",
            model_dict=model_dict,
            model_type="det",
            base_path=str(tmp_path),
        )

    # variables.json should still be created
    assert (tmp_path / "models" / "det" / "fail_model" / "variables.json").exists()
    # CSV should not exist
    assert not (tmp_path / "models" / "det" / "fail_model" / "taxon-mapping.csv").exists()


# --- distribute_individual_model_jsons ---

def test_distribute_individual_model_jsons(tmp_path):
    model_info = {
        "det": {
            "MegaDetector5": {"model_fname": "md_v5a.pt"},
            "MegaDetector4": {"model_fname": "md_v4.pt"},
        },
        "cls": {
            "SpeciesNet": {"model_fname": "species.pt"},
        },
    }
    info_file = tmp_path / "model_info.json"
    info_file.write_text(json.dumps(model_info))

    distribute_individual_model_jsons(str(info_file), base_path=str(tmp_path))

    # Check det models
    assert (tmp_path / "models" / "det" / "MegaDetector5" / "variables.json").exists()
    assert (tmp_path / "models" / "det" / "MegaDetector4" / "variables.json").exists()
    # Check cls model
    assert (tmp_path / "models" / "cls" / "SpeciesNet" / "variables.json").exists()

    md5_vars = json.loads(
        (tmp_path / "models" / "det" / "MegaDetector5" / "variables.json").read_text()
    )
    assert md5_vars["model_fname"] == "md_v5a.pt"


# --- is_first_startup / remove_first_startup_file ---

def test_is_first_startup_true(tmp_path):
    (tmp_path / "first-startup.txt").write_text("first run")
    assert is_first_startup(base_path=str(tmp_path)) is True


def test_is_first_startup_false(tmp_path):
    assert is_first_startup(base_path=str(tmp_path)) is False


def test_remove_first_startup_file(tmp_path):
    startup_file = tmp_path / "first-startup.txt"
    startup_file.write_text("first run")
    assert startup_file.exists()

    remove_first_startup_file(base_path=str(tmp_path))
    assert not startup_file.exists()


def test_remove_first_startup_file_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        remove_first_startup_file(base_path=str(tmp_path))


# --- environment_needs_downloading ---

def test_environment_needs_downloading_not_present(tmp_path):
    model_vars = {"env": "pytorch"}
    needs_download, env_name = environment_needs_downloading(
        model_vars, base_path=str(tmp_path)
    )
    assert needs_download is True
    assert env_name == "pytorch"


def test_environment_needs_downloading_already_present(tmp_path):
    (tmp_path / "envs" / "env-pytorch").mkdir(parents=True)
    model_vars = {"env": "pytorch"}
    needs_download, env_name = environment_needs_downloading(
        model_vars, base_path=str(tmp_path)
    )
    assert needs_download is False
    assert env_name == "pytorch"


def test_environment_needs_downloading_os_specific_windows(tmp_path):
    model_vars = {"env": "general", "env-windows": "pytorch-win"}
    with patch("addaxai.models.registry.os.name", "nt"):
        with patch("addaxai.models.registry.platform.system", return_value="Windows"):
            needs_download, env_name = environment_needs_downloading(
                model_vars, base_path=str(tmp_path)
            )
    assert env_name == "pytorch-win"


def test_environment_needs_downloading_os_specific_macos(tmp_path):
    model_vars = {"env": "general", "env-macos": "pytorch-mac"}
    with patch("addaxai.models.registry.os.name", "posix"):
        with patch("addaxai.models.registry.platform.system", return_value="Darwin"):
            needs_download, env_name = environment_needs_downloading(
                model_vars, base_path=str(tmp_path)
            )
    assert env_name == "pytorch-mac"


def test_environment_needs_downloading_os_specific_linux(tmp_path):
    model_vars = {"env": "general", "env-linux": "pytorch-linux"}
    with patch("addaxai.models.registry.os.name", "posix"):
        with patch("addaxai.models.registry.platform.system", return_value="Linux"):
            needs_download, env_name = environment_needs_downloading(
                model_vars, base_path=str(tmp_path)
            )
    assert env_name == "pytorch-linux"


def test_environment_needs_downloading_defaults_to_base(tmp_path):
    model_vars = {}  # no env key at all
    needs_download, env_name = environment_needs_downloading(
        model_vars, base_path=str(tmp_path)
    )
    assert env_name == "base"
