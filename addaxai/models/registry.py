"""Model registry utilities for AddaxAI.

Model discovery, setup, first-startup checks, and environment
resolution. These functions were extracted from AddaxAI_GUI.py
and parameterized to replace global variable access.
"""

import json
import os
import platform
from pathlib import Path

import requests


def fetch_known_models(root_dir):
    """List known model subdirectories, sorted alphabetically.

    Args:
        root_dir: Directory containing model subdirectories.

    Returns:
        Sorted list of subdirectory names.
    """
    return sorted([
        subdir for subdir in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, subdir))
    ])


def set_up_unknown_model(title, model_dict, model_type, base_path):
    """Create a model directory and write its variables.json.

    Does not download the model weights — just registers the model
    so it appears in the UI dropdown. Optionally downloads a taxonomy
    mapping CSV if specified in model_dict.

    Args:
        title: Model identifier (used as directory name).
        model_dict: Dict of model configuration to write as variables.json.
        model_type: "det" or "cls".
        base_path: Root AddaxAI directory.
    """
    model_dir = os.path.join(base_path, "models", model_type, title)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    var_file = os.path.join(model_dir, "variables.json")
    with open(var_file, "w") as f:
        json.dump(model_dict, f, indent=2)

    # Download taxonomy mapping CSV if specified
    if model_dict.get("taxon_mapping_csv"):
        taxon_mapping_csv_url = model_dict["taxon_mapping_csv"]
        taxon_mapping_csv_path = os.path.join(model_dir, "taxon-mapping.csv")
        if not os.path.exists(taxon_mapping_csv_path):
            try:
                response = requests.get(taxon_mapping_csv_url, timeout=1)
                if response.status_code == 200:
                    with open(taxon_mapping_csv_path, "wb") as f:
                        f.write(response.content)
            except Exception:
                pass


def distribute_individual_model_jsons(model_info_fpath, base_path):
    """Read a master model_info.json and create per-model directories.

    For each model in the "det" and "cls" sections, calls
    set_up_unknown_model to create the directory and variables.json.

    Args:
        model_info_fpath: Path to the master model_info.json file.
        base_path: Root AddaxAI directory.
    """
    with open(model_info_fpath, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    for typ in ["det", "cls"]:
        model_dicts = model_info[typ]
        for model_id in model_dicts:
            set_up_unknown_model(
                title=model_id,
                model_dict=model_dicts[model_id],
                model_type=typ,
                base_path=base_path,
            )


def is_first_startup(base_path):
    """Check whether this is the first startup since install.

    Args:
        base_path: Root AddaxAI directory.

    Returns:
        True if the first-startup.txt sentinel file exists.
    """
    return os.path.exists(os.path.join(base_path, "first-startup.txt"))


def remove_first_startup_file(base_path):
    """Delete the first-startup sentinel file.

    Args:
        base_path: Root AddaxAI directory.

    Raises:
        FileNotFoundError: If the sentinel file does not exist.
    """
    os.remove(os.path.join(base_path, "first-startup.txt"))


def environment_needs_downloading(model_vars, base_path):
    """Check whether a model's conda environment needs downloading.

    Resolves the OS-specific environment name from model_vars, then
    checks whether the environment directory exists.

    Args:
        model_vars: Dict of model variables (may contain env,
            env-windows, env-macos, env-linux keys).
        base_path: Root AddaxAI directory.

    Returns:
        Tuple of (needs_download: bool, env_name: str).
    """
    if os.name == "nt":
        env_name = model_vars.get("env-windows", model_vars.get("env", "base"))
    elif platform.system() == "Darwin":
        env_name = model_vars.get("env-macos", model_vars.get("env", "base"))
    else:
        env_name = model_vars.get("env-linux", model_vars.get("env", "base"))

    env_dir = os.path.join(base_path, "envs", f"env-{env_name}")
    if os.path.isdir(env_dir):
        return (False, env_name)
    else:
        return (True, env_name)
