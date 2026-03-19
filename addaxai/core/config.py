"""Configuration loading and saving for AddaxAI.

Handles global_vars.json (app settings) and per-model variables.json files.
All functions take an explicit base_path parameter instead of relying on globals.
"""

import json
import os


def load_global_vars(base_path):
    """Load global variables from AddaxAI/global_vars.json.

    Args:
        base_path: Root directory containing the AddaxAI/ subfolder.

    Returns:
        Dict of settings loaded from the JSON file.

    Raises:
        FileNotFoundError: If the global_vars.json file doesn't exist.
    """
    var_file = os.path.join(base_path, "AddaxAI", "global_vars.json")
    with open(var_file, 'r') as file:
        return json.load(file)


def write_global_vars(base_path, new_values=None):
    """Update global variables in AddaxAI/global_vars.json.

    Reads the existing file, merges new_values (if provided), and writes back.
    Unknown keys are warned about and skipped.

    Args:
        base_path: Root directory containing the AddaxAI/ subfolder.
        new_values: Optional dict of key-value pairs to update.
    """
    variables = load_global_vars(base_path)
    if new_values is not None:
        for key, value in new_values.items():
            if key in variables:
                variables[key] = value
            else:
                print(f"Warning: Variable {key} not found in the loaded model variables.")

    var_file = os.path.join(base_path, "AddaxAI", "global_vars.json")
    with open(var_file, 'w') as file:
        json.dump(variables, file, indent=4)


def load_model_vars_for(base_path, model_type, model_dir):
    """Load variables.json for a specific model.

    Args:
        base_path: Root directory containing the models/ subfolder.
        model_type: "cls" for classifier or "det" for detector.
        model_dir: Name of the model directory.

    Returns:
        Dict of model variables, or {} if file doesn't exist.
    """
    var_file = os.path.join(base_path, "models", model_type, model_dir, "variables.json")
    try:
        with open(var_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception:
        return {}
