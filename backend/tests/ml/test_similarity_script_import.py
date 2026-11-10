"""The backend imports the labels script in-process for the label tree's
counts (`label_service.count_detections_by_label`). The script's sibling
imports only resolve when it runs by path, so none may sit at module
level: in the packaged app the import failed with ModuleNotFoundError."""

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]


def test_the_labels_script_imports_without_its_folder_on_the_path() -> None:
    # A fresh interpreter, so no earlier test can have put ml/inference on
    # sys.path, which is what hid this in the suite.
    result = subprocess.run(
        [sys.executable, "-c", "import app.ml.inference.similarity_script"],
        cwd=BACKEND,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
