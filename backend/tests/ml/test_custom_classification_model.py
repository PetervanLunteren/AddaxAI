"""Tests for the classification worker's env building."""

from pathlib import Path

from app.ml.inference.custom_classification_model import _worker_path_prefix


def test_prefix_added_for_windows_tensorflow_v1():
    """Windows tensorflow-v1 gets its Library\\bin so TF 2.10 finds CUDA."""
    env_dir = Path("C:/Users/x/AddaxAI/envs/env-tensorflow-v1")
    prefix = _worker_path_prefix("tensorflow-v1", env_dir, "Windows")
    assert prefix == env_dir / "Library" / "bin"


def test_no_prefix_for_other_envs_on_windows():
    """pytorch/pywildlife self-load CUDA; tensorflow-v2 has no Windows GPU."""
    env_dir = Path("C:/Users/x/AddaxAI/envs/env-pytorch")
    assert _worker_path_prefix("pytorch", env_dir, "Windows") is None
    assert _worker_path_prefix("tensorflow-v2", env_dir, "Windows") is None
    assert _worker_path_prefix("pywildlife", env_dir, "Windows") is None


def test_no_prefix_off_windows():
    """macOS/Linux resolve conda libs differently and already work."""
    env_dir = Path("/home/x/AddaxAI/envs/env-tensorflow-v1")
    assert _worker_path_prefix("tensorflow-v1", env_dir, "Linux") is None
    assert _worker_path_prefix("tensorflow-v1", env_dir, "Darwin") is None
