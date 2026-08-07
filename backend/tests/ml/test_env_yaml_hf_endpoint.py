"""
Tests for substitute_hf_endpoint: the HF_ENDPOINT rewrite applied to the
writable env YAML copy before micromamba runs.

pip cannot redirect a direct-URL requirement through index configuration,
so on a network where huggingface.co is blocked the hardcoded wheel URL
in the bundled YAMLs fails the env build. The rewrite is the only code
path that fixes that; these tests pin its exact behaviour.
"""

from app.ml.environment_manager import substitute_hf_endpoint

_YAML = (
    "dependencies:\n"
    "  - pip:\n"
    "    - --extra-index-url https://download.pytorch.org/whl/cu128\n"
    "    - torch==2.8.0+cu128\n"
    "    - ultralytics-yolov5 @ https://huggingface.co/Addax-Data-Science/"
    "pip-wheels/resolve/main/ultralytics_yolov5-0.1.1-py3-none-any.whl"
    "#sha256=d532e62d\n"
)


def test_no_endpoint_returns_text_unchanged() -> None:
    assert substitute_hf_endpoint(_YAML, None) == _YAML
    assert substitute_hf_endpoint(_YAML, "") == _YAML


def test_endpoint_rewrites_host_and_keeps_path_and_hash() -> None:
    out = substitute_hf_endpoint(_YAML, "https://hf-mirror.com")
    assert "https://huggingface.co/" not in out
    assert (
        "ultralytics-yolov5 @ https://hf-mirror.com/Addax-Data-Science/"
        "pip-wheels/resolve/main/ultralytics_yolov5-0.1.1-py3-none-any.whl"
        "#sha256=d532e62d" in out
    )


def test_trailing_slash_on_endpoint_does_not_double_slash() -> None:
    out = substitute_hf_endpoint(_YAML, "https://hf-mirror.com/")
    assert "hf-mirror.com//" not in out
    assert "https://hf-mirror.com/Addax-Data-Science/" in out


def test_other_urls_are_untouched() -> None:
    out = substitute_hf_endpoint(_YAML, "https://hf-mirror.com")
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in out
    assert "torch==2.8.0+cu128" in out
