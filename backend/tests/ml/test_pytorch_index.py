"""
Tests for pointing the env YAML's PyTorch index at a mirror.

pip has no index priority, so a mirror in pip.ini competes with the
download.pytorch.org entry in the YAML instead of replacing it. Only we
can remove that entry, which is what this substitution is for.
"""

from pathlib import Path

from app.ml.environment_manager import (
    BUNDLED_WHEELS_DIR,
    substitute_bundled_wheels,
    substitute_pytorch_index,
)

ENVS_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "envs"

_ORIGIN = "https://download.pytorch.org/whl/"
_MIRROR = "https://mirror.nju.edu.cn/pytorch/whl"

_YAML = (
    "dependencies:\n"
    "  - pip:\n"
    f"    - --extra-index-url {_ORIGIN}cu128\n"
    "    - torch==2.8.0+cu128\n"
    "    - megadetector==10.0.24\n"
)


def test_no_mirror_leaves_the_text_alone() -> None:
    assert substitute_pytorch_index(_YAML, None) == _YAML
    assert substitute_pytorch_index(_YAML, "") == _YAML


def test_mirror_replaces_the_origin() -> None:
    out = substitute_pytorch_index(_YAML, _MIRROR)

    assert f"--extra-index-url {_MIRROR}/cu128" in out
    assert "download.pytorch.org" not in out


def test_cuda_suffix_is_preserved() -> None:
    """
    One replacement has to cover both indexes we ship, so it swaps the
    prefix and leaves whatever follows it alone.
    """
    text = f"- --extra-index-url {_ORIGIN}cu118\n- --extra-index-url {_ORIGIN}cu128\n"
    out = substitute_pytorch_index(text, _MIRROR)

    assert f"{_MIRROR}/cu118" in out
    assert f"{_MIRROR}/cu128" in out


def test_trailing_slash_does_not_double_up() -> None:
    out = substitute_pytorch_index(_YAML, _MIRROR + "/")

    assert "whl//" not in out
    assert f"{_MIRROR}/cu128" in out


def test_pins_and_other_urls_are_untouched() -> None:
    out = substitute_pytorch_index(_YAML, _MIRROR)

    assert "torch==2.8.0+cu128" in out
    assert "megadetector==10.0.24" in out


def test_every_shipped_pytorch_index_uses_the_prefix_we_replace() -> None:
    """
    The substitution is a plain prefix swap, so a YAML that spells the
    index differently would silently keep pointing at the origin. This
    fails the moment one does.
    """
    seen = 0
    for yaml_path in sorted(ENVS_DIR.glob("*/*/environment.yml")):
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if "download.pytorch.org" not in line:
                continue
            assert _ORIGIN in line, (
                f"{yaml_path} points at download.pytorch.org but not via "
                f"{_ORIGIN!r}, so the mirror substitution would miss it: "
                f"{line.strip()}"
            )
            seen += 1

    assert seen, "no environment.yml references the PyTorch index any more"


def test_real_shipped_yamls_are_fully_rewritten() -> None:
    """The production files, not a sample, end up free of the origin."""
    for yaml_path in sorted(ENVS_DIR.glob("*/*/environment.yml")):
        text = yaml_path.read_text(encoding="utf-8")
        if "download.pytorch.org" not in text:
            continue
        assert "download.pytorch.org" not in substitute_pytorch_index(
            text, _MIRROR
        )


def test_configured_yaml_reaches_no_blocked_host() -> None:
    """
    The whole point, checked against the real files in the order
    `_create_env` applies them: with a mirror configured, nothing a
    Chinese network blocks or throttles is left in the YAML micromamba
    receives. Windows is the platform that carries both.
    """
    for platform_dir in ("windows", "linux"):
        yaml_path = ENVS_DIR / "addaxai-base" / platform_dir / "environment.yml"
        text = substitute_bundled_wheels(
            yaml_path.read_text(encoding="utf-8"), BUNDLED_WHEELS_DIR
        )
        text = substitute_pytorch_index(text, _MIRROR)

        assert "huggingface.co" not in text, platform_dir
        assert "download.pytorch.org" not in text, platform_dir
