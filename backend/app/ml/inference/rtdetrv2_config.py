"""Safe config loading for the bundled, pinned RT-DETRv2 implementation.

RT-DETR's upstream YAML loader follows ``__include__`` paths and uses
``yaml.Loader``. User model packs therefore must be validated and flattened
with ``safe_load`` before they reach the upstream ``YAMLConfig`` loader.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml  # type: ignore[import-untyped]

_UPSTREAM_CONFIG_ALIAS = "../third_party/rtdetr/rtdetrv2_pytorch/"


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_include(
    include: Any,
    *,
    current_file: Path,
    model_root: Path,
    source_root: Path,
) -> Path:
    if not isinstance(include, str) or not include.strip():
        raise ValueError("RT-DETRv2 config include entries must be non-empty paths")
    normalized = include.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(include)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or normalized.startswith("~")
    ):
        raise ValueError(f"Absolute RT-DETRv2 config include is not allowed: {include}")

    if normalized.startswith(_UPSTREAM_CONFIG_ALIAS):
        suffix = normalized[len(_UPSTREAM_CONFIG_ALIAS) :]
        candidate = source_root / PurePosixPath(suffix)
    else:
        candidate = current_file.parent / PurePosixPath(normalized)

    resolved = candidate.resolve()
    if not (_within(resolved, model_root) or _within(resolved, source_root)):
        raise ValueError(f"RT-DETRv2 config include escapes allowed roots: {include}")
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"RT-DETRv2 config include must be YAML: {include}")
    if not resolved.is_file():
        raise FileNotFoundError(f"RT-DETRv2 config include not found: {resolved}")
    return resolved


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_file(
    config_path: Path,
    *,
    model_root: Path,
    source_root: Path,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    resolved = config_path.resolve()
    if not (_within(resolved, model_root) or _within(resolved, source_root)):
        raise ValueError(f"RT-DETRv2 config is outside allowed roots: {resolved}")
    if resolved in stack:
        chain = " -> ".join(str(path) for path in (*stack, resolved))
        raise ValueError(f"RT-DETRv2 config include cycle: {chain}")
    if resolved.suffix.lower() not in {".yaml", ".yml"} or not resolved.is_file():
        raise FileNotFoundError(f"RT-DETRv2 YAML config not found: {resolved}")

    with resolved.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"RT-DETRv2 YAML config must contain a mapping: {resolved}")

    includes = document.pop("__include__", [])
    if not isinstance(includes, list):
        raise ValueError(f"RT-DETRv2 __include__ must be a list: {resolved}")

    merged: dict[str, Any] = {}
    next_stack = (*stack, resolved)
    for include in includes:
        include_path = _resolve_include(
            include,
            current_file=resolved,
            model_root=model_root,
            source_root=source_root,
        )
        merged = _merge_dict(
            merged,
            _load_config_file(
                include_path,
                model_root=model_root,
                source_root=source_root,
                stack=next_stack,
            ),
        )
    return _merge_dict(merged, document)


def load_rtdetrv2_config(
    config_path: Path,
    *,
    model_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Load and flatten a model config after constraining every include."""
    resolved_model_root = model_root.resolve()
    resolved_source_root = source_root.resolve()
    if not _within(config_path.resolve(), resolved_model_root):
        raise ValueError("RT-DETRv2 model config must be inside its model directory")
    if not (resolved_source_root / "src" / "core" / "__init__.py").is_file():
        raise FileNotFoundError(f"Pinned RT-DETRv2 source is incomplete: {resolved_source_root}")
    return _load_config_file(
        config_path,
        model_root=resolved_model_root,
        source_root=resolved_source_root,
        stack=(),
    )
