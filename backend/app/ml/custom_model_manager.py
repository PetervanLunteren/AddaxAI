"""Safe, atomic storage for user-managed local detection/classification packs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.manifest_manager import ManifestManager
from app.ml.model_usage import is_model_active, model_usage_guard
from app.ml.schemas.model_manifest import ModelManifest
from app.models.project import Project

_MODEL_TYPE_DIR = {"detection": "det", "classification": "cls"}
_WEIGHT_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".keras", ".onnx",
    ".pb", ".safetensors", ".tflite",
}
_DISPLAY_FIELDS = {
    "friendly_name", "description", "description_short", "developer", "owner",
    "citation", "license", "info_url", "emoji", "region", "example_image_url",
}
_SEMANTIC_FIELDS = {
    "env", "model_fname", "detector_backend", "class_names",
    "detector_model_class", "detector_model_variant", "detector_config_fname",
    "full_image_cls",
}
_write_lock = threading.RLock()


class CustomModelError(ValueError):
    """Expected validation or state error for a custom-model operation."""


def _path_exists(path: Path) -> bool:
    """Return True for ordinary entries and dangling links alike."""
    return os.path.lexists(path)


def _reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CustomModelError(f"Cannot inspect model path: {path}: {exc}") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _reject_reparse_ancestors(path: Path) -> None:
    """Reject symlinks/junctions from the selected path through its root."""
    for candidate in (path, *path.parents):
        if _path_exists(candidate) and _reparse_point(candidate):
            raise CustomModelError(f"Symbolic links and junctions are not allowed: {candidate}")


def _reject_reparse_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            candidate = Path(current) / name
            if _reparse_point(candidate):
                raise CustomModelError(
                    f"Model packs cannot contain symbolic links or junctions: {candidate}"
                )


def _resolve_source(source_path: str, models_dir: Path) -> Path:
    lexical = Path(source_path).expanduser()
    if not lexical.is_absolute():
        raise CustomModelError("source_path must be an absolute local directory path")
    if ".." in PureWindowsPath(source_path).parts or ".." in PurePosixPath(source_path).parts:
        raise CustomModelError("source_path cannot contain parent-directory traversal")
    _reject_reparse_ancestors(lexical)
    try:
        source = lexical.resolve(strict=True)
    except OSError as exc:
        raise CustomModelError(f"Selected model directory is unavailable: {exc}") from exc
    if not source.is_dir():
        raise CustomModelError("Selected model pack must be a directory")
    managed_root = models_dir.resolve()
    if _is_within(source, managed_root) or _is_within(managed_root, source):
        raise CustomModelError("Choose a source pack outside AddaxAI's managed model folders")
    _reject_reparse_tree(source)
    return source


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_relative_file(relative: str, *, field_name: str) -> Path:
    if not relative or "\x00" in relative:
        raise CustomModelError(f"{field_name} must be a non-empty relative path")
    normalized = relative.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(relative)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise CustomModelError(f"{field_name} must remain inside the selected model directory")
    return Path(*normalized.split("/"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_manifest(source: Path) -> dict[str, Any]:
    path = source / "manifest.json"
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CustomModelError(f"Source manifest.json is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise CustomModelError("Source manifest.json must contain an object")
    return document


def _weight_file(source: Path, requested: str | None, source_manifest: dict[str, Any]) -> str:
    candidate = requested or source_manifest.get("model_fname")
    if candidate:
        relative = _safe_relative_file(str(candidate), field_name="model_fname")
        model_file = source / relative
        if (
            not model_file.is_file()
            or _reparse_point(model_file)
            or model_file.suffix.lower() not in _WEIGHT_SUFFIXES
        ):
            raise CustomModelError(f"Model weight file not found: {candidate}")
        return relative.as_posix()

    found = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
    )
    if len(found) != 1:
        raise CustomModelError(
            "Select model_fname explicitly when the pack has zero or multiple weight files"
        )
    return found[0]


def _validate_classifier_inference(source: Path) -> None:
    inference = source / "inference.py"
    if not inference.is_file() or _reparse_point(inference):
        raise CustomModelError("Classification packs must include inference.py")
    try:
        syntax = ast.parse(inference.read_text(encoding="utf-8"), filename=str(inference))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise CustomModelError(f"Classification inference.py is not valid Python: {exc}") from exc
    model_class = next(
        (
            node
            for node in syntax.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelInference"
        ),
        None,
    )
    if model_class is None:
        raise CustomModelError("inference.py must define an AddaxAI ModelInference class")
    methods = {
        node.name
        for node in model_class.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    required = {"check_gpu", "load_model", "get_crop", "get_classification", "get_class_names"}
    missing = sorted(required - methods)
    if missing:
        raise CustomModelError(
            "ModelInference is missing AddaxAI methods: " + ", ".join(missing)
        )


def _available_environments() -> list[str]:
    settings_root = Path(__file__).resolve().parent / "envs"
    platform_dir = {
        "win32": "windows", "darwin": "darwin", "linux": "linux",
    }.get(sys.platform)
    if platform_dir is None:
        return []
    return sorted(
        directory.name
        for directory in settings_root.iterdir()
        if directory.is_dir()
        and (directory / platform_dir / "environment.yml").is_file()
    )


def _source_value(
    payload: Any,
    source_manifest: dict[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    value = getattr(payload, name, None)
    return value if value is not None else source_manifest.get(name, default)


def _manifest_from_pack(payload: Any, source: Path, model_id: str) -> ModelManifest:
    source_manifest = _read_source_manifest(source)
    model_type = payload.type
    env = _source_value(payload, source_manifest, "env")
    if not env or env not in _available_environments():
        raise CustomModelError("env must be one of the environments packaged for this platform")
    model_fname = _weight_file(source, payload.model_fname, source_manifest)

    metadata: dict[str, Any] = {
        "model_id": model_id,
        "friendly_name": payload.friendly_name.strip(),
        "emoji": _source_value(payload, source_manifest, "emoji"),
        "env": env,
        "model_fname": model_fname,
        "description": payload.description or source_manifest.get("description", ""),
        "description_short": payload.description_short or source_manifest.get("description_short"),
        "developer": payload.developer or source_manifest.get("developer", "Local model"),
        "owner": payload.owner or source_manifest.get("owner"),
        "citation": payload.citation or source_manifest.get("citation"),
        "license": payload.license or source_manifest.get("license"),
        "info_url": payload.info_url or source_manifest.get("info_url", ""),
        "min_app_version": source_manifest.get("min_app_version", "0.0.0"),
        "local_only": True,
        "managed": True,
        "managed_created_at": datetime.now(UTC).isoformat(),
        "weights_sha256": _sha256_file(source / model_fname),
    }

    if model_type == "classification":
        _validate_classifier_inference(source)
        metadata.update(
            region=payload.region or source_manifest.get("region"),
            full_image_cls=bool(
                _source_value(payload, source_manifest, "full_image_cls", False)
            ),
            example_image_url=payload.example_image_url or source_manifest.get("example_image_url"),
        )
    else:
        backend = _source_value(payload, source_manifest, "detector_backend", "yolo")
        if backend not in {"yolo", "rfdetr", "rtdetr", "rtdetrv2"}:
            raise CustomModelError(
                "Custom detection backend must be YOLO, RF-DETR, RT-DETR or RT-DETRv2"
            )
        class_names = _source_value(payload, source_manifest, "class_names")
        metadata.update(
            detector_backend=backend,
            class_names=class_names,
            detector_model_class=_source_value(payload, source_manifest, "detector_model_class"),
            detector_model_variant=_source_value(
                payload, source_manifest, "detector_model_variant"
            ),
            detector_config_fname=_source_value(payload, source_manifest, "detector_config_fname"),
        )
        if backend == "rtdetrv2":
            config_name = metadata.get("detector_config_fname")
            if not config_name:
                raise CustomModelError("RT-DETRv2 packs require detector_config_fname")
            config_relative = _safe_relative_file(
                str(config_name), field_name="detector_config_fname"
            )
            config_path = source / config_relative
            if not config_path.is_file() or config_path.suffix.lower() not in {".yml", ".yaml"}:
                raise CustomModelError(
                    "RT-DETRv2 detector_config_fname must point to an existing YAML file"
                )
            metadata["detector_config_fname"] = config_relative.as_posix()

    try:
        return ModelManifest.model_validate(metadata)
    except Exception as exc:
        raise CustomModelError(f"Model pack configuration is invalid: {exc}") from exc


def _manifest_path(manifest: ModelManifest, models_dir: Path) -> Path:
    model_type = {"detection": "det", "classification": "cls"}.get(manifest.model_category or "")
    if model_type is None:
        raise CustomModelError("Only detection and classification models can be managed")
    return models_dir / model_type / manifest.model_id


def _model_in_use(db: Session, model_id: str) -> bool:
    referenced = db.execute(
        select(Project.id).where(
            or_(Project.detection_model_id == model_id, Project.classification_model_id == model_id)
        ).limit(1)
    ).first()
    return bool(referenced) or is_model_active(model_id)


def _model_record(manifest: ModelManifest) -> dict[str, Any]:
    model_type = manifest.model_category
    return {
        "model_id": manifest.model_id,
        "type": model_type,
        "friendly_name": manifest.friendly_name,
        "description": manifest.description,
        "description_short": manifest.description_short,
        "developer": manifest.developer,
        "owner": manifest.owner,
        "citation": manifest.citation,
        "license": manifest.license,
        "info_url": manifest.info_url,
        "emoji": manifest.emoji,
        "env": manifest.env,
        "model_fname": manifest.model_fname,
        "region": manifest.region,
        "full_image_cls": manifest.full_image_cls,
        "example_image_url": manifest.example_image_url,
        "detector_backend": manifest.detector_backend if model_type == "detection" else None,
        "detector_model_class": manifest.detector_model_class,
        "detector_model_variant": manifest.detector_model_variant,
        "detector_config_fname": manifest.detector_config_fname,
        "class_names": manifest.class_names if isinstance(manifest.class_names, dict) else None,
        "local_only": manifest.local_only,
        "managed": manifest.managed,
    }


class CustomModelManager:
    """Own managed model pack import, metadata update, and deletion."""

    def __init__(
        self,
        models_dir: Path | None = None,
        manifest_manager: ManifestManager | None = None,
    ):
        self.models_dir = models_dir or get_settings().models_dir
        self.manifest_manager = manifest_manager or ManifestManager(self.models_dir)

    @staticmethod
    def environments() -> list[str]:
        return _available_environments()

    def list_models(self) -> list[dict[str, Any]]:
        manifests = self.manifest_manager.load_manifests(force_refresh=True)
        return sorted(
            (_model_record(item) for item in manifests.values() if item.managed),
            key=lambda row: (row["type"] or "", row["friendly_name"].casefold()),
        )

    def create(self, payload: Any) -> dict[str, Any]:
        source = _resolve_source(payload.source_path, self.models_dir)
        generated_id = f"custom-{uuid.uuid4().hex}"
        manifest = _manifest_from_pack(payload, source, generated_id)
        category = _MODEL_TYPE_DIR[payload.type]
        destination_root = self.models_dir / category
        _reject_reparse_ancestors(destination_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        _reject_reparse_ancestors(destination_root)
        destination = destination_root / generated_id
        temp_dir = destination_root / f".{generated_id}.tmp-{uuid.uuid4().hex}"

        with _write_lock:
            for model_type_dir in ("det", "cls", "emb"):
                if _path_exists(self.models_dir / model_type_dir / generated_id):
                    raise FileExistsError(f"Model ID collision: {generated_id}")
            if _path_exists(destination) or _reparse_point(destination_root):
                raise FileExistsError(f"Model ID collision: {generated_id}")
            temp_created = False
            try:
                # copytree can create a partial destination before it raises;
                # make that path eligible for cleanup before starting the copy.
                temp_created = True
                shutil.copytree(source, temp_dir, symlinks=True)
                _reject_reparse_tree(temp_dir)
                copied_weights = temp_dir / manifest.model_fname
                if _sha256_file(copied_weights) != manifest.weights_sha256:
                    raise CustomModelError(
                        "Model weights changed while the pack was being copied; retry registration"
                    )
                manifest_target = temp_dir / "manifest.json"
                manifest_tmp = temp_dir / ".manifest.json.tmp"
                manifest_tmp.write_text(
                    json.dumps(
                        manifest.model_dump(exclude_none=True),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                os.replace(manifest_tmp, manifest_target)
                if _path_exists(destination):
                    raise FileExistsError(f"Model ID collision: {generated_id}")
                os.rename(temp_dir, destination)
            except Exception:
                if temp_created and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise

            self.manifest_manager.load_manifests(force_refresh=True)
        return _model_record(self.manifest_manager.get_model(generated_id))

    def update(self, model_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        if not changes:
            raise CustomModelError("At least one field must be updated")
        unknown = set(changes) - _DISPLAY_FIELDS - _SEMANTIC_FIELDS
        if unknown:
            raise CustomModelError("Unsupported custom-model fields: " + ", ".join(sorted(unknown)))
        with _write_lock:
            manifest = self.manifest_manager.get_model(model_id)
            if not manifest.managed or not manifest.local_only:
                raise PermissionError("Standard catalog models cannot be edited")
            semantic_changes = set(changes) & _SEMANTIC_FIELDS
            if semantic_changes:
                raise CustomModelError(
                    "Model weights, inference settings, class definitions, and full-image "
                    "classification behavior are fixed after registration. Register a new "
                    "model to change them."
                )
            model_dir = _manifest_path(manifest, self.models_dir)
            _reject_reparse_ancestors(model_dir)
            if not model_dir.is_dir():
                raise FileNotFoundError("Managed model directory is missing")
            _reject_reparse_tree(model_dir)
            model_data = manifest.model_dump(exclude_none=True)
            model_data.update(changes)
            updated = ModelManifest.model_validate(model_data)
            target_path = model_dir / "manifest.json"
            temp_path = model_dir / ".manifest.json.tmp"
            temp_path.write_text(
                json.dumps(updated.model_dump(exclude_none=True), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temp_path, target_path)
            manifests = self.manifest_manager.load_manifests(force_refresh=True)
            updated.model_category = manifest.model_category
            manifests[model_id] = updated
            self.manifest_manager._cache = manifests
            return _model_record(updated)

    def delete(self, model_id: str, db: Session) -> None:
        with _write_lock, model_usage_guard():
            manifest = self.manifest_manager.get_model(model_id)
            if not manifest.managed or not manifest.local_only:
                raise PermissionError("Standard catalog models cannot be deleted")
            if _model_in_use(db, model_id):
                raise CustomModelError(
                    "Model is referenced by a project or active inference and cannot be deleted"
                )
            model_dir = _manifest_path(manifest, self.models_dir)
            _reject_reparse_ancestors(model_dir)
            if not model_dir.is_dir():
                raise FileNotFoundError("Managed model directory is missing")
            _reject_reparse_tree(model_dir)
            shutil.rmtree(model_dir)
            self.manifest_manager.load_manifests(force_refresh=True)
