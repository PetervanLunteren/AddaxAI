"""Safe, atomic storage for user-managed local detection/classification packs."""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml  # type: ignore[import-untyped]
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.manifest_manager import ManifestManager
from app.ml.model_usage import is_model_active, model_usage_guard
from app.ml.schemas.model_manifest import (
    ModelManifest,
    uses_detection_classes_for_classification,
)
from app.models.project import Project

_MODEL_TYPE_DIR = {"detection": "det", "classification": "cls"}
_DETECTOR_ENVIRONMENTS = {"yolo": "pytorch", "rtdetrv2": "rtdetr"}
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
MAX_UPLOAD_FILE_BYTES = 50 * 1024**3
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
    except FileNotFoundError as exc:
        raise CustomModelError(
            "Model folder not found on this PC. Check the full path and try again."
        ) from exc
    except PermissionError as exc:
        raise CustomModelError(
            "Cannot access the selected model folder. Check its permissions and try again."
        ) from exc
    except OSError as exc:
        raise CustomModelError(
            "Selected model folder is unavailable. Check the path or permissions and try again."
        ) from exc
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
    issue = _classifier_inference_issue(source)
    if issue:
        raise CustomModelError(issue)


def _classifier_inference_issue(source: Path) -> str | None:
    inference = source / "inference.py"
    if not inference.is_file() or _reparse_point(inference):
        return "Classification packs must include inference.py"
    try:
        syntax = ast.parse(inference.read_text(encoding="utf-8"), filename=str(inference))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return f"Classification inference.py is not valid Python: {exc}"
    model_class = next(
        (
            node
            for node in syntax.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelInference"
        ),
        None,
    )
    if model_class is None:
        return "inference.py must define an AddaxAI ModelInference class"
    methods = {
        node.name
        for node in model_class.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    required = {"check_gpu", "load_model", "get_crop", "get_classification", "get_class_names"}
    missing = sorted(required - methods)
    if missing:
        return "ModelInference is missing AddaxAI methods: " + ", ".join(missing)
    return None


def _class_names_from_yaml(document: Any) -> dict[str, str] | None:
    if not isinstance(document, dict):
        return None
    raw_names = document.get("names")
    if isinstance(raw_names, list):
        pairs = list(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        pairs: list[tuple[int, Any]] = []
        for key, value in raw_names.items():
            if isinstance(key, bool):
                return None
            if isinstance(key, int):
                class_id = key
            elif isinstance(key, str) and key.isdecimal():
                class_id = int(key)
            else:
                return None
            pairs.append((class_id, value))
        pairs.sort(key=lambda pair: pair[0])
    else:
        return None
    if not pairs or [key for key, _ in pairs] != list(range(len(pairs))):
        return None
    if not all(isinstance(name, str) and name.strip() for _, name in pairs):
        return None
    return {str(class_id): name.strip() for class_id, name in pairs}


def _contains_rtdetrv2_signature(document: Any) -> bool:
    if isinstance(document, dict):
        for key, value in document.items():
            normalized = str(key).replace("_", "").lower()
            if normalized == "rtdetrtransformerv2":
                return True
            if _contains_rtdetrv2_signature(value):
                return True
    elif isinstance(document, list):
        return any(_contains_rtdetrv2_signature(value) for value in document)
    return False


def _read_small_yaml(path: Path, *, limit: int = 1024 * 1024) -> Any:
    if path.stat().st_size > limit:
        raise CustomModelError(f"YAML file is too large to inspect safely: {path.name}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CustomModelError(f"Could not parse YAML file {path.name}: {exc}") from exc


def _manifest_class_names(document: dict[str, Any]) -> dict[str, str] | None:
    value = document.get("class_names")
    if isinstance(value, list):
        if all(isinstance(name, str) and name.strip() for name in value):
            return {str(index): name.strip() for index, name in enumerate(value)}
        return None
    if isinstance(value, dict):
        normalized: dict[str, str] = {}
        for key, name in value.items():
            if not isinstance(key, str) or not key.isdecimal():
                return None
            if not isinstance(name, str) or not name.strip():
                return None
            normalized[key] = name.strip()
        return normalized or None
    return None


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
    backend = None
    if model_type == "classification":
        env = _source_value(payload, source_manifest, "env")
    else:
        backend = _source_value(payload, source_manifest, "detector_backend")
        if backend not in _DETECTOR_ENVIRONMENTS:
            raise CustomModelError(
                "New custom detection registrations support YOLO or official RT-DETRv2"
            )
        env = _DETECTOR_ENVIRONMENTS[backend]
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
        # Model class/variant only select legacy RF-DETR/RT-DETR builds, which
        # new registrations do not support. Reject rather than drop them.
        if getattr(payload, "detector_model_class", None) or getattr(
            payload, "detector_model_variant", None
        ):
            raise CustomModelError(
                "detector_model_class and detector_model_variant apply only to legacy "
                "RF-DETR/RT-DETR packs; YOLO and RT-DETRv2 registrations do not use them"
            )
        class_names = _source_value(payload, source_manifest, "class_names")
        metadata.update(
            detector_backend=backend,
            class_names=class_names,
            detector_config_fname=_source_value(payload, source_manifest, "detector_config_fname"),
            classification_uses_detection_classes=bool(
                getattr(payload, "classification_uses_detection_classes", False)
            ),
        )
        if backend == "rtdetrv2":
            template_name = getattr(payload, "detector_config_template", None)
            config_name = metadata.get("detector_config_fname")
            if config_name and template_name:
                raise CustomModelError(
                    "Choose an existing RT-DETRv2 config or a generated architecture "
                    "template, not both"
                )
            if template_name:
                if template_name not in _available_rtdetrv2_templates():
                    raise CustomModelError(
                        "Choose a supported RT-DETRv2 architecture template"
                    )
                names = _manifest_class_names({"class_names": class_names})
                if not names:
                    raise CustomModelError(
                        "RT-DETRv2 config generation requires class labels from a "
                        "dataset YAML or your input"
                    )
                config_name = "addaxai-generated-rtdetrv2.yml"
                metadata["detector_config_fname"] = config_name
            if not config_name:
                raise CustomModelError(
                    "RT-DETRv2 packs require an existing YAML config or a supported "
                    "architecture template"
                )
            if not template_name:
                config_relative = _safe_relative_file(
                    str(config_name), field_name="detector_config_fname"
                )
                config_path = source / config_relative
                if (
                    not config_path.is_file()
                    or config_path.suffix.lower() not in {".yml", ".yaml"}
                    or _reparse_point(config_path)
                ):
                    raise CustomModelError(
                        "RT-DETRv2 detector_config_fname must point to an existing YAML file"
                    )
                config_document = _read_small_yaml(config_path)
                if not isinstance(config_document, dict) or not _contains_rtdetrv2_signature(
                    config_document
                ):
                    raise CustomModelError(
                        "RT-DETRv2 detector_config_fname must be a valid RT-DETRv2 YAML config"
                    )
                metadata["detector_config_fname"] = config_relative.as_posix()

    try:
        manifest = ModelManifest.model_validate(metadata)
        manifest.model_category = model_type
        if (
            model_type == "detection"
            and manifest.classification_uses_detection_classes
            and not uses_detection_classes_for_classification(manifest)
        ):
            raise CustomModelError(
                "Using detection output as classification requires unique class names and "
                "canonical numeric class IDs"
            )
        manifest.model_category = None
        return manifest
    except CustomModelError:
        raise
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
        "classification_uses_detection_classes": (
            uses_detection_classes_for_classification(manifest)
        ),
        "local_only": manifest.local_only,
        "managed": manifest.managed,
    }


def _available_rtdetrv2_templates() -> list[str]:
    return list(_bundled_rtdetrv2_templates())


@functools.cache
def _bundled_rtdetrv2_templates() -> tuple[str, ...]:
    """Parse the bundled templates once; they are fixed for the process lifetime."""
    template_dir = (
        Path(__file__).resolve().parent
        / "third_party"
        / "rtdetrv2_pytorch"
        / "configs"
        / "rtdetrv2"
    )
    names: list[str] = []
    for path in template_dir.glob("rtdetrv2_*.yml"):
        if not path.is_file():
            continue
        document = _read_small_yaml(path)
        if isinstance(document, dict) and isinstance(document.get("__include__"), list):
            names.append(path.name)
    return tuple(sorted(names))


def _generated_rtdetrv2_yaml(
    template_name: str, class_names: dict[str, str] | list[str] | None
) -> str:
    if template_name not in _available_rtdetrv2_templates():
        raise CustomModelError("Choose a supported RT-DETRv2 architecture template")
    names = _manifest_class_names({"class_names": class_names})
    if not names:
        raise CustomModelError("RT-DETRv2 config generation requires class labels")
    source_root = Path(__file__).resolve().parent / "third_party" / "rtdetrv2_pytorch"
    template_path = source_root / "configs" / "rtdetrv2" / template_name
    document = _read_small_yaml(template_path)
    includes = document.get("__include__", [])
    if not isinstance(includes, list) or not includes:
        raise CustomModelError("Supported RT-DETRv2 template has no safe include list")
    safe_includes: list[str] = []
    for entry in includes:
        if not isinstance(entry, str) or not entry.strip():
            raise CustomModelError("RT-DETRv2 template has an invalid include")
        include_path = (template_path.parent / entry).resolve()
        if not _is_within(include_path, source_root) or not include_path.is_file():
            raise CustomModelError("RT-DETRv2 template include is outside the bundled source")
        safe_includes.append(
            "../third_party/rtdetr/rtdetrv2_pytorch/"
            + include_path.relative_to(source_root).as_posix()
        )
    document["__include__"] = safe_includes
    document["num_classes"] = len(names)
    rtdetr_config = document.get("RTDETR") or {}
    if not isinstance(rtdetr_config, dict):
        raise CustomModelError("RT-DETRv2 template has an invalid detector configuration")
    backbone = rtdetr_config.get("backbone", "PResNet")
    if backbone not in {"PResNet", "HGNetv2"}:
        raise CustomModelError("RT-DETRv2 template uses an unsupported backbone")
    backbone_config = document.get(backbone) or {}
    if not isinstance(backbone_config, dict):
        raise CustomModelError("RT-DETRv2 template has an invalid backbone configuration")
    document[backbone] = {**backbone_config, "pretrained": False}
    # The bundled COCO dataset include remaps class IDs to the COCO taxonomy.
    # Custom class_names are already indexed from zero, so preserve those IDs.
    document["remap_mscoco_category"] = False
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


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

    def rtdetrv2_templates(self) -> list[str]:
        return _available_rtdetrv2_templates()

    def _upload_root(self) -> Path:
        return self.models_dir.parent / ".custom-model-imports"

    def create_upload_session(self) -> tuple[str, Path]:
        root = self._upload_root()
        _reject_reparse_ancestors(root)
        root.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired_upload_sessions()
        upload_id = uuid.uuid4().hex
        session = root / upload_id
        session.mkdir()
        return upload_id, session

    def cleanup_expired_upload_sessions(self, *, max_age_seconds: int = 24 * 60 * 60) -> None:
        root = self._upload_root()
        if not root.is_dir() or _reparse_point(root):
            return
        cutoff = time.time() - max_age_seconds
        for session in root.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", session.name):
                continue
            if _reparse_point(session):
                continue
            try:
                stale = session.stat().st_mtime < cutoff
            except OSError:
                continue
            if stale and session.is_dir() and _is_within(session, root):
                shutil.rmtree(session)

    def _upload_session(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise CustomModelError("Invalid upload session")
        root = self._upload_root()
        session = root / upload_id
        _reject_reparse_ancestors(session)
        if not session.is_dir() or not _is_within(session, root):
            raise FileNotFoundError("Upload session has expired; choose the weight file again")
        return session

    def begin_upload_file(self, upload_id: str, filename: str) -> tuple[Path, Path]:
        session = self._upload_session(upload_id)
        try:
            relative = _safe_relative_file(filename, field_name="filename")
        except CustomModelError:
            self.discard_upload_session(upload_id)
            raise
        if len(relative.parts) != 1:
            self.discard_upload_session(upload_id)
            raise CustomModelError("Choose individual files; folder paths are not accepted")
        target = session / relative
        _reject_reparse_ancestors(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if _path_exists(target):
            self.discard_upload_session(upload_id)
            raise CustomModelError(
                "A file with this name is already in the selected model pack"
            )
        temporary = target.with_name(f".{target.name}.upload-{uuid.uuid4().hex}.part")
        return target, temporary

    def finish_upload_file(self, upload_id: str, target: Path, temporary: Path) -> None:
        session = self._upload_session(upload_id)
        if not _is_within(target, session) or not _is_within(temporary, session):
            self.discard_upload_session(upload_id)
            raise CustomModelError("Uploaded file must remain inside its upload session")
        _reject_reparse_ancestors(target)
        if not temporary.is_file() or _reparse_point(temporary):
            self.discard_upload_session(upload_id)
            raise CustomModelError("Uploaded file was not written safely")
        os.replace(temporary, target)

    def discard_upload_session(self, upload_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            return
        session = self._upload_root() / upload_id
        if _path_exists(session):
            _reject_reparse_ancestors(session)
            if not _is_within(session, self._upload_root()):
                raise CustomModelError("Upload session escaped its staging folder")
            shutil.rmtree(session)

    def inspect(self, source_path: str) -> dict[str, Any]:
        """Inspect a source pack without copying it or reading model weights."""
        source = _resolve_source(source_path, self.models_dir)
        source_manifest = _read_source_manifest(source)
        environments = _available_environments()
        warnings: list[str] = []

        file_rows: list[dict[str, Any]] = []
        for current, directories, files in os.walk(source, followlinks=False):
            directories.sort()
            for filename in sorted(files):
                path = Path(current) / filename
                try:
                    info = path.lstat()
                except OSError as exc:
                    raise CustomModelError(
                        f"Could not inspect model file {path.name}: {exc}"
                    ) from exc
                if not stat.S_ISREG(info.st_mode):
                    continue
                file_rows.append(
                    {
                        "path": path.relative_to(source).as_posix(),
                        "size_bytes": info.st_size,
                    }
                )
        file_rows.sort(key=lambda row: row["path"].casefold())
        files_truncated = len(file_rows) > 2000
        files_to_return = file_rows[:2000]
        total_size_bytes = sum(row["size_bytes"] for row in file_rows)
        if files_truncated:
            warnings.append(
                "Only the first 2,000 files are listed; all files remain in the pack summary."
            )

        weights = [
            row["path"]
            for row in file_rows
            if Path(row["path"]).suffix.lower() in _WEIGHT_SUFFIXES
        ]
        suggested_model_fname = weights[0] if len(weights) == 1 else None

        inference_path = source / "inference.py"
        classifier_issue = _classifier_inference_issue(source)
        classifier_compatible = classifier_issue is None
        if inference_path.exists() and classifier_issue:
            warnings.append(classifier_issue)

        rtdetrv2_configs: list[str] = []
        dataset_candidates: list[dict[str, Any]] = []
        for row in file_rows:
            relative = row["path"]
            path = source / Path(relative)
            if path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            try:
                document = _read_small_yaml(path)
            except CustomModelError as exc:
                warnings.append(str(exc))
                continue
            if _contains_rtdetrv2_signature(document):
                rtdetrv2_configs.append(relative)
            class_names = _class_names_from_yaml(document)
            if class_names:
                dataset_candidates.append({"path": relative, "class_names": class_names})

        rtdetrv2_configs.sort(key=str.casefold)
        dataset_candidates.sort(key=lambda item: item["path"].casefold())

        manifest_backend = source_manifest.get("detector_backend")
        allowed_backends = {"yolo", "rfdetr", "rtdetr", "rtdetrv2"}
        explicit_backend = manifest_backend if manifest_backend in allowed_backends else None
        if manifest_backend and manifest_backend not in allowed_backends | {"megadetector"}:
            warnings.append("Source manifest contains an unsupported detector_backend value.")
        if manifest_backend == "megadetector":
            warnings.append("MegaDetector catalog packs cannot be registered as custom packs.")

        explicit_detection = explicit_backend is not None or source_manifest.get(
            "model_category", source_manifest.get("type")
        ) == "detection"
        detection_evidence = bool(
            rtdetrv2_configs or explicit_detection or (weights and not classifier_compatible)
        )
        type_candidates: list[str] = []
        if classifier_compatible:
            type_candidates.append("classification")
        if detection_evidence:
            type_candidates.append("detection")
        if not type_candidates:
            warnings.append(
                "No recognizable model type was found. The pack needs a supported weight file "
                "and, for classification, an AddaxAI-compatible inference.py."
            )
        suggested_type = type_candidates[0] if len(type_candidates) == 1 else None

        detector_config_candidates = list(rtdetrv2_configs)
        manifest_config = source_manifest.get("detector_config_fname")
        if isinstance(manifest_config, str):
            try:
                relative_config = _safe_relative_file(
                    manifest_config, field_name="detector_config_fname"
                )
                config_path = source / relative_config
                if (
                    config_path.is_file()
                    and config_path.suffix.lower() in {".yaml", ".yml"}
                    and not _reparse_point(config_path)
                ):
                    config_name = relative_config.as_posix()
                    if config_name not in detector_config_candidates:
                        detector_config_candidates.append(config_name)
            except CustomModelError:
                warnings.append(
                    "Source manifest detector_config_fname is not a safe relative file."
                )
        detector_config_candidates.sort(key=str.casefold)
        suggested_detector_config = (
            detector_config_candidates[0] if len(detector_config_candidates) == 1 else None
        )

        recognized_backends = set()
        if explicit_backend in _DETECTOR_ENVIRONMENTS:
            recognized_backends.add(explicit_backend)
        if rtdetrv2_configs:
            recognized_backends.add("rtdetrv2")
        if explicit_backend in {"rfdetr", "rtdetr"}:
            warnings.append(
                "This pack declares a legacy RF-DETR or RT-DETR backend. New custom registrations "
                "support YOLO and official RT-DETRv2; existing registered legacy models remain "
                "available."
            )
        if len(recognized_backends) > 1:
            detector_backend_candidates = ["yolo", "rtdetrv2"]
        elif recognized_backends:
            detector_backend_candidates = sorted(recognized_backends)
        elif detection_evidence:
            detector_backend_candidates = ["yolo", "rtdetrv2"]
        else:
            detector_backend_candidates = []
        backend_conflict = len(recognized_backends) > 1 or (
            explicit_backend in {"rfdetr", "rtdetr"} and bool(rtdetrv2_configs)
        )
        if backend_conflict:
            suggested_backend = None
            warnings.append(
                "Source manifest detector backend conflicts with the RT-DETRv2 config; "
                "choose the backend explicitly."
            )
        elif len(recognized_backends) == 1:
            suggested_backend = next(iter(recognized_backends))
        elif (
            explicit_backend in _DETECTOR_ENVIRONMENTS
            and explicit_backend in detector_backend_candidates
        ):
            suggested_backend = explicit_backend
        elif len(detector_backend_candidates) == 1:
            suggested_backend = detector_backend_candidates[0]
        else:
            suggested_backend = None
        suggested_detector_model_class = None
        source_detector_class = source_manifest.get("detector_model_class")
        if isinstance(source_detector_class, str) and source_detector_class.strip():
            suggested_detector_model_class = source_detector_class.strip()
        suggested_detector_model_variant = None
        source_detector_variant = source_manifest.get("detector_model_variant")
        if isinstance(source_detector_variant, str) and source_detector_variant.strip():
            suggested_detector_model_variant = source_detector_variant.strip()

        suggested_env = None
        suggested_env_source = None
        source_env = source_manifest.get("env")
        if backend_conflict:
            if source_env:
                warnings.append(
                    "Choose the inference environment after resolving the detector backend "
                    "conflict."
                )
        elif suggested_backend in _DETECTOR_ENVIRONMENTS:
            backend_env = _DETECTOR_ENVIRONMENTS[suggested_backend]
            if backend_env in environments:
                suggested_env = backend_env
                suggested_env_source = (
                    "rtdetrv2_config" if suggested_backend == "rtdetrv2" else "detector_backend"
                )
        elif isinstance(source_env, str) and source_env in environments:
            suggested_env = source_env
            suggested_env_source = "manifest"
        elif source_env:
            warnings.append("Source manifest environment is not available in this installation.")

        suggested_class_names = _manifest_class_names(source_manifest)
        suggested_names_source = "manifest.json" if suggested_class_names else None
        if suggested_class_names is None and len(dataset_candidates) == 1:
            suggested_class_names = dataset_candidates[0]["class_names"]
            suggested_names_source = dataset_candidates[0]["path"]
        elif suggested_class_names is None and len(dataset_candidates) > 1:
            distinct_names = {
                json.dumps(candidate["class_names"], sort_keys=True)
                for candidate in dataset_candidates
            }
            if len(distinct_names) == 1:
                suggested_class_names = dataset_candidates[0]["class_names"]
                suggested_names_source = dataset_candidates[0]["path"]
            else:
                warnings.append(
                    "Multiple dataset YAML files contain different class names; choose the "
                    "correct labels before registration."
                )

        missing_required: list[str] = []
        if len(type_candidates) != 1:
            missing_required.append("Choose whether this pack is for detection or classification.")
        if len(weights) != 1:
            missing_required.append(
                "Choose one model weight file."
                if weights
                else "Add a supported model weight file."
            )
        if suggested_env is None:
            missing_required.append("Choose a packaged inference environment.")
        if "detection" in type_candidates and suggested_backend is None:
            missing_required.append("Choose the detection backend.")
        if (
            suggested_backend == "rtdetrv2"
            and suggested_detector_config is None
        ):
            missing_required.append(
                "Choose an existing RT-DETRv2 YAML config or a known architecture template."
            )
        if (
            suggested_class_names is None
            and len(dataset_candidates) > 1
            and len(
                {json.dumps(row["class_names"], sort_keys=True) for row in dataset_candidates}
            )
            > 1
        ):
            missing_required.append(
                "Choose the dataset YAML that defines this model's class labels."
            )

        return {
            "source_path": str(source),
            "type_candidates": type_candidates,
            "suggested_type": suggested_type,
            "weights": weights,
            "suggested_model_fname": suggested_model_fname,
            "detector_backend_candidates": detector_backend_candidates,
            "suggested_detector_backend": suggested_backend,
            "suggested_detector_model_class": suggested_detector_model_class,
            "suggested_detector_model_variant": suggested_detector_model_variant,
            "detector_config_candidates": detector_config_candidates,
            "detector_config_templates": self.rtdetrv2_templates(),
            "suggested_detector_config_fname": suggested_detector_config,
            "environments": environments,
            "suggested_env": suggested_env,
            "suggested_env_source": suggested_env_source,
            "dataset_candidates": dataset_candidates,
            "suggested_class_names": suggested_class_names,
            "suggested_class_names_source": suggested_names_source,
            "source_manifest_detected": (source / "manifest.json").is_file(),
            "classifier_inference_compatible": classifier_compatible,
            "files": files_to_return,
            "total_file_count": len(file_rows),
            "total_size_bytes": total_size_bytes,
            "files_truncated": files_truncated,
            "missing_required": missing_required,
            "warnings": warnings,
        }

    def list_models(self) -> list[dict[str, Any]]:
        manifests = self.manifest_manager.load_manifests(force_refresh=True)
        return sorted(
            (_model_record(item) for item in manifests.values() if item.managed),
            key=lambda row: (row["type"] or "", row["friendly_name"].casefold()),
        )

    def create(self, payload: Any) -> dict[str, Any]:
        upload_id = getattr(payload, "upload_id", None)
        source_path = (
            payload.source_path
            if payload.source_path
            else str(self._upload_session(upload_id))
        )
        # A failed registration keeps the staged upload so the user can fix
        # the form and retry without re-sending multi-GB weights. Closing
        # the dialog or the 24 h expiry sweep removes it otherwise.
        record = self._create_from_source(payload, source_path)
        if upload_id:
            self.discard_upload_session(upload_id)
        return record

    def _create_from_source(self, payload: Any, source_path: str) -> dict[str, Any]:
        source = _resolve_source(source_path, self.models_dir)
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
                template_name = getattr(payload, "detector_config_template", None)
                if template_name and manifest.detector_config_fname:
                    generated_config = temp_dir / manifest.detector_config_fname
                    generated_config.write_text(
                        _generated_rtdetrv2_yaml(template_name, manifest.class_names),
                        encoding="utf-8",
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
            self.manifest_manager.load_manifests(force_refresh=True)
            return _model_record(self.manifest_manager.get_model(model_id))

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
