"""
Model manifest schema for ML models.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Clear documentation

Based on proven patterns from streamlit-AddaxAI.
"""

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, model_validator

# Region the cls model is trained for. Drives how the classification
# dropdown groups its options. None for detection / embedding models
# (region-agnostic) and as a fallback for legacy cls manifests.
ModelRegion = Literal[
    "global", "africa", "americas", "asia", "europe", "oceania"
]

# Detection backends are an explicit allow-list. Catalog data cannot select
# arbitrary Python imports or executables.
DetectorBackend = Literal["megadetector", "yolo", "rfdetr", "rtdetr", "rtdetrv2"]
_RTDETR_VARIANTS = frozenset({"MDV6-apa-rtdetr-c", "MDV6-apa-rtdetr-e"})
_RFDETR_MODEL_CLASSES = frozenset(
    {
        "RFDETRBase", "RFDETRNano", "RFDETRSmall", "RFDETRMedium",
        "RFDETRLarge", "RFDETRXLarge", "RFDETR2XLarge", "RFDETRSegNano",
        "RFDETRSegSmall", "RFDETRSegMedium", "RFDETRSegLarge",
        "RFDETRSegXLarge", "RFDETRSeg2XLarge", "RFDETRKeypointPreview",
        "RFDETRSegPreview",
    }
)

# HuggingFace org that hosts the model repos. A manifest may override the
# repo with an explicit `hf_repo`; everything else follows the convention
# `<DEFAULT_HF_ORG>/<model_id>`.
DEFAULT_HF_ORG = "Addax-Data-Science"


def resolve_hf_repo(model_id: str, hf_repo: str | None = None) -> str:
    """
    Return the HuggingFace repo id for a model.

    Always go through this helper rather than rebuilding the convention
    at the call site. Forgetting the `hf_repo or ...` half is exactly how
    the catalog's taxonomy download ended up pinned to the default org
    and silently 404'ing for the one model that overrides it.
    """
    return hf_repo or f"{DEFAULT_HF_ORG}/{model_id}"


class ModelManifest(BaseModel):
    """
    Model manifest defining all metadata and configuration for an ML model.

    This schema is used to define both detection and classification models.
    Manifests are stored in JSON format and loaded at runtime.
    """

    # Identity
    model_id: str
    friendly_name: str
    # Optional decorative/regional icon. Classification models carry a
    # regional flag; detection / embedding models omit it.
    emoji: str | None = None
    type: str | None = (
        None  # Unused legacy field, kept for backward compatibility with existing manifests
    )
    model_category: str | None = (
        None  # "detection"/"classification"/"embedding" - set during loading
    )

    # Environment & Model Files
    env: str
    model_fname: str
    hf_repo: str | None = None
    # A local manifest.json holds nothing beyond its catalog entry. Whether
    # an install still matches upstream is answered by comparing the files
    # themselves (model_storage.find_stale_files), so there is no recorded
    # state here to fall out of date or to be overwritten by write_manifest.

    # Metadata
    description: str
    description_short: str | None = None
    developer: str
    owner: str | None = None
    citation: str | None = None
    license: str | None = None
    info_url: str
    min_app_version: str

    # Classification-specific
    species_list: list[str] | None = None
    # Region the model is trained for. Used to group cls models in the
    # UI dropdown. None for detection / embedding (region-agnostic).
    region: ModelRegion | None = None
    # Full-image classifier flag. When True, the model labels the whole
    # frame and the worker skips MegaDetector entirely; a synthetic
    # detection covering the full image is fed straight into the
    # classification phase. See app.ml.full_image_detection.
    full_image_cls: bool = False
    # Picture of what the model expects to see, shown in the model info
    # sheet. A URL only, the image never lives in the repo or the app.
    # Meant for models with a specific setup (a drift-fence bucket, a
    # baited tray) so a user can compare it with their own photos.
    example_image_url: str | None = None

    # Detection backend configuration. Legacy manifests retain MegaDetector
    # by default; non-MD packages are loaded in a constrained subprocess.
    detector_backend: DetectorBackend = "megadetector"
    class_names: dict[str, str] | list[str] | None = None
    detector_model_class: str | None = None
    detector_model_variant: str | None = None
    detector_config_fname: str | None = None

    # User-managed packs live only in the local models directory. Catalog
    # sync must preserve their manifest and these fields are never written to
    # the central models.json catalog.
    local_only: bool = False
    managed: bool = False
    managed_created_at: str | None = None
    weights_sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_detector_manifest(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            raise TypeError("model manifest must be an object")
        backend = values.get("detector_backend", "megadetector")
        if backend == "rfdetr":
            model_class = values.get("detector_model_class") or "RFDETRMedium"
            if model_class not in _RFDETR_MODEL_CLASSES:
                raise ValueError("detector_model_class is not supported for rfdetr")
            values["detector_model_class"] = model_class
        elif values.get("detector_model_class"):
            raise ValueError("detector_model_class is only valid for rfdetr manifests")

        if backend == "rtdetr":
            variant = values.get("detector_model_variant") or "MDV6-apa-rtdetr-c"
            if variant not in _RTDETR_VARIANTS:
                raise ValueError("detector_model_variant is not supported for rtdetr")
            values["detector_model_variant"] = variant
        elif values.get("detector_model_variant"):
            raise ValueError("detector_model_variant is only valid for rtdetr manifests")

        config_fname = values.get("detector_config_fname")
        if backend == "rtdetrv2":
            if not isinstance(config_fname, str) or not config_fname.strip():
                raise ValueError("detector_config_fname is required for rtdetrv2 manifests")
            normalized = config_fname.replace("\\", "/")
            posix_path = PurePosixPath(normalized)
            windows_path = PureWindowsPath(config_fname)
            if (
                posix_path.is_absolute()
                or windows_path.is_absolute()
                or windows_path.drive
                or any(part in {"", ".", ".."} for part in normalized.split("/"))
            ):
                raise ValueError("detector_config_fname must be a safe relative path")
        elif config_fname:
            raise ValueError("detector_config_fname is only valid for rtdetrv2 manifests")

        if values.get("local_only") and values.get("hf_repo"):
            raise ValueError("local_only manifests must not specify hf_repo")
        if backend in {"rtdetr", "rtdetrv2"} and not values.get("local_only", False):
            raise ValueError(f"{backend} manifests must be local_only")

        class_names = values.get("class_names")
        if isinstance(class_names, list):
            if not all(isinstance(name, str) and name.strip() for name in class_names):
                raise ValueError("class_names entries must be non-empty strings")
            values["class_names"] = {str(index): name for index, name in enumerate(class_names)}
        elif class_names is not None and (
            not isinstance(class_names, dict)
            or not all(
                isinstance(key, str) and isinstance(name, str) and name.strip()
                for key, name in class_names.items()
            )
        ):
            raise ValueError("class_names must be a string list or string-to-string map")
        return values

    # Embedding-specific
    embedding_dim: int | None = None  # 384, 768, or 1024
    input_size: int | None = None  # e.g., 224
    torch_hub_model: str | None = None  # e.g., "dinov2_vits14" (for architecture loading)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "model_id": "MD5A-0-0",
                "friendly_name": "MegaDetector 5a",
                "emoji": "🔍",
                "env": "megadetector",
                "model_fname": "md_v5a.0.0.pt",
                "hf_repo": "Addax-Data-Science/MD5A-0-0",
                "description": "MegaDetector v5a for animal detection in camera trap images",
                "developer": "Dan Morris",
                "license": "MIT",
                "info_url": "https://github.com/agentmorris/MegaDetector",
                "min_app_version": "0.1.0",
            }
        }


def uses_detection_classes_for_classification(manifest: ModelManifest) -> bool:
    """Whether this managed custom detector can expose its output classes as labels.

    The same model ID may be selected in both project fields for these packs.
    Classification then reuses the detector's category and confidence; no
    classifier subprocess or second inference pass is needed.
    """
    class_names = manifest.class_names
    names = []
    canonical_ids = False
    if isinstance(class_names, dict):
        canonical_ids = all(
            re.fullmatch(r"(?:0|[1-9]\d*)", class_id)
            for class_id in class_names
        )
        names = [
            name.strip().casefold()
            for name in class_names.values()
            if isinstance(name, str) and name.strip()
        ]
    has_names = (
        bool(names)
        and canonical_ids
        and len(names) == len(class_names)
        and len(set(names)) == len(names)
    )
    return bool(
        manifest.model_category == "detection"
        and manifest.managed
        and manifest.local_only
        and manifest.detector_backend in {"yolo", "rfdetr", "rtdetr", "rtdetrv2"}
        and has_names
    )
