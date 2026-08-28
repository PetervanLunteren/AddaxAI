"""
Model manifest schema for ML models.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Clear documentation

Based on proven patterns from streamlit-AddaxAI.
"""

from typing import Literal

from pydantic import BaseModel

# Region the cls model is trained for. Drives how the classification
# dropdown groups its options. None for detection / embedding models
# (region-agnostic) and as a fallback for legacy cls manifests.
ModelRegion = Literal[
    "global", "africa", "americas", "asia", "europe", "oceania"
]

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
