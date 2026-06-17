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
    # Commit SHA of the HuggingFace repo at the time the model was
    # downloaded. Stored locally only (not in the central catalog) and
    # used by ModelCatalogUpdater.sync() to detect when the upstream
    # repo has moved past what's on disk. None on legacy installs that
    # pre-date drift detection; absence is treated as "unknown but
    # valid" rather than as drift.
    hf_revision_sha: str | None = None

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
