"""
Model manifest schema for ML models.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear documentation

Based on proven patterns from streamlit-AddaxAI.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ModelManifest(BaseModel):
    """
    Model manifest defining all metadata and configuration for an ML model.

    This schema is used to define both detection and classification models.
    Manifests are stored in JSON format and loaded at runtime.
    """

    # Identity
    model_id: str = Field(..., description="Unique identifier for the model (e.g., 'MD5A-0-0', 'NAM-ADS-v1')")
    friendly_name: str = Field(..., description="Human-readable display name")
    emoji: str = Field(..., description="Emoji icon for UI display")
    type: Literal["detection", "classification"] = Field(..., description="Model type")

    # Environment & Model Files
    env: str = Field(..., description="Environment name (points to envs/{env}/ directory)")
    model_fname: str = Field(..., description="Model weights filename (e.g., 'md_v5a.0.0.pt')")
    hf_repo: str | None = Field(None, description="HuggingFace repo (defaults to Addax-Data-Science/{model_id})")

    # Metadata
    description: str = Field(..., description="Detailed description of the model")
    developer: str = Field(..., description="Organization or person who created the model")
    citation: str | None = Field(None, description="DOI or URL for academic citation")
    license: str | None = Field(None, description="License type or URL")
    info_url: str = Field(..., description="Website with more information")
    min_app_version: str = Field(..., description="Minimum AddaxAI version required (e.g., '0.1.0')")

    # Runtime Configuration
    confidence_threshold: float = Field(0.1, description="Minimum confidence threshold for detections")

    # Classification-specific
    species_list: list[str] | None = Field(None, description="List of species this classifier can recognize")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "model_id": "MD5A-0-0",
                "friendly_name": "MegaDetector 5a",
                "emoji": "🔍",
                "type": "detection",
                "env": "megadetector",
                "model_fname": "md_v5a.0.0.pt",
                "hf_repo": "Addax-Data-Science/MD5A-0-0",
                "description": "MegaDetector v5a for animal detection in camera trap images",
                "developer": "Dan Morris",
                "license": "MIT",
                "info_url": "https://github.com/agentmorris/MegaDetector",
                "min_app_version": "0.1.0",
                "confidence_threshold": 0.1,
            }
        }
