"""
Model manifest schema for ML models.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear documentation
"""

from typing import Literal

from pydantic import BaseModel, Field


class ModelManifest(BaseModel):
    """
    Model manifest defining all metadata and configuration for an ML model.

    This schema is used to define both detection and classification models.
    Manifests are stored in YAML format and loaded at runtime.
    """

    # Identity
    model_id: str = Field(..., description="Unique identifier for the model (e.g., 'MDV5A', 'EUR-DF-v1-3')")
    friendly_name: str = Field(..., description="Human-readable display name")
    emoji: str = Field(..., description="Emoji icon for UI display")
    type: Literal["detection", "classification"] = Field(..., description="Model type")
    framework: Literal["pytorch", "tensorflow"] = Field(..., description="ML framework used")

    # Model Files
    model_fname: str | None = Field(None, description="Model weights filename (for classification models)")
    package: str | None = Field(None, description="Python package to install (e.g., 'megadetector>=5.0')")
    weights_url: str | None = Field(None, description="URL to download model weights from")
    checksum_sha256: str | None = Field(None, description="SHA256 checksum for weight file verification")

    # Environment Configuration
    environment: str = Field(..., description="Micromamba environment name")
    python_version: str = Field(..., description="Python version for environment (e.g., '3.11')")
    dependencies: list[str] = Field(..., description="List of pip/conda packages to install")

    # Metadata
    description: str = Field(..., description="Detailed description of the model")
    developer: str = Field(..., description="Organization or person who created the model")
    citation: str | None = Field(None, description="DOI or URL for academic citation")
    license: str = Field(..., description="License type or URL")
    info_url: str = Field(..., description="Website with more information")
    min_app_version: str = Field(..., description="Minimum AddaxAI version required")

    # Runtime Configuration
    input_size: tuple[int, int] | None = Field(None, description="Model input size (width, height)")
    batch_size_gpu: int = Field(8, description="Batch size for GPU inference")
    batch_size_cpu: int = Field(2, description="Batch size for CPU inference")
    confidence_threshold: float = Field(0.1, description="Minimum confidence threshold for detections")

    # Classification-specific
    species_list: list[str] | None = Field(None, description="List of species this classifier can recognize")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "model_id": "MDV5A",
                "friendly_name": "MegaDetector v5a",
                "emoji": "🔍",
                "type": "detection",
                "framework": "pytorch",
                "package": "megadetector>=5.0",
                "model_fname": "md_v5a.0.0.pt",
                "environment": "megadetector-env",
                "python_version": "3.11",
                "dependencies": ["megadetector>=5.0"],
                "description": "MegaDetector v5a for animal detection in camera trap images",
                "developer": "Microsoft AI for Earth",
                "license": "MIT",
                "info_url": "https://github.com/agentmorris/MegaDetector",
                "min_app_version": "0.1.0",
                "batch_size_gpu": 8,
                "batch_size_cpu": 2,
                "confidence_threshold": 0.1,
            }
        }
