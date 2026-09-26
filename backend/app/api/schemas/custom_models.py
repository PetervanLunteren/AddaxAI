"""Request and response models for host-managed local model packs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml.schemas.model_manifest import DetectorBackend, ModelRegion


class CustomModelCreate(BaseModel):
    """Register a model pack copied from a user-selected local directory."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["detection", "classification"]
    source_path: str | None = Field(default=None, min_length=1, max_length=4096)
    upload_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    friendly_name: str = Field(min_length=1, max_length=120)
    env: str = Field(min_length=1, max_length=80)
    model_fname: str | None = Field(default=None, max_length=512)
    description: str = Field(default="", max_length=1000)
    description_short: str | None = Field(default=None, max_length=240)
    developer: str = Field(default="", max_length=160)
    owner: str | None = Field(default=None, max_length=160)
    citation: str | None = Field(default=None, max_length=1000)
    license: str | None = Field(default=None, max_length=160)
    info_url: str = Field(default="", max_length=2048)
    emoji: str | None = Field(default=None, max_length=16)
    region: ModelRegion | None = None
    full_image_cls: bool | None = None
    example_image_url: str | None = Field(default=None, max_length=2048)
    detector_backend: DetectorBackend | None = None
    class_names: dict[str, str] | list[str] | None = None
    detector_model_class: str | None = Field(default=None, max_length=80)
    detector_model_variant: str | None = Field(default=None, max_length=80)
    detector_config_fname: str | None = Field(default=None, max_length=512)
    detector_config_template: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def _require_one_source(self):
        if bool(self.source_path) == bool(self.upload_id):
            raise ValueError("Provide exactly one of source_path or upload_id")
        return self


class CustomModelInspectRequest(BaseModel):
    """Inspect a local pack without copying or opening its model weights."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1, max_length=4096)


class CustomModelFileInfo(BaseModel):
    """Relative file path and size that registration will copy."""

    path: str
    size_bytes: int


class CustomModelDatasetCandidate(BaseModel):
    """Safely parsed class names from a dataset YAML file."""

    path: str
    class_names: dict[str, str]


class CustomModelInspectResponse(BaseModel):
    """Read-only pack inspection results and unresolved choices."""

    source_path: str
    type_candidates: list[Literal["detection", "classification"]]
    suggested_type: Literal["detection", "classification"] | None = None
    weights: list[str]
    suggested_model_fname: str | None = None
    detector_backend_candidates: list[Literal["yolo", "rfdetr", "rtdetr", "rtdetrv2"]]
    suggested_detector_backend: Literal["yolo", "rfdetr", "rtdetr", "rtdetrv2"] | None = None
    suggested_detector_model_class: str | None = None
    suggested_detector_model_variant: str | None = None
    detector_config_candidates: list[str]
    detector_config_templates: list[str] = Field(default_factory=list)
    suggested_detector_config_fname: str | None = None
    environments: list[str]
    suggested_env: str | None = None
    suggested_env_source: Literal["manifest", "rtdetrv2_config"] | None = None
    dataset_candidates: list[CustomModelDatasetCandidate]
    suggested_class_names: dict[str, str] | None = None
    suggested_class_names_source: str | None = None
    source_manifest_detected: bool
    classifier_inference_compatible: bool
    files: list[CustomModelFileInfo]
    total_file_count: int
    total_size_bytes: int
    files_truncated: bool = False
    missing_required: list[str]
    warnings: list[str]


class CustomModelUpdate(BaseModel):
    """Display metadata for a local pack; inference behavior is immutable."""

    model_config = ConfigDict(extra="forbid")

    friendly_name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    description_short: str | None = Field(default=None, max_length=240)
    developer: str | None = Field(default=None, max_length=160)
    owner: str | None = Field(default=None, max_length=160)
    citation: str | None = Field(default=None, max_length=1000)
    license: str | None = Field(default=None, max_length=160)
    info_url: str | None = Field(default=None, max_length=2048)
    emoji: str | None = Field(default=None, max_length=16)
    region: ModelRegion | None = None
    example_image_url: str | None = Field(default=None, max_length=2048)


class CustomModelInfo(BaseModel):
    """Editable managed-model metadata returned by the management surface."""

    model_id: str
    type: Literal["detection", "classification"]
    friendly_name: str
    description: str
    description_short: str | None = None
    developer: str
    owner: str | None = None
    citation: str | None = None
    license: str | None = None
    info_url: str
    emoji: str | None = None
    env: str
    model_fname: str
    region: str | None = None
    full_image_cls: bool = False
    example_image_url: str | None = None
    detector_backend: str | None = None
    detector_model_class: str | None = None
    detector_model_variant: str | None = None
    detector_config_fname: str | None = None
    class_names: dict[str, str] | None = None
    local_only: bool
    managed: bool


class CustomModelsResponse(BaseModel):
    models: list[CustomModelInfo]
    environments: list[str]
