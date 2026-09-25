"""Request and response models for host-managed local model packs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ml.schemas.model_manifest import DetectorBackend, ModelRegion


class CustomModelCreate(BaseModel):
    """Register a model pack copied from a user-selected local directory."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["detection", "classification"]
    source_path: str = Field(min_length=1, max_length=4096)
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
