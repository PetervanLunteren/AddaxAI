"""Schemas for project packages: export request, import preview and
result. The package format itself lives in
`app/services/project_package.py`."""

from typing import Literal

from pydantic import BaseModel, Field

from app.api.schemas.export import ExportTarget


class PackageExportRequest(ExportTarget):
    """Body of ``POST /api/projects/{id}/export/package/prepare``. The
    package folder is written into ``output_dir``."""

    # Copy the deployment folders (hidden .addaxai included) into the
    # package, so it can be opened on a computer without the photos.
    include_media: bool = False


class PackageSizeResponse(BaseModel):
    """What the export dialog shows before Start: how many files and how
    many bytes of media the selected deployments hold, and an estimate of
    the bundle's data."""

    deployments: int
    files: int
    media_bytes: int
    data_bytes: int


class PackageExportResult(BaseModel):
    """`job.result` of a finished export."""

    package_dir: str
    deployments: int
    files: int
    includes_media: bool


class PackagePathRequest(BaseModel):
    """Body of ``POST /api/packages/preview``."""

    path: str = Field(..., min_length=1)


class PackageImportRequest(PackagePathRequest):
    """Body of ``POST /api/packages/import``."""

    # Deployments that already exist on this computer: replace them with
    # the package's version, or leave them alone.
    replace_existing: bool = False


class PackageDeploymentPreview(BaseModel):
    id: str
    files: int
    # Already on this computer: skipped or replaced, the user's choice.
    exists: bool


class PackagePreview(BaseModel):
    """What an import would do, computed without writing anything."""

    project_id: str
    project_name: str
    # Same project id already here: deployments are added to it.
    project_exists: bool
    app_version: str
    # current: same schema. older: upgraded on import. newer: refused.
    revision_status: Literal["current", "older", "newer"]
    includes_media: bool
    deployments: list[PackageDeploymentPreview]
    # Why the import is refused. Empty means it can run.
    problems: list[str]
    # Things to know that do not stop the import.
    warnings: list[str]


class PackageImportResult(BaseModel):
    """`job.result` of a finished import."""

    project_id: str
    project_created: bool
    deployments_created: int
    deployments_replaced: int
    deployments_skipped: int
