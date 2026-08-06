"""
Pydantic schemas for the CSV bulk imports (sites and deployment queue).

Both imports share one wire shape so the frontend renders their previews and
their problem lists with one component.

A note on `CsvImportProblem.message`: it must be a CONSTANT string. The
offending cell goes in `value`, never inside the message. The dialog groups
problems by message to collapse "40 rows have the same mistake" into one
line, and an interpolated message cannot group.
"""

from pydantic import BaseModel, Field


class CsvImportProblem(BaseModel):
    """One thing the user has to fix before the import can go ahead."""

    # Line number as a spreadsheet shows it (the header is row 1). None for
    # problems about the file as a whole, which sort first.
    row: int | None = None
    column: str | None = None
    message: str
    value: str | None = None


class SiteImportRow(BaseModel):
    """One site row that parsed cleanly. Mirrors the columns of the site CSV."""

    row: int
    name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None
    habitat_type: str | None = None
    notes: str | None = None


class SiteImportPreview(BaseModel):
    """Dry run of a site CSV. Empty `problems` means the import may go ahead."""

    rows: list[SiteImportRow]
    problems: list[CsvImportProblem]


class DeploymentImportRow(BaseModel):
    """One deployment row that parsed cleanly.

    `image_count` / `video_count` are filled in by validation, from a cheap
    filename-only walk of the folder. They are zero until then.
    """

    row: int
    folder: str
    site: str | None = None
    notes: str | None = None
    image_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)


class DeploymentImportPreview(BaseModel):
    """Dry run of a deployment CSV."""

    rows: list[DeploymentImportRow]
    problems: list[CsvImportProblem]


class CsvImportResult(BaseModel):
    """Outcome of a real import. `imported` is 0 whenever `problems` is not empty:
    the import is all or nothing."""

    imported: int
    problems: list[CsvImportProblem]
