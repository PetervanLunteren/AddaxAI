"""
Site CSV import: parsing and validation.

Split in two on purpose:

- `parse_site_csv` is pure. Shape and format only, no database.
- `validate_site_rows` is everything that has to be true at the moment of
  writing. Both the preview route and the import route call it, which is why
  the import route needs no validator of its own.

A row that fails parsing is left out of the returned rows and reported
instead. Validation still runs on the rows that did parse, so the user sees
every problem in one pass rather than one round per mistake.
"""

import math
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.csv_import import CsvImportProblem, SiteImportRow
from app.api.schemas.site import reject_null_island
from app.models import Site
from app.services.csv_import import RawCsvRow, blank_to_none, read_csv_rows

SITE_REQUIRED_COLUMNS = ("name", "latitude", "longitude")
SITE_OPTIONAL_COLUMNS = ("elevation_m", "habitat_type", "notes")

_MAX_NAME = 255
_MAX_HABITAT = 255
_MAX_NOTES = 1000

# Constant messages, so the dialog can group rows that share a mistake.
NAME_EMPTY = "Name is empty. Every site needs a name."
NAME_TOO_LONG = "Name is longer than 255 characters. Use a shorter name."
LATITUDE_EMPTY = "Latitude is empty. Enter a number between -90 and 90."
LONGITUDE_EMPTY = "Longitude is empty. Enter a number between -180 and 180."
LATITUDE_NOT_A_NUMBER = (
    "Latitude is not a number. Use a dot as the decimal separator, for example 52.1."
)
LONGITUDE_NOT_A_NUMBER = (
    "Longitude is not a number. Use a dot as the decimal separator, for example 5.2."
)
LATITUDE_OUT_OF_RANGE = "Latitude is out of range. Use a number between -90 and 90."
LONGITUDE_OUT_OF_RANGE = "Longitude is out of range. Use a number between -180 and 180."
ELEVATION_NOT_A_NUMBER = (
    "Elevation is not a number. Use digits and a dot, for example 1620 or 1702.5, "
    "with no thousands separator. Leave it empty if you do not know it."
)
HABITAT_TOO_LONG = "Habitat type is longer than 255 characters. Use a shorter value."
NOTES_TOO_LONG = "Notes are longer than 1000 characters. Use a shorter text."
NAME_DUPLICATED_IN_FILE = (
    "This site name is used in more than one row. Site names must be unique. "
    "Change one of them."
)
NAME_ALREADY_IN_PROJECT = (
    "A site with this name already exists in this project. Rename it in the file, "
    "or remove this row."
)


def parse_site_csv(content: bytes) -> tuple[list[SiteImportRow], list[CsvImportProblem]]:
    """Read a site CSV into typed rows. No database access."""
    raw_rows, problems = read_csv_rows(
        content, SITE_REQUIRED_COLUMNS, SITE_OPTIONAL_COLUMNS, "site"
    )

    rows: list[SiteImportRow] = []
    for raw in raw_rows:
        row, row_problems = _parse_row(raw)
        if row is None:
            problems.extend(row_problems)
        else:
            rows.append(row)

    return rows, problems


def validate_site_rows(
    db: Session, project_id: str, rows: list[SiteImportRow]
) -> list[CsvImportProblem]:
    """Check the rows against the project as it is right now.

    Two rules: a name may not repeat inside the file, and it may not already
    exist in the project. Both mirror the `(project_id, name)` unique
    constraint on `sites`, so the import fails here with a readable message
    instead of at the insert with an IntegrityError.
    """
    existing = {
        name
        for (name,) in db.execute(select(Site.name).where(Site.project_id == project_id)).all()
    }
    # Every row of a duplicated name is flagged, not just the later one, so
    # the user does not have to hunt for the row it collides with.
    repeated = {name for name, count in Counter(row.name for row in rows).items() if count > 1}

    problems: list[CsvImportProblem] = []

    for row in rows:
        if row.name in repeated:
            problems.append(
                CsvImportProblem(
                    row=row.row, column="name", message=NAME_DUPLICATED_IN_FILE, value=row.name
                )
            )

        if row.name in existing:
            problems.append(
                CsvImportProblem(
                    row=row.row, column="name", message=NAME_ALREADY_IN_PROJECT, value=row.name
                )
            )

    return problems


def _parse_row(raw: RawCsvRow) -> tuple[SiteImportRow | None, list[CsvImportProblem]]:
    """One CSV record into a typed row, or None plus everything wrong with it."""
    problems: list[CsvImportProblem] = []

    name = raw.values["name"]
    if not name:
        problems.append(CsvImportProblem(row=raw.row, column="name", message=NAME_EMPTY))
    elif len(name) > _MAX_NAME:
        problems.append(
            CsvImportProblem(row=raw.row, column="name", message=NAME_TOO_LONG, value=name)
        )

    latitude = _parse_coordinate(
        raw, "latitude", LATITUDE_EMPTY, LATITUDE_NOT_A_NUMBER, LATITUDE_OUT_OF_RANGE, 90, problems
    )
    longitude = _parse_coordinate(
        raw,
        "longitude",
        LONGITUDE_EMPTY,
        LONGITUDE_NOT_A_NUMBER,
        LONGITUDE_OUT_OF_RANGE,
        180,
        problems,
    )

    if latitude is not None and longitude is not None:
        try:
            reject_null_island(latitude, longitude)
        except ValueError as e:
            problems.append(
                CsvImportProblem(row=raw.row, column="latitude", message=str(e), value=name)
            )

    elevation: float | None = None
    raw_elevation = raw.values["elevation_m"]
    if raw_elevation:
        elevation = _to_float(raw_elevation)
        if elevation is None:
            problems.append(
                CsvImportProblem(
                    row=raw.row,
                    column="elevation_m",
                    message=ELEVATION_NOT_A_NUMBER,
                    value=raw_elevation,
                )
            )

    habitat = raw.values["habitat_type"]
    if len(habitat) > _MAX_HABITAT:
        problems.append(
            CsvImportProblem(row=raw.row, column="habitat_type", message=HABITAT_TOO_LONG)
        )

    notes = raw.values["notes"]
    if len(notes) > _MAX_NOTES:
        problems.append(CsvImportProblem(row=raw.row, column="notes", message=NOTES_TOO_LONG))

    if problems:
        return None, problems

    # Narrowing for the type checker: no problems means both parsed.
    assert latitude is not None and longitude is not None
    return (
        SiteImportRow(
            row=raw.row,
            name=name,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            habitat_type=blank_to_none(habitat),
            notes=blank_to_none(notes),
            tags=raw.tags,
        ),
        [],
    )


def _parse_coordinate(
    raw: RawCsvRow,
    column: str,
    empty_message: str,
    not_a_number_message: str,
    out_of_range_message: str,
    limit: float,
    problems: list[CsvImportProblem],
) -> float | None:
    """Parse one coordinate cell, appending at most one problem."""
    value = raw.values[column]
    if not value:
        problems.append(CsvImportProblem(row=raw.row, column=column, message=empty_message))
        return None

    parsed = _to_float(value)
    if parsed is None:
        problems.append(
            CsvImportProblem(
                row=raw.row, column=column, message=not_a_number_message, value=value
            )
        )
        return None

    if not -limit <= parsed <= limit:
        problems.append(
            CsvImportProblem(
                row=raw.row, column=column, message=out_of_range_message, value=value
            )
        )
        return None

    return parsed


def _to_float(value: str) -> float | None:
    """None when the cell is not a plain finite number.

    `float()` happily accepts "nan" and "inf", which would reach the database
    as coordinates no map can draw, so they are rejected here.
    """
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None
