"""
Deployment CSV import: parsing and validation.

What this creates is a `DeploymentQueue` entry, not a `Deployment`. The
deployment row itself is built later by the detection worker when the queue
runs, which is also where the capture dates come from. That is why the CSV
has no date columns and must never grow any.

Same split as the site importer: `parse_deployment_csv` is pure,
`validate_deployment_rows` is everything that has to be true at the moment of
writing, and both routes call the second one.
"""

import os
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.csv_import import CsvImportProblem, DeploymentImportRow
from app.core.logging_config import get_logger
from app.models import Deployment, DeploymentQueue, Site
from app.services.csv_import import RawCsvRow, blank_to_none, read_csv_rows
from app.services.folder_scanner import count_media_files

logger = get_logger(__name__)

DEPLOYMENT_REQUIRED_COLUMNS = ("folder",)
DEPLOYMENT_OPTIONAL_COLUMNS = ("site", "notes")

# Queue entries in these states own their folder. A completed or failed
# entry does not, so its folder can be queued again.
_ACTIVE_QUEUE_STATUSES = ("pending", "processing")

_MAX_NOTES = 1000

# Constant messages, so the dialog can group rows that share a mistake.
FOLDER_EMPTY = "Folder is empty. Enter the full path to the folder with the images or videos."
NOTES_TOO_LONG = "Notes are longer than 1000 characters. Use a shorter text."
FOLDER_NOT_ABSOLUTE = (
    "This is not a full path. Enter the full path, for example /Volumes/Data/CAM01 "
    "or D:\\Data\\CAM01."
)
FOLDER_NOT_FOUND = (
    "This folder was not found. Check the path, and make sure the drive is connected."
)
FOLDER_IS_A_FILE = (
    "This is a file, not a folder. Enter the folder that contains the images or videos."
)
FOLDER_DUPLICATED_IN_FILE = (
    "This folder is listed in more than one row. Each folder can only be added once. "
    "Remove one of the rows."
)
FOLDER_ALREADY_QUEUED = (
    "This folder is already in the queue. Remove this row, or remove the entry from "
    "the queue first."
)
FOLDER_ALREADY_DEPLOYED = (
    "This folder is already used by a deployment in this project. Remove this row."
)
FOLDER_HAS_NO_MEDIA = (
    "This folder contains no images or videos. Check the path, or remove this row."
)
FOLDER_UNREADABLE = (
    "This folder could not be read. The drive may have disconnected or be failing. "
    "Check the connection and import again."
)
FOLDER_CONTAINS_ANOTHER_ROW = (
    "This folder contains another folder listed in this file. The images inside "
    "it would be analysed twice. Remove one of the two rows."
)
FOLDER_INSIDE_ANOTHER_ROW = (
    "This folder is inside another folder listed in this file. Its images would "
    "be analysed twice. Remove one of the two rows."
)
SITE_NOT_FOUND = (
    "There is no site with this name in this project. Import your sites first, "
    "or correct the spelling."
)


def strip_surrounding_quotes(value: str) -> str:
    """Remove a quote character from either end of a path, independently.

    A path holding a space has to be quoted in a shell, so people quote it in
    their spreadsheet too. CSV's own quote character is the double quote,
    which csv.reader already consumes, but a single quote is an ordinary
    character to CSV and survives into the path, which is then never found.

    The two ends are handled separately rather than as a matching pair,
    because a spreadsheet usually destroys the pair: a **leading apostrophe
    is Excel's "treat this as text" marker**, so typing '/data/cam 1' into a
    cell stores /data/cam 1' and exports only the closing quote. Google
    Sheets does the same. Requiring a pair therefore fails on the single
    commonest way this reaches us.

    The trade this makes: a folder whose name really ends in a quote, say
    /data/Rangers', now imports as /data/Rangers and is reported as missing.
    Accepted, because a quote is an illegal filename character on Windows and
    such a name is vanishingly rare next to how often a spreadsheet does the
    above.

    The folder column only. Shell quoting is a habit about paths, and a site
    name that really contains quotes is likelier than a path that does.

    Deliberately does NOT un-escape backslashes, which is the other artifact
    (dragging a folder into a terminal gives `Kifaru\\ Plains`). A backslash
    is legal in a POSIX filename, so un-escaping could silently point the
    import at a different folder. Those rows keep failing as not found.
    """
    return value.lstrip("'\"").rstrip("'\"").strip()


def normalize_folder(value: str) -> str:
    """Collapse trailing slashes and doubled separators so two spellings of
    the same folder compare equal.

    Deliberately no `resolve()`: that follows symlinks and rewrites the path
    the user typed, and the queue is supposed to store what they gave it.
    """
    return str(Path(value))


def parse_deployment_csv(
    content: bytes,
) -> tuple[list[DeploymentImportRow], list[CsvImportProblem]]:
    """Read a deployment CSV into typed rows. No database and no disk access."""
    raw_rows, problems = read_csv_rows(
        content, DEPLOYMENT_REQUIRED_COLUMNS, DEPLOYMENT_OPTIONAL_COLUMNS, "deployment"
    )

    rows: list[DeploymentImportRow] = []
    for raw in raw_rows:
        row, row_problems = _parse_row(raw)
        if row is None:
            problems.extend(row_problems)
        else:
            rows.append(row)

    return rows, problems


def validate_deployment_rows(
    db: Session, project_id: str, rows: list[DeploymentImportRow]
) -> tuple[list[DeploymentImportRow], list[CsvImportProblem]]:
    """Check the rows against the project and the disk, and count the media.

    Returns the rows with their image and video counts filled in, plus every
    problem found. The folder checks run as a chain that stops at the first
    hit, so a path that does not exist never triggers a walk.
    """
    site_ids = _site_ids_by_name(db, project_id)
    queued = {
        normalize_folder(path)
        for path in db.execute(
            select(DeploymentQueue.folder_path)
            .where(DeploymentQueue.project_id == project_id)
            .where(DeploymentQueue.status.in_(_ACTIVE_QUEUE_STATUSES))
        ).scalars()
    }
    deployed = {
        normalize_folder(path)
        for path in db.execute(
            select(Deployment.folder_path)
            .where(Deployment.project_id == project_id)
            .where(Deployment.folder_path.is_not(None))
        ).scalars()
    }

    # Every row of a repeated folder is flagged, not just the later one, so
    # the user does not have to hunt for the row it collides with.
    repeated = {
        folder
        for folder, count in Counter(normalize_folder(row.folder) for row in rows).items()
        if count > 1
    }

    row_folders = [normalize_folder(row.folder) for row in rows]

    checked: list[DeploymentImportRow] = []
    problems: list[CsvImportProblem] = []

    for row in rows:
        folder = normalize_folder(row.folder)
        images, videos = 0, 0

        folder_problem = _check_folder(folder, repeated, queued, deployed, row_folders)
        if folder_problem is None:
            # An unreadable folder is a problem with this row, not with the
            # import: one flaky drive must not throw away a 30-row CSV the
            # user just filled in. It must also never fall through to
            # FOLDER_HAS_NO_MEDIA, which would send them looking for the
            # wrong thing.
            try:
                images, videos = count_media_files(Path(folder))
            except OSError as e:
                logger.warning(f"Could not read folder {folder}: {e}")
                folder_problem = (FOLDER_UNREADABLE, None)
            else:
                if images == 0 and videos == 0:
                    folder_problem = (FOLDER_HAS_NO_MEDIA, None)

        if folder_problem is not None:
            message, collides_with = folder_problem
            problems.append(
                CsvImportProblem(
                    row=row.row,
                    column="folder",
                    message=message,
                    # For an overlap the row number already identifies the
                    # user's own row; what they cannot work out is what it
                    # collides with, so that is what is shown.
                    value=collides_with or row.folder,
                )
            )

        if row.site is not None and row.site not in site_ids:
            problems.append(
                CsvImportProblem(
                    row=row.row, column="site", message=SITE_NOT_FOUND, value=row.site
                )
            )

        checked.append(
            row.model_copy(update={"image_count": images, "video_count": videos})
        )

    return checked, problems


def resolve_site_ids(
    db: Session, project_id: str, rows: list[DeploymentImportRow]
) -> dict[int, str | None]:
    """Site id per CSV row number, None for rows with no site.

    Only called on the write path. Validation has already proved every name
    exists, so a missing one here would be a race and the lookup returns None
    rather than inventing a site.
    """
    site_ids = _site_ids_by_name(db, project_id)
    return {row.row: site_ids.get(row.site) if row.site else None for row in rows}


def _check_folder(
    folder: str,
    repeated: set[str],
    queued: set[str],
    deployed: set[str],
    row_folders: list[str],
) -> tuple[str, str | None] | None:
    """The folder rules, in order, stopping at the first failure.

    Returns the message plus, when the problem is a collision with another
    folder, that other folder so the message can name it. None means the row
    is fine.

    Media counting is deliberately not here: it is the only check that walks
    the disk, so the caller runs it last and only when everything else passed.
    """
    path = Path(folder)
    if not path.is_absolute():
        return FOLDER_NOT_ABSOLUTE, None
    if not path.exists():
        return FOLDER_NOT_FOUND, None
    if not path.is_dir():
        return FOLDER_IS_A_FILE, None
    if folder in repeated:
        return FOLDER_DUPLICATED_IN_FILE, None
    if folder in queued:
        return FOLDER_ALREADY_QUEUED, None
    if folder in deployed:
        return FOLDER_ALREADY_DEPLOYED, None
    return _check_overlap(folder, row_folders)


def _check_overlap(folder: str, row_folders: list[str]) -> tuple[str, str] | None:
    """Whether this row's folder contains, or sits inside, another row's.

    Listing a season folder and its camera subfolders in one file would
    ingest the same images into two deployments, so it is blocked.

    Pure string comparison, no filesystem calls. Appending the separator is
    what keeps /x/cam1 and /x/cam10 apart, and it makes the check self-safe,
    since a path never starts with itself plus a separator.

    Only rows of this one file are compared. A folder that overlaps an
    existing deployment is deliberately allowed: see the scope note in
    TODO.md. Being lexical, this also does not see through symlinks, `..`
    segments or letter case.
    """
    prefix = folder.rstrip(os.sep) + os.sep
    for other in row_folders:
        if other.startswith(prefix):
            return FOLDER_CONTAINS_ANOTHER_ROW, other
        if folder.startswith(other.rstrip(os.sep) + os.sep):
            return FOLDER_INSIDE_ANOTHER_ROW, other
    return None


def _site_ids_by_name(db: Session, project_id: str) -> dict[str, str]:
    """Site names to ids for one project.

    Matching is exact, after the parser has stripped surrounding spaces. Site
    names are unique per project only case sensitively, so folding case here
    could match two different sites with no way to choose between them.
    """
    return {
        name: site_id
        for name, site_id in db.execute(
            select(Site.name, Site.id).where(Site.project_id == project_id)
        ).all()
    }


def _parse_row(raw: RawCsvRow) -> tuple[DeploymentImportRow | None, list[CsvImportProblem]]:
    """One CSV record into a typed row, or None plus everything wrong with it."""
    problems: list[CsvImportProblem] = []

    folder = strip_surrounding_quotes(raw.values["folder"])
    if not folder:
        problems.append(CsvImportProblem(row=raw.row, column="folder", message=FOLDER_EMPTY))

    notes = raw.values["notes"]
    if len(notes) > _MAX_NOTES:
        problems.append(CsvImportProblem(row=raw.row, column="notes", message=NOTES_TOO_LONG))

    if problems:
        return None, problems

    return (
        DeploymentImportRow(
            row=raw.row,
            folder=folder,
            site=blank_to_none(raw.values["site"]),
            notes=blank_to_none(notes),
        ),
        [],
    )
