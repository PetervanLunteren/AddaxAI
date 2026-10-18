"""Project packages: a project's rows, and optionally its media, in a
folder that another AddaxAI imports.

A package ("AddaxAI bundle" in the UI) is
``<project name>.addaxai-bundle/`` holding:

- ``manifest.json``: format version, app version, schema revision, the
  project's id and name, a sha256 of the data file, and one entry per
  deployment with the name of its media folder when media was included.
- ``project.db``: a SQLite file with the app's own schema at head,
  holding only this project's rows for the selected deployments.
- ``media/<folder name>/``: the deployment folders copied whole, hidden
  ``.addaxai`` included, so the package opens on a computer that does not
  have the photos. Optional.

**Identity.** Every id is kept. Projects, sites and deployments follow one
rule on import: the id exists here, it is the same thing; the id is
absent, it is created; creating it would clash on a name that must be
unique (project name, site name within the project), the import is
refused before anything is written. Existing projects and sites are never
modified. A deployment that exists is skipped or replaced whole, the
user's choice; there is no merge inside a deployment.

Labels are the one exception: every machine creates its own label rows
(builtin, model classes, rollup taxa, custom labels), so the same label
has a different id on each computer. A package label is matched to a
local row by id, else by model and name; what is left is created with
its taxonomy. The local row wins on any difference.

**Versions.** ``project.db`` carries its alembic revision. A package from
an older AddaxAI is upgraded on a temporary copy before import; a package
from a newer one is refused. Both files are plain SQLite, so the copy in
each direction is ``INSERT INTO ... SELECT`` over an attached database:
no serialisation, and SQLite keeps the types.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import __version__
from app.api.crud.deployment import (
    check_all_deployment_folders,
    purge_deployment_data,
)
from app.api.crud.export_formats import EXPORT_NAME_MAX_CHARS, unique_export_path
from app.api.schemas.package import (
    PackageDeploymentPreview,
    PackageExportResult,
    PackageImportResult,
    PackagePreview,
)
from app.core.config import Settings
from app.core.logging_config import get_logger
from app.db.backup import manual_snapshot
from app.db.base import Base, refresh_query_statistics
from app.db.migrations import (
    get_head_revision,
    revision_is_known,
    upgrade_to_head,
)
from app.ml.postprocessing_outputs.separate_folders import _unique_destination
from app.models import Deployment, Project, Site

logger = get_logger(__name__)

PACKAGE_SUFFIX = ".addaxai-bundle"
MANIFEST_NAME = "manifest.json"
DB_NAME = "project.db"
MEDIA_DIR = "media"
FORMAT_VERSION = 1

# `report(message, done, total)`. Called between steps and per copied
# file; the worker turns it into progress messages and raises
# `JobCancelledError` from inside it to stop an export.
ProgressFn = Callable[[str, int, int], None]


class PackageError(ValueError):
    """A package that cannot be read or imported. The message is for the
    user and says what to do."""


class ManifestDeployment(BaseModel):
    id: str
    # Subfolder of ``media/`` holding this deployment, when included.
    media_dir: str | None


class Manifest(BaseModel):
    format: int
    app_version: str
    alembic_revision: str
    project_id: str
    project_name: str
    project_db_sha256: str
    includes_media: bool
    deployments: list[ManifestDeployment]


# ---------------------------------------------------------------------------
# Which rows travel
# ---------------------------------------------------------------------------

# Subqueries over the *source* side of a copy. On export the source is
# the live database attached as ``src`` and the scope is the selected
# deployments; on import the source is the package attached as ``pkg``
# and the scope is the deployments being imported. Both keep the scope
# in a temp table so the statements stay the same shape.
_SCOPE = "(SELECT id FROM temp.scope)"


def _files_in(src: str) -> str:
    return f"(SELECT id FROM {src}.files WHERE deployment_id IN {_SCOPE})"


def _detections_in(src: str) -> str:
    return f"(SELECT id FROM {src}.detections WHERE file_id IN {_files_in(src)})"


def _events_in(src: str) -> str:
    return f"(SELECT id FROM {src}.events WHERE deployment_id IN {_SCOPE})"


def _deployment_tables(src: str) -> list[tuple[str, str]]:
    """The tables under a deployment, parents first, with the predicate
    that selects the rows in scope from `src`."""
    return [
        ("deployments", f"t.id IN {_SCOPE}"),
        ("files", f"t.deployment_id IN {_SCOPE}"),
        ("detections", f"t.file_id IN {_files_in(src)}"),
        ("detection_embeddings", f"t.detection_id IN {_detections_in(src)}"),
        ("events", f"t.deployment_id IN {_SCOPE}"),
        ("event_files", f"t.event_id IN {_events_in(src)}"),
        ("event_observations", f"t.event_id IN {_events_in(src)}"),
    ]


def _export_tables() -> list[tuple[str, str]]:
    """Every table the export copies, parents first."""
    src = "src"
    labels = (
        f"t.id IN (SELECT label_taxonomy_id FROM src.detections "
        f"WHERE file_id IN {_files_in(src)} UNION "
        f"SELECT label_taxonomy_id FROM src.event_observations "
        f"WHERE event_id IN {_events_in(src)})"
    )
    jobs = (
        f"t.id IN (SELECT job_id FROM src.detections "
        f"WHERE file_id IN {_files_in(src)} UNION "
        f"SELECT job_id FROM src.detection_embeddings "
        f"WHERE detection_id IN {_detections_in(src)})"
    )
    return [
        ("projects", "t.id = :project_id"),
        ("sites", "t.project_id = :project_id"),
        ("label_taxonomy", labels),
        ("jobs", jobs),
        *_deployment_tables(src),
    ]


# Columns that hold something machine-local and travel as NULL. The
# thumbnail is a file under the user data dir and is re-picked from the
# media; a job's payload and result name folders on the exporting
# computer. Carrying the job rows at all is what keeps `job_id IS NULL`
# meaning "drawn by a person" (see DEVELOPERS.md).
_NULLED: dict[str, frozenset[str]] = {
    "projects": frozenset({"thumbnail_path"}),
    "jobs": frozenset({"payload", "result", "error"}),
}

# Columns that point at a label row, remapped through ``temp.tax_map``.
_LABEL_COLUMNS: dict[str, str] = {
    "detections": "label_taxonomy_id",
    "event_observations": "label_taxonomy_id",
}


def _columns(table: str) -> list[str]:
    return [c.name for c in Base.metadata.tables[table].columns]


def _quoted(names: list[str]) -> str:
    return ", ".join(f'"{n}"' for n in names)


def _copy_sql(
    table: str,
    src: str,
    where: str,
    *,
    remap_labels: bool = False,
    or_ignore: bool = False,
) -> str:
    """``INSERT INTO main.<table> SELECT ... FROM <src>.<table> WHERE ...``
    with the machine-local columns nulled and, on import, the label
    pointer translated to the local row."""
    names = _columns(table)
    label_col = _LABEL_COLUMNS.get(table) if remap_labels else None
    exprs = []
    for n in names:
        if n in _NULLED.get(table, frozenset()):
            exprs.append("NULL")
        elif n == label_col:
            exprs.append(
                f'COALESCE((SELECT new_id FROM temp.tax_map WHERE old_id = t."{n}"), t."{n}")'
            )
        else:
            exprs.append(f't."{n}"')
    verb = "INSERT OR IGNORE INTO" if or_ignore else "INSERT INTO"
    return (
        f'{verb} main."{table}" ({_quoted(names)}) '
        f'SELECT {", ".join(exprs)} FROM {src}."{table}" t WHERE {where}'
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def package_folder_name(project_name: str) -> str:
    """``<name>.addaxai-bundle`` with what a filesystem refuses replaced
    and the name shortened as for every export (``EXPORT_NAME_MAX_CHARS``)."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", project_name)
    cleaned = cleaned[:EXPORT_NAME_MAX_CHARS].strip(" .")
    return f"{cleaned or 'project'}{PACKAGE_SUFFIX}"


def package_destination(
    output_dir: Path, project_name: str, deployment_folders: list[str]
) -> Path:
    """Where the package of `project_name` goes inside `output_dir`.

    A name that is taken gets ``_2`` and up, as every export does, so a
    lead can write one bundle per team member into one folder.

    Refuses a folder inside one of the deployments in the package: a copy
    with photos would walk into the package it is writing and copy itself
    until the disk is full, and even a data-only package there would be
    picked up as photos by the next scan of that folder. Both sides are
    resolved here only to compare them; nothing is stored.
    """
    out = Path(os.path.realpath(output_dir))
    for folder in deployment_folders:
        if out.is_relative_to(os.path.realpath(folder)):
            raise PackageError(
                f"Save the bundle outside the camera folders it contains. "
                f"Pick a folder that is not {folder} and not inside it."
            )
    return unique_export_path(output_dir, package_folder_name(project_name))


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _connect(
    path: Path, *, query: str = ""
) -> contextlib.closing[sqlite3.Connection]:
    """A raw connection with explicit transaction control, closed when the
    ``with`` block ends.

    Closed, not just committed: ``sqlite3``'s own ``with`` leaves the file
    open until the connection is garbage collected, and an exception's
    traceback keeps it alive through the cleanup. Windows cannot delete an
    open file, so a cancelled export could not remove its half-written
    bundle and the next export into that folder was refused.

    `isolation_level=None` because the module's implicit BEGIN would
    collide with the explicit ones below. `uri=True` so ATTACH accepts a
    ``file:`` URI with ``mode=ro``, which keeps the other database
    read-only whatever happens. `as_uri()` also percent-encodes
    characters SQLite would misread in a path.
    """
    return contextlib.closing(
        sqlite3.connect(
            f"{path.as_uri()}{query}", isolation_level=None, uri=True, timeout=30
        )
    )


def _attach_uri(path: Path, *, immutable: bool) -> str:
    uri = f"{path.as_uri()}?mode=ro"
    # A package is never written by anyone once made, so skipping the
    # locks lets it open from read-only media (a drive mounted read-only,
    # a share). The live database must not use it: it is being written.
    return f"{uri}&immutable=1" if immutable else uri


def export_package(
    live_db: Path,
    project_id: str,
    deployment_ids: list[str],
    output_dir: Path,
    *,
    include_media: bool,
    report: ProgressFn,
) -> PackageExportResult:
    """Write a package for `deployment_ids` of `project_id` into
    `output_dir`. Returns the summary the job stores."""
    with _connect(live_db, query="?mode=ro") as live:
        row = live.execute(
            "SELECT name, mode FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        marks = ",".join("?" * len(deployment_ids))
        folders = [
            f for (f,) in live.execute(
                f"SELECT folder_path FROM deployments WHERE id IN ({marks}) "
                "AND folder_path IS NOT NULL",
                deployment_ids,
            )
        ]
    if row is None:
        raise PackageError("Project not found")
    name, mode = row
    if mode != "research":
        raise PackageError("Only projects can be exported as a bundle")

    package_dir = package_destination(output_dir, name, folders)
    package_dir.mkdir(parents=True)
    try:
        return _write_package(
            live_db, project_id, deployment_ids, package_dir, name,
            include_media=include_media, report=report,
        )
    except BaseException:
        # A package without a manifest is not a package, and a cancelled
        # media copy can be most of a drive. Leave nothing behind.
        shutil.rmtree(package_dir, ignore_errors=True)
        raise


def _write_package(
    live_db: Path,
    project_id: str,
    deployment_ids: list[str],
    package_dir: Path,
    name: str,
    *,
    include_media: bool,
    report: ProgressFn,
) -> PackageExportResult:
    db_path = package_dir / DB_NAME
    tables = _export_tables()
    total_steps = len(tables) + 1
    report("Preparing bundle", 0, total_steps)
    # The migration chain builds every table plus `alembic_version` at
    # head, which is what the import checks. `create_all` would stamp
    # nothing.
    upgrade_to_head(f"sqlite:///{db_path}")

    with _connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "ATTACH DATABASE ? AS src", (_attach_uri(live_db, immutable=False),)
        )
        try:
            # One transaction, so every table is read from the same
            # snapshot while the app keeps writing.
            conn.execute("BEGIN")
            conn.execute("CREATE TEMP TABLE scope (id TEXT PRIMARY KEY)")
            conn.executemany(
                "INSERT INTO temp.scope (id) VALUES (?)",
                [(d,) for d in deployment_ids],
            )
            for i, (table, where) in enumerate(tables, start=1):
                report(f"Copying {table}", i, total_steps)
                conn.execute(_copy_sql(table, "src", where), {"project_id": project_id})
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("DROP TABLE IF EXISTS temp.scope")
            conn.execute("DETACH DATABASE src")

        deployments = conn.execute(
            "SELECT id, folder_path FROM deployments ORDER BY folder_path"
        ).fetchall()
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        # Alembic's engine put the file in WAL mode, which a read-only
        # medium cannot open. Back to a single plain file, compacted.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")

    manifest_deployments = [
        ManifestDeployment(id=dep_id, media_dir=None) for dep_id, _ in deployments
    ]
    if include_media:
        _copy_media(package_dir, manifest_deployments, dict(deployments), report)

    manifest = Manifest(
        format=FORMAT_VERSION,
        app_version=__version__,
        alembic_revision=get_head_revision(),
        project_id=project_id,
        project_name=name,
        project_db_sha256=_sha256(db_path),
        includes_media=include_media,
        deployments=manifest_deployments,
    )
    # Last, so a package without one is a package that was not finished.
    (package_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))
    logger.info(
        f"Wrote package {package_dir} ({len(deployments)} deployments, "
        f"{files} files, media={'yes' if include_media else 'no'})"
    )
    return PackageExportResult(
        package_dir=str(package_dir),
        deployments=len(deployments),
        files=files,
        includes_media=include_media,
    )


def _copy_media(
    package_dir: Path,
    deployments: list[ManifestDeployment],
    folders: dict[str, str | None],
    report: ProgressFn,
) -> None:
    """Copy every deployment folder into ``media/``, one subfolder each.

    A folder that is not there fails the export: a package missing a
    camera's photos would be reported as complete, and the user cannot
    tell afterwards.
    """
    media_root = package_dir / MEDIA_DIR
    media_root.mkdir()
    sources: list[tuple[ManifestDeployment, Path]] = []
    for dep in deployments:
        folder = folders.get(dep.id)
        if not folder:
            continue
        src = Path(folder)
        if not src.is_dir():
            raise PackageError(f"Folder not found: {folder}")
        sources.append((dep, src))

    total = sum(len(files) for _dep, src in sources for _, _, files in os.walk(src))
    done = 0

    def copy_and_report(src_file: str, dst_file: str) -> None:
        # `copytree` lets anything but an OSError out at once, so a cancel
        # raised by `report` stops the copy at this file.
        nonlocal done
        shutil.copy2(src_file, dst_file)
        done += 1
        report("Copying photos and videos", done, total)

    reserved: set[Path] = set()
    for dep, src in sources:
        dest, _renamed = _unique_destination(media_root, src.name, reserved)
        reserved.add(dest)
        dep.media_dir = dest.name
        shutil.copytree(src, dest, copy_function=copy_and_report)


# ---------------------------------------------------------------------------
# Import: reading a package
# ---------------------------------------------------------------------------


def read_manifest(package_dir: Path) -> Manifest:
    if not package_dir.is_dir():
        raise PackageError("That is not a folder")
    manifest_path = package_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PackageError(
            "This folder is not an AddaxAI bundle: it has no manifest.json. "
            "Pick the folder whose name ends in .addaxai-bundle."
        )
    try:
        manifest = Manifest.model_validate_json(manifest_path.read_text())
    except (ValidationError, ValueError) as e:
        logger.warning(f"Unreadable package manifest {manifest_path}: {e}")
        raise PackageError(
            "The bundle is damaged: its manifest.json cannot be read. "
            "Export the bundle again."
        ) from e
    if manifest.format != FORMAT_VERSION:
        raise PackageError(
            "This bundle was made by a newer AddaxAI. Update AddaxAI and try again."
        )
    if not (package_dir / DB_NAME).is_file():
        raise PackageError("The bundle is incomplete: project.db is missing.")
    return manifest


RevisionStatus = Literal["current", "older", "newer"]


def _revision_status(revision: str) -> RevisionStatus:
    if revision == get_head_revision():
        return "current"
    return "older" if revision_is_known(revision) else "newer"


def _package_revision(db_path: Path) -> str:
    with _connect(db_path, query="?mode=ro&immutable=1") as conn:
        rows = conn.execute("SELECT version_num FROM alembic_version").fetchall()
    if len(rows) != 1:
        raise PackageError(_DAMAGED_DB)
    return rows[0][0]


_DAMAGED_DB = (
    "The bundle is damaged: its project.db cannot be read. "
    "Export the bundle again."
)


def inspect_package(db: Session, package_dir: Path) -> PackagePreview:
    """What importing `package_dir` would do. Reads only; the hash is
    checked by the import itself, because hashing a large file over a
    slow drive does not belong in a request."""
    try:
        return _inspect(db, package_dir)
    except sqlite3.DatabaseError as e:
        # A truncated or overwritten file is not a SQLite database, or one
        # without our tables. Either way the copy is the problem.
        logger.warning(f"Unreadable package data file in {package_dir}: {e}")
        raise PackageError(_DAMAGED_DB) from e


def _inspect(db: Session, package_dir: Path) -> PackagePreview:
    manifest = read_manifest(package_dir)
    db_path = package_dir / DB_NAME
    revision = _package_revision(db_path)
    status = _revision_status(revision)
    problems: list[str] = []
    warnings: list[str] = []
    if status == "newer":
        problems.append(
            f"This bundle was made by AddaxAI {manifest.app_version}, which is "
            "newer than this one. Update AddaxAI and try again."
        )
    elif status == "older":
        warnings.append(
            f"Made by AddaxAI {manifest.app_version}. It will be brought up to "
            "date as part of the import."
        )

    with _connect(db_path, query="?mode=ro&immutable=1") as pkg:
        sites = pkg.execute("SELECT id, name FROM sites").fetchall()
        file_counts = dict(
            pkg.execute(
                "SELECT deployment_id, COUNT(*) FROM files GROUP BY deployment_id"
            ).fetchall()
        )

    local_project = db.get(Project, manifest.project_id)
    if local_project is None:
        same_name = db.execute(
            select(Project.id).where(Project.name == manifest.project_name)
        ).scalar_one_or_none()
        if same_name is not None:
            problems.append(
                f'A different project named "{manifest.project_name}" already '
                "exists on this computer. Rename it and import again."
            )

    local_site_ids = set(
        db.execute(
            select(Site.id).where(Site.project_id == manifest.project_id)
        ).scalars()
    )
    local_site_names = set(
        db.execute(
            select(Site.name).where(Site.project_id == manifest.project_id)
        ).scalars()
    )
    for site_id, site_name in sites:
        if site_id not in local_site_ids and site_name in local_site_names:
            problems.append(
                f'A different site named "{site_name}" already exists in this '
                "project. Rename it and import again."
            )

    existing_ids = set(
        db.execute(
            select(Deployment.id).where(
                Deployment.id.in_([d.id for d in manifest.deployments])
            )
        ).scalars()
    )
    previews = []
    for dep in manifest.deployments:
        if dep.media_dir and not (package_dir / MEDIA_DIR / dep.media_dir).is_dir():
            warnings.append(
                f"The photos of {dep.media_dir} are missing from the bundle; "
                "that deployment will need reconnecting."
            )
        previews.append(
            PackageDeploymentPreview(
                id=dep.id,
                files=int(file_counts.get(dep.id, 0)),
                exists=dep.id in existing_ids,
            )
        )

    return PackagePreview(
        project_id=manifest.project_id,
        project_name=manifest.project_name,
        project_exists=local_project is not None,
        app_version=manifest.app_version,
        revision_status=status,
        includes_media=manifest.includes_media,
        deployments=previews,
        problems=problems,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Import: writing
# ---------------------------------------------------------------------------

# Rewrites the part of a stored path below the old deployment folder onto
# the new one. The prefix test keeps rows that never sat under the folder
# untouched (`relink_deployment` tolerates those too), the `ltrim` covers
# a folder stored with a trailing separator (a drive root), and the
# `replace` turns the other operating system's separators into this
# one's, so a package made on Windows reads on macOS and the reverse.
_REWRITE_SQL = (
    'UPDATE files SET "{col}" = :new || :sep || '
    'replace(ltrim(substr("{col}", length(:old) + 1), \'/\\\'), :other, :sep) '
    'WHERE deployment_id = :dep AND substr("{col}", 1, length(:old)) = :old'
)


def _rewrite_paths(
    raw: sqlite3.Connection, deployment_id: str, old_folder: str, new_folder: str
) -> None:
    sep = os.sep
    other = "/" if sep == "\\" else "\\"
    params = {
        "dep": deployment_id,
        "old": old_folder,
        "new": new_folder.rstrip("/\\"),
        "sep": sep,
        "other": other,
    }
    for col in ("file_path", "best_frame_path"):
        raw.execute(_REWRITE_SQL.format(col=col), params)
    raw.execute(
        "UPDATE deployments SET folder_path = :new, folder_status = 'needs_relink', "
        "last_validated_at_utc = NULL WHERE id = :dep",
        params,
    )


def _upgraded_copy(
    db_path: Path, status: RevisionStatus
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """The package data file to read: itself, or an upgraded copy in a
    temp dir when the package is older than this app."""
    if status != "older":
        return db_path, None
    tmp = tempfile.TemporaryDirectory(prefix="addaxai-package-")
    copy = Path(tmp.name) / DB_NAME
    shutil.copyfile(db_path, copy)
    upgrade_to_head(f"sqlite:///{copy}")
    with _connect(copy) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
    return copy, tmp


def import_package(
    db: Session,
    settings: Settings,
    package_dir: Path,
    *,
    replace_existing: bool,
    report: ProgressFn,
) -> PackageImportResult:
    """Import `package_dir` into the live database.

    One write transaction on the session's own connection: the attached
    package, the leaf-first purge of replaced deployments
    (`purge_deployment_data`, through the session) and the inserts all
    share it, so either everything lands or nothing does. The session
    must not have a transaction open when this is called.
    """
    preview = inspect_package(db, package_dir)
    if preview.problems:
        raise PackageError(" ".join(preview.problems))
    manifest = read_manifest(package_dir)
    db_path = package_dir / DB_NAME
    tables = _deployment_tables("pkg")
    total_steps = len(tables) + 4

    report("Checking the bundle", 0, total_steps)
    if _sha256(db_path) != manifest.project_db_sha256:
        raise PackageError(
            "The bundle's data file does not match its manifest. The copy "
            "may be damaged; export the bundle again."
        )
    source_db, tmp = _upgraded_copy(db_path, preview.revision_status)

    snapshot = manual_snapshot(settings, note="before-import")
    logger.info(f"Backed up before import: {snapshot.name}")

    existing = {d.id for d in preview.deployments if d.exists}
    to_import = [
        d.id
        for d in preview.deployments
        if d.id not in existing or replace_existing
    ]
    replaced = [d for d in to_import if d in existing]
    media_dirs = {
        d.id: d.media_dir for d in manifest.deployments if d.media_dir
    }

    try:
        db.commit()
        raw = db.connection().connection.dbapi_connection
        assert isinstance(raw, sqlite3.Connection)
        raw.execute("ATTACH DATABASE ? AS pkg", (_attach_uri(source_db, immutable=True),))
        try:
            raw.execute("BEGIN IMMEDIATE")
            raw.execute("CREATE TEMP TABLE scope (id TEXT PRIMARY KEY)")
            raw.executemany(
                "INSERT INTO temp.scope (id) VALUES (?)", [(d,) for d in to_import]
            )

            report("Creating project, sites and labels", 1, total_steps)
            project_created = (
                raw.execute(
                    _copy_sql(
                        "projects",
                        "pkg",
                        "t.id = :project_id AND NOT EXISTS "
                        "(SELECT 1 FROM main.projects WHERE id = :project_id)",
                    ),
                    {"project_id": manifest.project_id},
                ).rowcount
                == 1
            )
            # NOT EXISTS rather than OR IGNORE: a name clash the preview
            # missed must fail loudly, not vanish into an ignored insert
            # that leaves a deployment pointing at a site that is not there.
            raw.execute(
                _copy_sql(
                    "sites",
                    "pkg",
                    "NOT EXISTS (SELECT 1 FROM main.sites s WHERE s.id = t.id)",
                )
            )

            # Labels: same id, else same model and name (MIN(id) in case
            # a machine holds duplicates; SQLite's UNIQUE ignores the NULL
            # project_id), else created with the package's id.
            raw.execute("CREATE TEMP TABLE tax_map (old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL)")
            raw.execute(
                "INSERT INTO temp.tax_map (old_id, new_id) "
                "SELECT p.id, COALESCE("
                "  (SELECT l.id FROM main.label_taxonomy l WHERE l.id = p.id),"
                "  (SELECT MIN(l.id) FROM main.label_taxonomy l "
                "   WHERE l.classification_model_id = p.classification_model_id "
                "   AND lower(l.name) = lower(p.name))"
                ") FROM pkg.label_taxonomy p "
                "WHERE EXISTS (SELECT 1 FROM main.label_taxonomy l WHERE l.id = p.id "
                "  OR (l.classification_model_id = p.classification_model_id "
                "      AND lower(l.name) = lower(p.name)))"
            )
            raw.execute(
                _copy_sql(
                    "label_taxonomy",
                    "pkg",
                    "t.id NOT IN (SELECT old_id FROM temp.tax_map)",
                )
            )
            raw.execute(_copy_sql("jobs", "pkg", "1=1", or_ignore=True))

            report("Removing replaced deployments", 2, total_steps)
            kept_folders: dict[str, str | None] = {}
            if replaced:
                kept_folders = dict(
                    db.execute(
                        select(Deployment.id, Deployment.folder_path).where(
                            Deployment.id.in_(replaced)
                        )
                    ).all()
                )
                ids = select(Deployment.id).where(Deployment.id.in_(replaced))
                purge_deployment_data(db, ids)
                db.execute(
                    delete(Deployment).where(Deployment.id.in_(replaced)),
                    execution_options={"synchronize_session": False},
                )

            for i, (table, where) in enumerate(tables, start=3):
                report(f"Copying {table}", i, total_steps)
                raw.execute(_copy_sql(table, "pkg", where, remap_labels=True))

            report("Connecting folders", total_steps - 1, total_steps)
            rows = raw.execute(
                f"SELECT id, folder_path FROM pkg.deployments WHERE id IN {_SCOPE} "
                "AND folder_path IS NOT NULL"
            ).fetchall()
            # One order: the photos inside this package, else the folder
            # this computer already used for the deployment (a replace),
            # else the path the package recorded.
            for dep_id, old_folder in rows:
                media = package_dir / MEDIA_DIR / media_dirs.get(dep_id, "")
                if dep_id in media_dirs and media.is_dir():
                    new_folder = str(media)
                elif kept_folders.get(dep_id):
                    new_folder = kept_folders[dep_id]
                else:
                    new_folder = old_folder
                _rewrite_paths(raw, dep_id, old_folder, new_folder)

            raw.execute("COMMIT")
        except BaseException:
            if raw.in_transaction:
                raw.execute("ROLLBACK")
            raise
        finally:
            # The connection goes back to the pool, so leave nothing behind.
            raw.execute("DROP TABLE IF EXISTS temp.scope")
            raw.execute("DROP TABLE IF EXISTS temp.tax_map")
            raw.execute("DETACH DATABASE pkg")
            db.rollback()
    finally:
        if tmp is not None:
            tmp.cleanup()

    report("Checking folders", total_steps, total_steps)
    check_all_deployment_folders(db, manifest.project_id)
    refresh_query_statistics(db)
    db.commit()

    result = PackageImportResult(
        project_id=manifest.project_id,
        project_created=project_created,
        deployments_created=len(to_import) - len(replaced),
        deployments_replaced=len(replaced),
        deployments_skipped=len(existing) - len(replaced),
    )
    logger.info(f"Imported package {package_dir}: {result.model_dump()}")
    return result

