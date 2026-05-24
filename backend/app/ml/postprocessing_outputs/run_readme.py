"""Human-readable run summary written into every folder-run output.

A `README.txt` at the root of the output directory carries the
complete picture of the run so a user (or a colleague) opening the
folder weeks later can see exactly what produced the deliverables:

- App and run metadata (version, run date, source folder)
- Model lineage (detection + classification, friendly name + id)
- All project settings (threshold, smoothing, rollup, geofence,
  video FPS, etc.)
- Results summary (by category, top species)
- Verification state
- Output manifest (what's in each subfolder / file)

Hardcoded as an always-on output: every save run writes one. The
file is plain text so any OS file manager renders it as a preview
and any text editor opens it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.manifest_manager import ManifestManager
from app.models import Deployment, Detection, File, Project

logger = get_logger(__name__)

README_FILENAME = "README.txt"


@dataclass
class RunReadmeResult:
    """Summary of the run-README write."""

    output_path: str = ""
    bytes_written: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "bytes_written": self.bytes_written,
            "errors": list(self.errors),
        }


def _model_label(model_id: str | None, mgr: ManifestManager) -> str:
    """Resolve a model id to "Friendly Name (model_id)" or just the
    id if the manifest doesn't know about it. Never raises."""
    if not model_id:
        return "(none)"
    try:
        m = mgr.get_model(model_id)
        return f"{m.friendly_name} ({m.model_id})"
    except Exception:
        return model_id


def _file_counts(db: Session, project_id: str) -> tuple[int, int, str | None, str | None]:
    """Return (image_count, video_count, earliest_capture, latest_capture).

    Capture dates serialise to ISO strings or None when there are no
    files yet. Both are naive wall-clock per the datetime convention.
    """
    image_count = (
        db.scalar(
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == project_id)
            .where(File.file_type == "image")
        )
        or 0
    )
    video_count = (
        db.scalar(
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == project_id)
            .where(File.file_type == "video")
        )
        or 0
    )
    earliest = db.scalar(
        select(func.min(File.captured_at_local))
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    )
    latest = db.scalar(
        select(func.max(File.captured_at_local))
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    )
    return (
        int(image_count),
        int(video_count),
        earliest.isoformat() if earliest else None,
        latest.isoformat() if latest else None,
    )


def _detection_counts_by_category(
    db: Session, project_id: str, threshold: float
) -> dict[str, int]:
    """Detections per category (animal / person / vehicle) passing
    the project threshold-with-verified rule."""
    from sqlalchemy import or_

    rows = db.execute(
        select(Detection.category, func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
        .group_by(Detection.category)
    ).all()
    return {category: int(count) for category, count in rows}


def _top_species(
    db: Session, project_id: str, threshold: float, limit: int = 20
) -> list[tuple[str, int]]:
    """Top-N species by detection count, threshold-aware."""
    from sqlalchemy import or_

    rows = db.execute(
        select(Detection.label, func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .where(Detection.label.is_not(None))
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
        .group_by(Detection.label)
        .order_by(func.count(Detection.id).desc())
        .limit(limit)
    ).all()
    return [(label, int(count)) for label, count in rows]


def _verification_stats(
    db: Session, project_id: str
) -> tuple[int, int]:
    """(verified_file_count, total_file_count) for the project."""
    total = (
        db.scalar(
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == project_id)
        )
        or 0
    )
    verified = (
        db.scalar(
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == project_id)
            .where(File.verified == True)  # noqa: E712
        )
        or 0
    )
    return int(verified), int(total)


def _section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}\n"


def _kv(key: str, value: object) -> str:
    return f"  {key:<32} {value}\n"


def _build_readme_text(
    *,
    project: Project,
    source_folder: str | None,
    run_started_at: datetime,
    file_counts: tuple[int, int, str | None, str | None],
    detection_counts: dict[str, int],
    top_species: list[tuple[str, int]],
    verification: tuple[int, int],
    manifest_mgr: ManifestManager,
) -> str:
    """Compose the README body. Returns a single newline-delimited
    string ready to write to disk."""
    image_count, video_count, earliest, latest = file_counts
    verified_files, total_files = verification

    lines: list[str] = []
    lines.append("=" * 72 + "\n")
    lines.append(
        f"AddaxAI folder analysis  -  {project.name}\n"
    )
    lines.append("=" * 72 + "\n")

    lines.append(_section("Run"))
    lines.append(_kv("AddaxAI version", APP_VERSION))
    lines.append(
        _kv("Run finished (UTC)", run_started_at.strftime("%Y-%m-%d %H:%M:%S"))
    )
    lines.append(_kv("Project id", project.id))
    lines.append(_kv("Source folder", source_folder or "(unknown)"))
    lines.append(
        _kv("Project timezone (metadata)", project.timezone)
    )

    lines.append(_section("Source media"))
    lines.append(_kv("Images", image_count))
    lines.append(_kv("Videos", video_count))
    lines.append(_kv("Earliest capture", earliest or "(no files)"))
    lines.append(_kv("Latest capture", latest or "(no files)"))

    lines.append(_section("Models"))
    lines.append(
        _kv("Detection model", _model_label(project.detection_model_id, manifest_mgr))
    )
    lines.append(
        _kv(
            "Classification model",
            _model_label(project.classification_model_id, manifest_mgr),
        )
    )
    lines.append(
        _kv(
            "Embedding model",
            _model_label(project.embedding_model_id, manifest_mgr),
        )
    )

    lines.append(_section("Detection settings"))
    lines.append(_kv("Detection threshold", project.detection_threshold))
    lines.append(
        _kv("Detection batch size", project.detection_batch_size or "(auto)")
    )
    lines.append(_kv("Country (geofence)", project.country_code or "(none)"))
    lines.append(_kv("State (geofence)", project.state_code or "(none)"))
    if project.excluded_classes:
        lines.append(
            _kv("Excluded classes", ", ".join(project.excluded_classes))
        )
    else:
        lines.append(_kv("Excluded classes", "(none)"))

    lines.append(_section("Classification & smoothing"))
    lines.append(
        _kv(
            "Classification batch size",
            project.classification_batch_size or "(auto)",
        )
    )
    lines.append(_kv("Event smoothing", project.event_smoothing))
    lines.append(_kv("Smoothing strength", project.smoothing_strength))
    lines.append(_kv("Taxonomic rollup", project.taxonomic_rollup))
    lines.append(
        _kv("Rollup threshold", project.taxonomic_rollup_threshold)
    )
    lines.append(
        _kv(
            "Independence interval (s)",
            project.independence_interval,
        )
    )

    lines.append(_section("Video"))
    lines.append(_kv("Sampling rate (fps)", project.video_fps))

    lines.append(_section("Results"))
    lines.append(_kv("Files total", total_files))
    if detection_counts:
        for cat in ("animal", "person", "vehicle"):
            if cat in detection_counts:
                lines.append(_kv(f"Detections ({cat})", detection_counts[cat]))
        # Surface any unexpected categories so they don't get silently lost.
        for cat in sorted(detection_counts):
            if cat in {"animal", "person", "vehicle"}:
                continue
            lines.append(_kv(f"Detections ({cat})", detection_counts[cat]))
    else:
        lines.append(_kv("Detections", 0))

    if total_files > 0:
        pct = (verified_files / total_files) * 100
        lines.append(
            _kv(
                "Files verified",
                f"{verified_files} / {total_files} "
                f"({pct:.1f}%)",
            )
        )
    else:
        lines.append(_kv("Files verified", "0 / 0"))

    if top_species:
        lines.append(_section("Top species (by detection count)"))
        for label, count in top_species:
            lines.append(f"  {label:<40} {count}\n")
    else:
        lines.append(_section("Top species"))
        lines.append("  (no species labels yet)\n")

    lines.append(_section("Output manifest"))
    lines.append(
        "  README.txt                          this file\n"
        "  <label>/...                         files grouped by species\n"
        "                                      (when separation was enabled)\n"
        "  <file>.jpg                          annotated copies with boxes\n"
        "                                      drawn and / or people blurred\n"
        "                                      (when separation was off)\n"
        "  observations.csv                    flat observation rows\n"
        "  observations.xlsx                   same rows, Excel format\n"
        "  recognition.json                    canonical recognition JSON\n"
        "\n"
        "  (subfolders / files only appear when the matching output\n"
        "  option was selected for this run; EXIF detection tags are\n"
        "  embedded silently on every image written)\n"
    )

    return "".join(lines)


def write_run_readme(
    db: Session,
    project_id: str,
    target_dir: Path,
) -> RunReadmeResult:
    """Write README.txt summarising the run at ``target_dir/README.txt``.

    Always-on output — every save run produces one. Errors during
    composition / write are recorded but never raise; the README is
    a nice-to-have, not load-bearing.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)

    source_folder: str | None = None
    deployment = db.execute(
        select(Deployment)
        .where(Deployment.project_id == project_id)
        .limit(1)
    ).scalar_one_or_none()
    if deployment is not None:
        source_folder = deployment.folder_path

    file_counts = _file_counts(db, project_id)
    detection_counts = _detection_counts_by_category(
        db, project_id, project.detection_threshold
    )
    top_species = _top_species(
        db, project_id, project.detection_threshold
    )
    verification = _verification_stats(db, project_id)

    settings = get_settings()
    manifest_mgr = ManifestManager(settings.user_data_dir / "models")

    text = _build_readme_text(
        project=project,
        source_folder=source_folder,
        run_started_at=datetime.now(UTC),
        file_counts=file_counts,
        detection_counts=detection_counts,
        top_species=top_species,
        verification=verification,
        manifest_mgr=manifest_mgr,
    )

    output_path = target_dir / README_FILENAME
    payload = text.encode("utf-8")
    with open(output_path, "wb") as f:
        f.write(payload)

    logger.info(
        f"run_readme: project={project_id} bytes={len(payload)} "
        f"path={output_path}"
    )

    return RunReadmeResult(
        output_path=str(output_path),
        bytes_written=len(payload),
    )
