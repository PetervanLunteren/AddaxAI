"""Human-readable run summary written into every folder-run output.

An `addaxai-run-info.txt` at the root of the output directory carries
the complete picture of the run so a user (or a colleague) opening the
folder weeks later can see exactly what produced the deliverables:

- App and run metadata (version, run date, source folder)
- Model lineage (detection + classification, friendly name + id)
- All project settings (threshold, smoothing, rollup, geofence,
  video FPS, etc.)
- Results summary (by category, top species)
- Verification state

Written when the Save step's "Run details" checkbox is on (the
default). The file is plain text so any OS file manager renders it as
a preview and any text editor opens it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.confidence import ROLLUP_THRESHOLD, format_confidence_pct
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.manifest_manager import ManifestManager
from app.models import Deployment, Detection, File, Project

logger = get_logger(__name__)

SUMMARY_FILENAME = "addaxai-run-info.txt"

# Human wording for Project.media_filter. Only the non-default values are ever
# printed, so "all" is here for completeness rather than use.
MEDIA_FILTER_LABELS = {
    "all": "images and videos",
    "images": "only images",
    "videos": "only videos",
}


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


def _geofence_summary(project: Project, models_dir: Path) -> str:
    """One-line species inclusion summary for the geofence, e.g.
    ``288 included, 1,712 excluded``. Avoids dumping the full exclusion
    list (which can be ~40 KB for SpeciesNet). Falls back to just the
    excluded count if the model's full label list can't be read."""
    excluded = len(project.excluded_classes or [])
    if excluded == 0:
        return "all included"
    try:
        from app.ml.geofence import get_all_labels

        model_dir = models_dir / "cls" / (project.classification_model_id or "")
        total = len(get_all_labels(model_dir))
        included = max(total - excluded, 0)
        return f"{included:,} included, {excluded:,} excluded"
    except Exception:
        return f"{excluded:,} excluded"


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
    geofence_summary: str,
    manifest_mgr: ManifestManager,
    media_threshold: float,
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
    # Only when it excluded something. The counts above come from the
    # database, so a filtered-out kind reads as 0 — identical to a folder
    # that never held any. This line is what tells those two apart; without
    # it the file quietly implies the folder had no videos.
    if project.media_filter != "all":
        lines.append(_kv("Media filter", MEDIA_FILTER_LABELS[project.media_filter]))
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
    # Media copies (separated folders, drawn boxes, blurs) only show
    # detections at or above this confidence. The data exports (CSV,
    # XLSX, recognition JSON) always contain every detection.
    lines.append(_kv("Media output threshold", format_confidence_pct(media_threshold)))
    lines.append(
        _kv("Data exports", "complete, no confidence filter")
    )
    lines.append(
        _kv("Classification gate", format_confidence_pct(project.classification_gate))
    )
    lines.append(
        _kv("Detection batch size", project.detection_batch_size or "(auto)")
    )
    lines.append(_kv("Country (geofence)", project.country_code or "(none)"))
    lines.append(_kv("State (geofence)", project.state_code or "(none)"))
    lines.append(_kv("Species (geofence)", geofence_summary))

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
        _kv("Rollup threshold", format_confidence_pct(ROLLUP_THRESHOLD))
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

    return "".join(lines)


def write_run_readme(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    media_threshold: float,
) -> RunReadmeResult:
    """Write the run info at ``target_dir/addaxai-run-info.txt``.

    ``media_threshold`` is the Save step's media-output confidence,
    reported so a reader knows which detections the media copies show.
    The data exports are always complete and say so in the summary.

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
        db, project_id, project.counting_threshold
    )
    top_species = _top_species(
        db, project_id, project.counting_threshold
    )
    verification = _verification_stats(db, project_id)

    settings = get_settings()
    models_dir = settings.user_data_dir / "models"
    manifest_mgr = ManifestManager(models_dir)
    geofence_summary = _geofence_summary(project, models_dir)

    text = _build_readme_text(
        project=project,
        source_folder=source_folder,
        run_started_at=datetime.now(UTC),
        file_counts=file_counts,
        detection_counts=detection_counts,
        top_species=top_species,
        verification=verification,
        geofence_summary=geofence_summary,
        manifest_mgr=manifest_mgr,
        media_threshold=media_threshold,
    )

    output_path = target_dir / SUMMARY_FILENAME
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
