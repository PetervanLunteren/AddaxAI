"""
Thumbnail service for project card images.

Generates resized JPEG thumbnails from source images and auto-selects
a random high-quality camera trap image for projects that lack a
user-uploaded image.
"""

import random
from pathlib import Path

from PIL import Image
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, File, Project

logger = get_logger(__name__)

_MAX_WIDTH = 512
_JPEG_QUALITY = 95


def generate_thumbnail(source_path: Path, dest_path: Path) -> Path:
    """Resize an image to max 512px wide and save as JPEG.

    Maintains aspect ratio. Converts to RGB for JPEG compatibility.
    Creates parent directories if needed.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as img:
        img = img.convert("RGB")

        if img.width > _MAX_WIDTH:
            ratio = _MAX_WIDTH / img.width
            new_height = int(img.height * ratio)
            img = img.resize((_MAX_WIDTH, new_height), Image.LANCZOS)

        img.save(dest_path, format="JPEG", quality=_JPEG_QUALITY)

    return dest_path


def _resolve_image_path(file: File) -> Path | None:
    """Get the displayable image path for a file.

    For videos, returns the best frame JPEG. For images, returns
    the file itself. Returns None if the path does not exist on disk.
    """
    if file.file_type == "video" and file.best_frame_path:
        p = Path(file.best_frame_path)
    else:
        p = Path(file.file_path)

    return p if p.exists() else None


def auto_select_project_thumbnails(
    db: Session,
    thumbnails_dir: Path,
) -> None:
    """Generate thumbnails for projects eligible for auto-selection.

    Picks a random image from the top 10% highest-scoring detections.
    Re-rolls auto-generated thumbnails on every startup. Skips
    user-uploaded images (stored in project-images/).
    """
    thumbnails_prefix = str(thumbnails_dir)
    projects = (
        db.query(Project)
        .filter(
            (Project.thumbnail_path.is_(None))
            | (Project.thumbnail_path.like(f"{thumbnails_prefix}%"))
        )
        .all()
    )

    if not projects:
        logger.info("No projects need thumbnail generation")
        return

    for project in projects:
        try:
            _auto_select_for_project(db, project, thumbnails_dir)
        except Exception:
            logger.warning(
                f"Failed to auto-select thumbnail for project "
                f"{project.name}",
                exc_info=True,
            )


def _auto_select_for_project(
    db: Session,
    project: Project,
    thumbnails_dir: Path,
) -> None:
    """Pick a random image from the top 10% detections."""
    # Get the 10 highest-confidence animal detections
    candidates = (
        db.query(Detection, File)
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .filter(
            Deployment.project_id == project.id,
            Detection.category == "animal",
        )
        .order_by(desc(Detection.confidence))
        .limit(10)
        .all()
    )

    if not candidates:
        return

    _detection, file = random.choice(candidates)
    source = _resolve_image_path(file)
    if source is None:
        logger.warning(
            f"Image file missing for project {project.name}: "
            f"{file.file_path}"
        )
        return

    dest = thumbnails_dir / f"{project.id}.jpg"
    generate_thumbnail(source, dest)

    project.thumbnail_path = str(dest)
    db.commit()

    logger.info(
        f"Auto-selected thumbnail for project {project.name}"
    )
