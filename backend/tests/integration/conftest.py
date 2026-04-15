"""
Shared fixtures for ML pipeline integration tests.

Provides:
- Tiny JPEG creation (valid 1x1 pixel images via PIL)
- JSON builder for MegaDetector-format results
- Video frame directory creator
- deployment_scaffold fixture with full DB + filesystem setup
"""

import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from tests.conftest import make_deployment, make_job, make_project, make_site

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_tiny_jpeg(path: Path) -> Path:
    """Write a valid 1x1 pixel JPEG to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = PILImage.new("RGB", (1, 1), color=(128, 128, 128))
    img.save(path, format="JPEG")
    return path


def write_json(path: Path, data: dict) -> Path:
    """Write *data* as JSON to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


_DEFAULT_EXIF_DATETIME = "2024:06:15 12:00:00"


def build_detection_json(
    images: list[dict],
    classification_categories: dict[str, str] | None = None,
    classification_category_descriptions: dict[str, str] | None = None,
) -> dict:
    """
    Build a MegaDetector-format JSON dict.

    Each element of *images* should have keys: file, detections,
    and optionally exif_metadata, width, height, best_frame_number,
    frame_rate, file_id.

    Phase 6 now requires every file to have an extractable timestamp
    (no silent fallbacks — see DEVELOPERS.md "Datetime conventions").
    For tests that don't explicitly set exif_metadata, default a
    plausible DateTimeOriginal so the loader has a timestamp to work
    with. Tests that care about timestamps set their own.
    """
    result = {
        "images": [_with_default_exif(img) for img in images],
        "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
        "info": {"detection_completion_time": "2026-01-01 00:00:00"},
    }
    if classification_categories is not None:
        result["classification_categories"] = classification_categories
    if classification_category_descriptions is not None:
        result["classification_category_descriptions"] = classification_category_descriptions
    return result


def _with_default_exif(img: dict) -> dict:
    """
    Add a default EXIF DateTimeOriginal if the test didn't set one at all.

    If the test explicitly sets `exif_metadata` to any dict (even `{}`),
    respect it verbatim: that's how the MissingTimestampError failure-path
    tests simulate "no extractable timestamp".
    """
    if "exif_metadata" in img:
        return img
    return {
        **img,
        "exif_metadata": {"DateTimeOriginal": _DEFAULT_EXIF_DATETIME},
    }


def create_video_frames(
    artifacts_folder: Path,
    video_relative_path: str,
    frame_numbers: list[int],
) -> list[Path]:
    """
    Create tiny frame JPEGs in video_frames/{video_dir}/frameNNNNNN.jpg.

    Returns list of created frame paths.
    """
    frames_dir = artifacts_folder / "video_frames" / video_relative_path
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in frame_numbers:
        p = frames_dir / f"frame{n:06d}.jpg"
        create_tiny_jpeg(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main scaffold fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def deployment_scaffold(db, tmp_path):
    """
    Build a full deployment scaffold on disk and in the DB.

    Creates:
    - 3 tiny JPEGs in subdir/
    - 1 dummy .mp4 in videos/
    - Project → Site → Deployment → Job DB records
    - Artifacts folder at deployment/.addaxai/projects/{project_id}/

    Returns a dict with all references.
    """
    # Filesystem ----------------------------------------------------------
    deploy_dir = tmp_path / "deployment"
    deploy_dir.mkdir()

    img_dir = deploy_dir / "subdir"
    img_dir.mkdir()
    img_paths = []
    for name in ["img_001.jpg", "img_002.jpg", "img_003.jpg"]:
        img_paths.append(create_tiny_jpeg(img_dir / name))

    vid_dir = deploy_dir / "videos"
    vid_dir.mkdir()
    vid_path = vid_dir / "clip.mp4"
    # Write a tiny valid-enough file (not a real MP4, but exists on disk)
    vid_path.write_bytes(b"\x00" * 64)

    # DB records ----------------------------------------------------------
    project = make_project(db, excluded_classes=[])
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(
        db,
        site_id=site.id,
        folder_path=str(deploy_dir),
    )
    job = make_job(db)

    # Artifacts folder ----------------------------------------------------
    artifacts = deploy_dir / ".addaxai" / "projects" / project.id
    artifacts.mkdir(parents=True)

    return {
        "tmp_path": tmp_path,
        "deploy_dir": deploy_dir,
        "img_paths": img_paths,
        "vid_path": vid_path,
        "project": project,
        "site": site,
        "deployment": deployment,
        "job": job,
        "artifacts": artifacts,
        "db": db,
    }
