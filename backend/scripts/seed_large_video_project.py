"""
Seed a throwaway AddaxAI data folder with one large synthetic video
project, to test exports (and anything else that has to cope with a big
database) at a real user's scale.

The default shape is the road-tunnel project that ran the export out of
memory on 2026-10-14: 6,000 videos, each with boxes on 85 sampled frames
and 5 boxes per frame, so 425 boxes per video and 2.55 million in all.
Only the saved frame of each video is what the tables show.

The database is built by the app's own ``init_db``, so the backend accepts
it. Nothing is written outside the folder you name, and it must not be
your real ``~/AddaxAI``:

    ADDAXAI_USER_DATA_DIR=/tmp/addaxai-large \\
        backend/venv/bin/python backend/scripts/seed_large_video_project.py

Then start a backend on that folder, on its own port, and export:

    ADDAXAI_USER_DATA_DIR=/tmp/addaxai-large ADDAXAI_DISABLE_MODEL_UPDATES=1 \\
        backend/venv/bin/python -m uvicorn app.main:app --port 8123

Takes a minute or two and about 1.5 GB of disk at the default size.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_BACKEND))

INSERT_CHUNK = 50_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", type=int, default=6_000)
    parser.add_argument("--frames", type=int, default=85, help="sampled frames per video")
    parser.add_argument("--boxes", type=int, default=5, help="boxes per frame")
    args = parser.parse_args()

    data_dir = os.environ.get("ADDAXAI_USER_DATA_DIR")
    if not data_dir:
        print("Set ADDAXAI_USER_DATA_DIR to a throwaway folder first.")
        return 1
    if Path(data_dir).resolve() == (Path.home() / "AddaxAI").resolve():
        print("Refusing to seed the real ~/AddaxAI.")
        return 1
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    from sqlalchemy import insert

    from app.db.base import get_session_factory, init_db
    from app.models import Deployment, Detection, File, Project, Site

    init_db()
    db = get_session_factory()()

    project_id = str(uuid.uuid4())
    site_id = str(uuid.uuid4())
    deployment_id = str(uuid.uuid4())
    db.add(
        Project(
            id=project_id,
            name="Large video project",
            timezone="Europe/Zurich",
            detection_model_id="MD5A-0-0",
            counting_threshold=0.2,
        )
    )
    db.add(
        Site(
            id=site_id, project_id=project_id, name="tunnel", latitude=47.3, longitude=8.5
        )
    )
    db.add(
        Deployment(
            id=deployment_id,
            project_id=project_id,
            site_id=site_id,
            folder_path="/data/tunnel-march",
            start_date_local=date(2026, 3, 1),
            end_date_local=date(2026, 3, 31),
        )
    )
    db.commit()

    start = datetime(2026, 3, 1)
    best_frame = args.frames // 2
    files: list[dict] = []
    boxes: list[dict] = []
    total_boxes = 0

    def flush() -> None:
        nonlocal files, boxes
        if files:
            db.execute(insert(File), files)
        if boxes:
            db.execute(insert(Detection), boxes)
        db.commit()
        files, boxes = [], []

    for v in range(args.videos):
        file_id = str(uuid.uuid4())
        files.append(
            {
                "id": file_id,
                "deployment_id": deployment_id,
                "file_path": f"/data/tunnel-march/{v:06d}.mp4",
                "file_type": "video",
                "file_format": "mp4",
                "captured_at_local": start + timedelta(minutes=7 * v),
                "observation_type": "animal",
                "best_frame_number": best_frame,
                "frames_processed": list(range(args.frames)),
                "width_px": 1920,
                "height_px": 1080,
            }
        )
        for frame in range(args.frames):
            for b in range(args.boxes):
                boxes.append(
                    {
                        "id": str(uuid.uuid4()),
                        "file_id": file_id,
                        "category": "animal",
                        "confidence": 0.5 + (b / (2 * args.boxes)),
                        "bbox_x": 0.1 + 0.1 * b,
                        "bbox_y": 0.2,
                        "bbox_width": 0.1,
                        "bbox_height": 0.1,
                        "label": "red fox",
                        "label_confidence": 0.8,
                        "frame_number": frame,
                    }
                )
        total_boxes += args.frames * args.boxes
        if len(boxes) >= INSERT_CHUNK:
            flush()
            print(f"  {v + 1:,} videos, {total_boxes:,} boxes", end="\r")
    flush()
    print(f"\nSeeded project {project_id}: {args.videos:,} videos, {total_boxes:,} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
