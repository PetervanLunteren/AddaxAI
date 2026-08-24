"""A user media path is the deployment folder plus the JSON's relative
entry, in the form the user picked. It is never `resolve()`d.

On Windows, `Path.resolve()` expands a mapped or `subst` drive letter to
its UNC target (CPython bpo-37993), while the deployment folder keeps the
`U:\\...` form the user chose. Resolving the file and then calling
`relative_to(deployment_folder)` on it raised ValueError for every video,
which failed a beta tester's whole deployment (Eco-Web, 2026-08-20) and
looked like an AVI bug because only the video paths reach a relative_to.

A symlinked folder reproduces the divergence on POSIX: `resolve()`
follows the link, `relative_to(link)` fails. Windows cannot create
symlinks without privileges and never runs these tests; CI is
Linux/macOS. See DEVELOPERS.md "Paths to user media are never resolved".
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from app.ml.best_frame import select_best_frames_streaming  # noqa: E402
from app.ml.json_pipeline import run_classification_on_json  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink creation needs privileges on Windows"
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# The files that produce or consume the stored media path form. A
# `.resolve()` in any of them reintroduces the mapped-drive failure.
MEDIA_PATH_FILES = (
    "app/ml/json_pipeline.py",
    "app/ml/best_frame.py",
    "app/ml/postprocessing.py",
    "app/api/crud/deployment_split.py",
)


@pytest.fixture
def linked_folder(tmp_path):
    """A deployment folder reached through a symlink.

    Returns (real, link). The link is what the "user picked"; resolving
    anything under it lands in `real`, which is the divergence under test.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert link.resolve() == real  # the premise of every test here
    return real, link


def _write_video_json(path: Path, video_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
                "images": [
                    {
                        "file": video_name,
                        "detections": [
                            {
                                "category": "1",
                                "conf": 0.9,
                                "bbox": [0.4, 0.4, 0.2, 0.2],
                                "frame_number": 3,
                            }
                        ],
                    }
                ],
            }
        )
    )


class _RecordingClassifier:
    """Captures the path-keyed structures the pipeline hands the worker."""

    def __init__(self):
        self.scoring = None
        self.best_frame_outputs = None

    def classify_detections(
        self,
        items,
        *,
        best_frame_outputs,
        scoring_detections,
        batch_size,
        progress_callback,
        device_callback=None,
        job_id,
    ):
        self.scoring = scoring_detections
        self.best_frame_outputs = best_frame_outputs
        return [None] * len(items), {}, "cpu", {}


def test_classifier_path_keeps_the_picked_folder_form(
    linked_folder, tmp_path, make_video
):
    real, link = linked_folder
    make_video(real / "clip.mp4", total_frames=12, fps=10)
    out = tmp_path / "detection_video.json"
    _write_video_json(out, "clip.mp4")

    stub = _RecordingClassifier()
    asyncio.run(
        run_classification_on_json(
            json_path=out,
            classification_model=stub,
            deployment_folder=link,
            batch_size=8,
            classification_gate=0.2,
            best_frame_output_base=tmp_path / "frames",
        )
    )

    picked = str(link / "clip.mp4")
    assert list(stub.scoring) == [picked]
    assert stub.best_frame_outputs == {picked: str(tmp_path / "frames" / "clip.mp4")}


def test_no_classifier_path_keeps_the_picked_folder_form(
    linked_folder, tmp_path, make_video
):
    real, link = linked_folder
    make_video(real / "clip.mp4", total_frames=12, fps=10)
    out = tmp_path / "detection_video.json"
    _write_video_json(out, "clip.mp4")

    select_best_frames_streaming(out, link, tmp_path / "frames")

    written = sorted(p.name for p in (tmp_path / "frames" / "clip.mp4").glob("*.jpg"))
    assert written == ["frame000003.jpg"]
    assert json.loads(out.read_text())["images"][0]["best_frame_number"] == 3


def test_media_path_code_never_resolves():
    """The guard. Every file that writes or compares a stored media path
    must build it as `deployment_folder / relative`, nothing else."""
    offenders = [
        rel
        for rel in MEDIA_PATH_FILES
        if ".resolve()" in (BACKEND_ROOT / rel).read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f".resolve() found in {offenders}. User media paths are never resolved: "
        "on Windows that turns a mapped drive into its UNC form and breaks "
        "every relative_to against the deployment folder. See DEVELOPERS.md "
        '"Paths to user media are never resolved".'
    )
