"""`detect_to_json` resumes from a checkpoint that exists, and reports
progress over the whole folder while MegaDetector counts only what is
left. The subprocess is faked; the command line and the progress
callback are what is under test.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from app.ml.inference import megadetector as md_module
from app.ml.inference.megadetector import MegaDetectorV1000


class _FakeEnvManager:
    def get_python(self, env_name: str) -> Path:
        return Path("/usr/bin/python3")


class _FakeProcess:
    """Stands in for the `run_detector_batch` subprocess: writes one result
    per listed image to the output file and replays a few tqdm lines. On a
    resume it does what MegaDetector does: the checkpoint's entries come
    first, verbatim, whether or not their files are still in the list."""

    def __init__(self, cmd, lines, **kwargs):
        self.cmd = cmd
        file_list = json.loads(Path(cmd[-2]).read_text())
        results = []
        if "--resume_from_checkpoint" in cmd:
            checkpoint = Path(cmd[cmd.index("--resume_from_checkpoint") + 1])
            results = json.loads(checkpoint.read_text())["checkpoint"]
        done = {r["file"] for r in results}
        results += [{"file": f, "detections": []} for f in file_list if f not in done]
        Path(cmd[-1]).write_text(json.dumps({"images": results}))
        self.stdout = io.StringIO("\n".join(lines) + "\n")
        self.returncode = 0

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(md_module, "cuda_guard_overrides", lambda env_manager: {})
    model_path = tmp_path / "md.pt"
    model_path.write_bytes(b"weights")
    return MegaDetectorV1000(model_path, _FakeEnvManager())


@pytest.fixture
def folder(tmp_path):
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    for i in range(4):
        (deployment / f"IMG_{i}.jpg").write_bytes(b"jpg")
    return deployment


def _run(model, folder, monkeypatch, *, lines, **kwargs):
    commands: list[list[str]] = []
    calls: list[tuple] = []

    def fake_popen(cmd, **popen_kwargs):
        commands.append(cmd)
        return _FakeProcess(cmd, lines)

    monkeypatch.setattr(md_module, "popen_group", fake_popen)

    def progress(message, progress_value, metrics=None):
        calls.append((message, progress_value, metrics))

    model.detect_to_json(
        image_paths=sorted(folder.glob("*.jpg")),
        deployment_folder=folder,
        confidence_threshold=0.01,
        progress_callback=progress,
        output_path=folder / "detection_image.json",
        **kwargs,
    )
    return commands[0], calls


def test_no_checkpoint_file_means_a_fresh_run(model, folder, monkeypatch):
    checkpoint = folder / "md_checkpoint.json"
    cmd, calls = _run(
        model, folder, monkeypatch, lines=[],
        checkpoint_path=checkpoint, checkpoint_frequency=500,
    )
    assert "--checkpoint_path" in cmd
    assert "--resume_from_checkpoint" not in cmd
    assert not any("Continuing" in c[0] for c in calls)


def test_an_existing_checkpoint_is_resumed_and_progress_is_offset(
    model, folder, monkeypatch
):
    checkpoint = folder / "md_checkpoint.json"
    checkpoint.write_text(json.dumps({"checkpoint": [
        {"file": str(folder / "IMG_0.jpg"), "detections": []},
        {"file": str(folder / "IMG_1.jpg"), "detections": []},
    ]}))
    # MegaDetector's tqdm counts the 2 remaining images.
    cmd, calls = _run(
        model, folder, monkeypatch,
        lines=[" 50%|█████     | 1/2 [00:01<00:01,  1.00it/s]"],
        checkpoint_path=checkpoint, checkpoint_frequency=500, images_done=2,
    )
    assert cmd[cmd.index("--resume_from_checkpoint") + 1] == str(checkpoint)

    resumed = [c for c in calls if c[0].startswith("Continuing")]
    assert len(resumed) == 1
    assert resumed[0][0] == "Continuing where detection stopped: 2 of 4 images already done"
    assert resumed[0][2] == {"current": 2, "total": 4}

    tqdm_calls = [c for c in calls if c[2] and "rate" in c[2]]
    assert tqdm_calls, calls
    metrics = tqdm_calls[0][2]
    assert (metrics["current"], metrics["total"]) == (3, 4)
    # The phase bar reads over the whole folder too: 3 of 4 done sits at
    # 0.1 + 0.8 * 0.75.
    assert tqdm_calls[0][1] == pytest.approx(0.1 + 0.8 * 0.75)


def test_batch_mode_remap_happens_before_the_offset(model, folder, monkeypatch):
    checkpoint = folder / "md_checkpoint.json"
    checkpoint.write_text(json.dumps({"checkpoint": [
        {"file": str(folder / "IMG_0.jpg"), "detections": []},
        {"file": str(folder / "IMG_1.jpg"), "detections": []},
    ]}))
    # batch_size 2 over the 2 remaining images is a single batch; tqdm
    # reports 1/1 batches, which is 2 images, plus the 2 already done.
    _, calls = _run(
        model, folder, monkeypatch,
        lines=["100%|██████████| 1/1 [00:01<00:00,  1.00it/s]"],
        batch_size=2,
        checkpoint_path=checkpoint, checkpoint_frequency=500, images_done=2,
    )
    metrics = [c for c in calls if c[2] and "rate" in c[2]][0][2]
    assert (metrics["current"], metrics["total"]) == (4, 4)


def test_checkpoint_entries_for_files_no_longer_listed_are_dropped(
    model, folder, monkeypatch
):
    """A file renamed or removed between the crash and the resume is still
    in the checkpoint. MegaDetector keeps it; we must not, or the database
    gets a row for a file that does not exist."""
    checkpoint = folder / "md_checkpoint.json"
    gone = str(folder / "IMG_GONE.jpg")
    checkpoint.write_text(json.dumps({"checkpoint": [
        {"file": str(folder / "IMG_0.jpg"), "detections": []},
        {"file": gone, "detections": []},
    ]}))
    _run(
        model, folder, monkeypatch, lines=[],
        checkpoint_path=checkpoint, checkpoint_frequency=500, images_done=2,
    )
    written = json.loads((folder / "detection_image.json").read_text())
    files = sorted(img["file"] for img in written["images"])
    assert files == ["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]
    assert "IMG_GONE.jpg" not in files
