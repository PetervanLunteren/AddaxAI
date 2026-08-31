"""The rules behind resuming an interrupted image detection.

One meta file decides whether anything in the artifacts folder may be
reused; see "Resuming an interrupted analysis" in DEVELOPERS.md.
"""

import json
from pathlib import Path

import pytest

from app.ml.detection_checkpoint import (
    CHECKPOINT_FILE,
    CHECKPOINT_FILES,
    IMAGE_DETECTION_JSON,
    META_FILE,
    CheckpointMeta,
    ResumeState,
    checkpoint_frequency,
    discard,
    inspect,
)

META = CheckpointMeta(
    detection_model_id="MD1000-REDWOOD", image_size=None, augment=False, image_count=2281
)


# --- frequency ------------------------------------------------------------


def test_frequency_floor_is_500():
    assert checkpoint_frequency(0, None) == 500
    assert checkpoint_frequency(12, None) == 500
    assert checkpoint_frequency(49_999, None) == 500


def test_frequency_grows_with_the_folder_to_cap_the_rewrites():
    # MegaDetector rewrites the whole results list at every checkpoint, so a
    # 100k-image folder gets one per thousand images, not one per 500.
    assert checkpoint_frequency(100_000, None) == 1000
    assert checkpoint_frequency(50_000, 1) == 500


def test_frequency_is_a_multiple_of_the_batch_size():
    # In batch mode MegaDetector counts whole batches and only writes when
    # the count is an exact multiple of the frequency: 16 never hits 500.
    assert checkpoint_frequency(2281, 16) == 512
    assert checkpoint_frequency(100_000, 16) == 1008
    assert checkpoint_frequency(100_000, 7) % 7 == 0
    assert checkpoint_frequency(0, 64) == 512


# --- meta -------------------------------------------------------------------


def test_meta_round_trips_and_writes_atomically(tmp_path: Path):
    META.write(tmp_path)
    assert CheckpointMeta.read(tmp_path) == META
    assert sorted(p.name for p in tmp_path.iterdir()) == [META_FILE]


@pytest.mark.parametrize("junk", ["", "{", "[]", '{"unexpected": 1}'])
def test_unreadable_meta_reads_as_absent(tmp_path: Path, junk: str):
    (tmp_path / META_FILE).write_text(junk)
    assert CheckpointMeta.read(tmp_path) is None


# --- inspect ----------------------------------------------------------------


def _checkpoint(folder: Path, n: int) -> None:
    (folder / CHECKPOINT_FILE).write_text(
        json.dumps({"checkpoint": [{"file": f"{i}.jpg"} for i in range(n)]})
    )


def test_nothing_to_resume_in_an_empty_folder(tmp_path: Path):
    assert inspect(tmp_path, META) is None


def test_a_checkpoint_without_meta_is_not_trusted(tmp_path: Path):
    _checkpoint(tmp_path, 500)
    assert inspect(tmp_path, META) is None


def test_partial_resume_counts_the_checkpoint_entries(tmp_path: Path):
    META.write(tmp_path)
    _checkpoint(tmp_path, 1000)
    assert inspect(tmp_path, META) == ResumeState(
        complete=False, images_done=1000, images_total=2281
    )


def test_a_finished_detection_json_is_a_complete_resume(tmp_path: Path):
    META.write(tmp_path)
    (tmp_path / IMAGE_DETECTION_JSON).write_text(json.dumps({"images": []}))
    assert inspect(tmp_path, META) == ResumeState(
        complete=True, images_done=2281, images_total=2281
    )


def test_a_finished_detection_json_wins_over_a_leftover_checkpoint(tmp_path: Path):
    META.write(tmp_path)
    _checkpoint(tmp_path, 1000)
    (tmp_path / IMAGE_DETECTION_JSON).write_text(json.dumps({"images": []}))
    state = inspect(tmp_path, META)
    assert state is not None and state.complete


@pytest.mark.parametrize(
    "other",
    [
        CheckpointMeta("MD5A-0-0", None, False, 2281),
        CheckpointMeta("MD1000-REDWOOD", 1280, False, 2281),
        CheckpointMeta("MD1000-REDWOOD", None, True, 2281),
        CheckpointMeta("MD1000-REDWOOD", None, False, 2282),
    ],
)
def test_any_setting_difference_means_no_resume(tmp_path: Path, other: CheckpointMeta):
    META.write(tmp_path)
    _checkpoint(tmp_path, 1000)
    assert inspect(tmp_path, other) is None


def test_a_truncated_checkpoint_reads_as_absent(tmp_path: Path):
    META.write(tmp_path)
    (tmp_path / CHECKPOINT_FILE).write_text('{"checkpoint": [{"file": "a.jpg"}, {"fi')
    assert inspect(tmp_path, META) is None


def test_a_truncated_detection_json_falls_back_to_the_checkpoint(tmp_path: Path):
    META.write(tmp_path)
    _checkpoint(tmp_path, 700)
    (tmp_path / IMAGE_DETECTION_JSON).write_text('{"images": [{"file": "a.jp')
    assert inspect(tmp_path, META) == ResumeState(
        complete=False, images_done=700, images_total=2281
    )


# --- discard ----------------------------------------------------------------


def test_discard_removes_only_the_checkpoint_files(tmp_path: Path):
    for name in CHECKPOINT_FILES:
        (tmp_path / name).write_text("{}")
    (tmp_path / "results.json").write_text("{}")
    (tmp_path / "video_frames").mkdir()

    discard(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["results.json", "video_frames"]


def test_discard_is_a_no_op_on_an_empty_folder(tmp_path: Path):
    discard(tmp_path)
    assert list(tmp_path.iterdir()) == []
