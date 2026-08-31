"""
Resume an interrupted image detection.

MegaDetector's ``run_detector_batch`` can write its results so far to a
checkpoint file every N images and pick up from it on the next run. This
module holds the few rules around that file so the worker, the folder-run
lookup and the reset path all agree on them. See "Resuming an interrupted
analysis" in DEVELOPERS.md.

Three files live in the deployment's artifacts folder
(``<folder>/.addaxai/projects/<project_id>/``):

- ``md_checkpoint.json``: written by MegaDetector, deleted by it on success.
- ``md_checkpoint.meta.json``: written by us before detection starts. It
  records what the checkpoint is valid for. A checkpoint made under other
  detection settings, or for a different number of images, is discarded.
- ``detection_image.json``: the finished phase-3 output. When it is whole
  the detection is done and the phase is skipped.

One rule decides everything: the meta file must match the current run
exactly, else there is no checkpoint. No partial credit, no heuristics.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CHECKPOINT_FILE = "md_checkpoint.json"
META_FILE = "md_checkpoint.meta.json"
IMAGE_DETECTION_JSON = "detection_image.json"
# Everything a reset keeps when the user chooses Continue. The ``_tmp``
# sibling is MegaDetector's own backup of the previous checkpoint.
CHECKPOINT_FILES = (
    CHECKPOINT_FILE,
    CHECKPOINT_FILE + "_tmp",
    META_FILE,
    IMAGE_DETECTION_JSON,
)

# 500 is the example the legacy app used. MegaDetector rewrites the whole
# results list at every checkpoint, so the cost grows with the folder;
# capping the number of checkpoints per run keeps a 100k-image folder at
# about a hundred rewrites instead of two hundred.
MIN_CHECKPOINT_FREQUENCY = 500
MAX_CHECKPOINTS_PER_RUN = 100


def checkpoint_frequency(image_count: int, batch_size: int | None) -> int:
    """How many images MegaDetector processes between two checkpoints.

    In batch mode MegaDetector only writes when ``images_done %
    frequency == 0``, and it counts in whole batches, so a frequency that
    is not a multiple of the batch size never fires. Round up to one.
    """
    frequency = max(MIN_CHECKPOINT_FREQUENCY, image_count // MAX_CHECKPOINTS_PER_RUN)
    if batch_size is not None and batch_size > 1:
        frequency = -(-frequency // batch_size) * batch_size
    return frequency


def artifacts_dir(deployment_folder: Path, project_id: str) -> Path:
    """The project-scoped artifacts folder of a deployment."""
    return deployment_folder / ".addaxai" / "projects" / project_id


@dataclass(frozen=True)
class CheckpointMeta:
    """What a checkpoint is valid for. Compared as a whole."""

    detection_model_id: str
    image_size: int | None
    augment: bool
    image_count: int

    def write(self, folder: Path) -> None:
        # Written before detection starts, so a crash during this tiny
        # write must not leave a half file: an unreadable meta reads as
        # "no checkpoint" and would cost a good one on the next inspect.
        target = folder / META_FILE
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self)), encoding="utf-8")
        os.replace(tmp, target)

    @classmethod
    def read(cls, folder: Path) -> CheckpointMeta | None:
        data = _read_json(folder / META_FILE)
        if not isinstance(data, dict):
            return None
        try:
            return cls(**data)
        except TypeError:
            return None


@dataclass(frozen=True)
class ResumeState:
    """What an interrupted run left behind for the current settings."""

    # True: ``detection_image.json`` is whole, phase 3 can be skipped.
    # False: MegaDetector's checkpoint holds ``images_done`` images.
    complete: bool
    images_done: int
    images_total: int


def inspect(folder: Path, expected: CheckpointMeta) -> ResumeState | None:
    """Return what can be resumed under ``expected``, or None.

    A truncated file (crash mid-write) is a JSON error and reads as
    absent, so the run starts over rather than trusting it.
    """
    if CheckpointMeta.read(folder) != expected:
        return None
    detection = _read_json(folder / IMAGE_DETECTION_JSON)
    if isinstance(detection, dict) and isinstance(detection.get("images"), list):
        return ResumeState(
            complete=True,
            images_done=expected.image_count,
            images_total=expected.image_count,
        )
    checkpoint = _read_json(folder / CHECKPOINT_FILE)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("checkpoint"), list):
        return ResumeState(
            complete=False,
            images_done=len(checkpoint["checkpoint"]),
            images_total=expected.image_count,
        )
    return None


def discard(folder: Path) -> None:
    """Remove every checkpoint file that exists. Nothing else is touched."""
    for name in CHECKPOINT_FILES:
        try:
            (folder / name).unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> object | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
