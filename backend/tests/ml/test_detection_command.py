"""Unit tests for the MegaDetector command builders.

These are the one piece of logic behind the advanced detection settings
(image size + augmentation): the pure functions that assemble the
subprocess command line. The real detectors always spawn a subprocess,
so the flag logic is isolated here where it can be tested directly.
"""

from pathlib import Path

from app.ml.inference.megadetector import _build_run_detector_batch_cmd
from app.ml.inference.video_detector import _build_process_video_cmd


def _image_cmd(**overrides) -> list[str]:
    kwargs = dict(
        python_path=Path("python"),
        model_path=Path("model.pt"),
        file_list_json=Path("files.json"),
        output_file=Path("out.json"),
        confidence_threshold=0.005,
        batch_size=None,
        image_size=None,
        augment=False,
    )
    kwargs.update(overrides)
    return _build_run_detector_batch_cmd(**kwargs)


def _video_cmd(**overrides) -> list[str]:
    kwargs = dict(
        python_path=Path("python"),
        model_path=Path("model.pt"),
        video_folder=Path("videos"),
        output_json=Path("out.json"),
        time_sample=0.5,
        confidence_threshold=0.005,
        image_size=None,
        augment=False,
    )
    kwargs.update(overrides)
    return _build_process_video_cmd(**kwargs)


# --- images -------------------------------------------------------------


def test_image_cmd_defaults_add_no_inference_flags():
    cmd = _image_cmd()
    assert "--image_size" not in cmd
    assert "--augment" not in cmd
    assert "--batch_size" not in cmd
    # The three trailing entries stay positional (model, file list, output).
    assert cmd[-3:] == ["model.pt", "files.json", "out.json"]


def test_image_cmd_extracts_the_camera_exif_tags():
    """The Files export's camera columns exist only if the detector was
    asked for these tags at analysis time; a tag missing here is blank in
    every export forever (reprocessing reuses the stored JSON)."""
    cmd = _image_cmd()
    tags = cmd[cmd.index("--include_exif_tags") + 1]
    assert tags == (
        "datetimeoriginal,gpsinfo,make,model,ambienttemperature,bodyserialnumber"
    )


def test_image_cmd_image_size_and_augment():
    cmd = _image_cmd(image_size=1920, augment=True)
    assert cmd[cmd.index("--image_size") + 1] == "1920"
    assert "--augment" in cmd
    # Flags land before the positional args, which stay last.
    assert cmd[-3:] == ["model.pt", "files.json", "out.json"]
    assert cmd.index("--augment") < cmd.index("model.pt")


def test_image_cmd_batch_size_unchanged_by_new_flags():
    cmd = _image_cmd(batch_size=8, image_size=2560, augment=True)
    assert cmd[cmd.index("--batch_size") + 1] == "8"
    assert cmd[cmd.index("--image_size") + 1] == "2560"
    assert "--augment" in cmd
    assert cmd[-3:] == ["model.pt", "files.json", "out.json"]


# --- videos -------------------------------------------------------------


def test_video_cmd_defaults_add_no_inference_flags():
    cmd = _video_cmd()
    assert "--image_size" not in cmd
    assert "--augment" not in cmd
    # Base process_video flags are still present.
    assert "megadetector.detection.process_video" in cmd
    assert cmd[cmd.index("--json_confidence_threshold") + 1] == "0.005"


def test_video_cmd_image_size_and_augment():
    cmd = _video_cmd(image_size=2560, augment=True)
    assert cmd[cmd.index("--image_size") + 1] == "2560"
    assert "--augment" in cmd


def test_video_cmd_augment_only():
    cmd = _video_cmd(augment=True)
    assert "--image_size" not in cmd
    assert cmd[-1] == "--augment"
