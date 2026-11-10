"""The vendored BoT-SORT (``app/ml/inference/botsort.py``).

It replaced ultralytics' BoT-SORT on 2026-11-09 and was proved to give the
same rows frame for frame (DEVELOPERS.md, "The tracker script"). These
tests pin the behaviour that proof depends on, so a tidy-up that changes a
track fails here first. Frame numbers in the buffer test were measured on
ultralytics 8.3.177 with the same input.
"""

import numpy as np
import pytest

from app.ml.inference import botsort

BOX = [100, 100, 200, 200]


def _tracker(track_buffer: int = 6, downscale: int = 2) -> botsort.BoTSORT:
    return botsort.BoTSORT(
        high_thresh=0.5, low_thresh=0.01, match_thresh=0.97,
        track_buffer=track_buffer, gmc_downscale=downscale,
    )


def _rows(*rows: list[float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _textured(seed: int = 0, size: tuple[int, int] = (240, 320)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (*size, 3), dtype=np.uint8)


def test_iou_cost_is_float32_without_the_plus_one_pixel():
    a = [np.array([0, 0, 10, 10])]
    cost = botsort.iou_cost(a, [np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])])
    assert cost.dtype == np.float32
    assert cost[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert cost[0, 1] == pytest.approx(1 - 50 / 150, abs=1e-6)
    # Touching boxes share no pixel (the +1 convention would say they do).
    assert botsort.iou_cost(a, [np.array([10, 0, 20, 10])])[0, 0] == 1.0
    # A zero-area box costs 1, not nan.
    assert botsort.iou_cost(a, [np.array([3, 3, 3, 3])])[0, 0] == 1.0
    assert botsort.iou_cost([], a).shape == (0, 1)


def test_assignment_respects_the_cost_limit():
    cost = np.array([[0.1, 0.99], [0.99, 0.98]], dtype=np.float32)
    matches, rows, cols = botsort.linear_assignment(cost, 0.97)
    assert matches == [[0, 0]]
    assert rows.tolist() == [1] and cols.tolist() == [1]
    matches, rows, cols = botsort.linear_assignment(np.zeros((0, 2), np.float32), 0.97)
    assert matches == [] and rows.tolist() == [] and cols.tolist() == [0, 1]


def test_a_box_exactly_at_the_high_threshold_starts_a_track():
    """``>=``, as ultralytics: MegaDetector rounds to three decimals, so a
    confidence of exactly the threshold is common."""
    confirmed, _ = _tracker().update(_rows([*BOX, 0.5, 0]), None)
    assert confirmed[:, 4].tolist() == [1]
    confirmed, born = _tracker().update(_rows([*BOX, 0.499, 0]), None)
    assert len(confirmed) == 0 and len(born) == 0


def test_the_first_frame_confirms_at_once_later_frames_report_a_birth_first():
    tracker = _tracker()
    far = [400, 400, 450, 450]
    confirmed, born = tracker.update(_rows([*BOX, 0.9, 0]), None)
    assert confirmed[:, 4].tolist() == [1] and len(born) == 0
    confirmed, born = tracker.update(_rows([*BOX, 0.9, 0], [*far, 0.8, 0]), None)
    assert confirmed[:, 4].tolist() == [1] and born[:, 4].tolist() == [2]
    confirmed, born = tracker.update(_rows([*BOX, 0.9, 0], [*far, 0.8, 0]), None)
    assert confirmed[:, 4].tolist() == [1, 2] and len(born) == 0


def test_a_birth_not_matched_on_the_next_frame_is_gone():
    tracker = _tracker()
    tracker.update(_rows([*BOX, 0.9, 0]), None)
    tracker.update(_rows([*BOX, 0.9, 0], [400, 400, 450, 450, 0.8, 0]), None)
    confirmed, born = tracker.update(_rows([*BOX, 0.9, 0]), None)
    assert confirmed[:, 4].tolist() == [1] and len(born) == 0
    confirmed, born = tracker.update(_rows([*BOX, 0.9, 0], [400, 400, 450, 450, 0.8, 0]), None)
    assert born[:, 4].tolist() == [3]


def test_rows_carry_the_detectors_own_box_and_score():
    """The tracker's own estimate trails a moving animal; every row is the
    detection it matched."""
    tracker = _tracker()
    for step in range(6):
        row = [100 + 7 * step, 100, 200 + 7 * step, 200, 0.9 - step / 100, 0]
        confirmed, _ = tracker.update(_rows(row), None)
        assert np.array_equal(confirmed[0], np.float32([*row[:4], 1, row[4], 0]))
    assert not np.allclose(tracker.tracked[0].xyxy, confirmed[0, :4])


def test_low_score_boxes_extend_a_track_but_never_start_one():
    tracker = _tracker()
    tracker.update(_rows([*BOX, 0.9, 0]), None)
    confirmed, _ = tracker.update(_rows([*BOX, 0.2, 0]), None)
    assert confirmed[:, 4].tolist() == [1]
    assert confirmed[0, 5] == np.float32(0.2)
    confirmed, born = _tracker().update(_rows([*BOX, 0.2, 0]), None)
    assert len(confirmed) == 0 and len(born) == 0


@pytest.mark.parametrize(("back", "same_animal"), [(5, True), (6, True), (7, False)])
def test_a_lost_track_is_kept_for_the_buffer_and_one_frame_more(back, same_animal):
    """Seen on frames 1 and 2, buffer 2: it expires on frame 5 but, by
    ByteTrack's list order, can still be found on frame 6. Measured the
    same on ultralytics 8.3.177."""
    tracker = _tracker(track_buffer=2)
    empty = _rows()
    for frame in range(1, back + 1):
        confirmed, born = tracker.update(
            _rows([*BOX, 0.9, 0]) if frame in (1, 2, back) else empty, None
        )
    if same_animal:
        assert confirmed[:, 4].tolist() == [1]
    else:
        assert len(confirmed) == 0 and born[:, 4].tolist() == [2]


def test_camera_motion_failure_falls_back_to_no_motion(monkeypatch):
    def broken(_frame):
        raise RuntimeError("no transform")

    with_frames, without = _tracker(), _tracker()
    monkeypatch.setattr(with_frames.gmc, "apply", broken)
    for step in range(4):
        rows = _rows([100 + 3 * step, 100, 200 + 3 * step, 200, 0.9, 0])
        a, _ = with_frames.update(rows, _textured(step))
        b, _ = without.update(rows, None)
        assert np.array_equal(a, b)
    assert np.array_equal(with_frames.tracked[0].mean, without.tracked[0].mean)


def test_camera_motion_on_a_flat_frame_is_none_and_recovers():
    gmc = botsort.GMC(downscale=2)
    flat = np.zeros((240, 320, 3), np.uint8)
    assert np.array_equal(gmc.apply(flat), np.eye(2, 3))
    assert np.array_equal(gmc.apply(flat), np.eye(2, 3))
    assert gmc.apply(_textured()).shape == (2, 3)
    assert gmc.apply(_textured()).shape == (2, 3)


def test_camera_motion_is_scaled_back_from_the_shrunk_frame():
    frame = _textured(size=(480, 640))
    gmc = botsort.GMC(downscale=2)
    gmc.apply(frame)
    H = gmc.apply(np.roll(frame, 8, axis=1))
    assert abs(H[0, 2] - 8) < 1 and abs(H[1, 2]) < 1


def test_detections_must_have_six_columns():
    with pytest.raises(ValueError, match="6"):
        _tracker().update(np.zeros((1, 5), np.float32), None)
