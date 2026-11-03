"""A track's frames are decoded in chunks, one seek for up to 64 of them.

Opening a track used to decode every frame on its own, a seek into the
clip per card, four at a time. On an external drive that chokes on
concurrent random reads each seek took minutes and the whole app queued
behind them (2026-11-02). The chunk decode seeks once and keeps only
the track's sampled frames; it must give the same bytes the single
decode gives, or a card would show a neighbouring moment.

The decode tests build a small clip with the app's own ffmpeg and skip
where it is not installed; the rest of the file fakes the decoder.
"""

import subprocess

import pytest

from app.services import video_frame_service as vfs
from app.utils.ffmpeg_bin import APP_ENV, resolve_ffmpeg

FPS = 10.0


@pytest.fixture(autouse=True)
def _empty_cache():
    vfs._frames.clear()
    vfs._clip_locks.clear()
    yield
    vfs._frames.clear()
    vfs._clip_locks.clear()


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Six seconds of the ffmpeg test pattern at 10 fps, a keyframe
    every 12 frames, so a wanted frame usually sits between keyframes."""
    try:
        ffmpeg = resolve_ffmpeg(APP_ENV)
    except RuntimeError as e:
        pytest.skip(f"no ffmpeg: {e}")
    path = tmp_path_factory.mktemp("clip") / "pattern.mp4"
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate={int(FPS)}:duration=6",
            "-pix_fmt", "yuv420p", "-g", "12", str(path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not write a test clip: {result.stderr.decode()[-200:]}")
    return str(path)


def test_the_chunk_decode_gives_the_single_decodes_bytes(clip):
    frames = [3, 7, 12, 13, 25, 40, 59]
    single = {n: vfs.decode_frame_jpeg(clip, n, FPS) for n in frames}
    assert all(single.values())
    vfs._frames.clear()

    chunk = vfs.decode_frames_jpeg(clip, frames, FPS)

    assert chunk == single
    # And it filled the cache the single decode reads.
    assert vfs.decode_frame_jpeg(clip, 25, FPS) is single[25] or (
        vfs.decode_frame_jpeg(clip, 25, FPS) == single[25]
    )


def test_frames_past_the_end_are_left_out(clip):
    got = vfs.decode_frames_jpeg(clip, [55, 59, 70, 80], FPS)
    assert sorted(got) == [55, 59]


def test_warm_decodes_one_chunk_from_the_start_frame_and_skips_what_is_cached(monkeypatch):
    """A 150-frame track is three chunks, lined up from each missed
    frame. A warm at a cached frame asks for nothing, so the frame after
    a chunk's first does not seek for the one frame past its end; a
    warm at a frame the track does not have asks for nothing either."""
    track = list(range(0, 300, 2))  # 150 frames, every other frame
    asked: list[list[int]] = []

    def fake_decode(video_path, frame_numbers, frame_rate):
        asked.append(list(frame_numbers))
        for n in frame_numbers:
            vfs._store((video_path, n, frame_rate), b"jpeg")
        return {n: b"jpeg" for n in frame_numbers}

    monkeypatch.setattr(vfs, "decode_frames_jpeg", fake_decode)

    vfs.warm_track_frames("/clip.mp4", track, 0, FPS)
    assert asked == [track[:64]]

    vfs.warm_track_frames("/clip.mp4", track, 2, FPS)  # cached: no seek for frame 128
    vfs.warm_track_frames("/clip.mp4", track, 126, FPS)  # the chunk's last, cached
    assert len(asked) == 1

    vfs.warm_track_frames("/clip.mp4", track, 128, FPS)  # the 65th frame
    assert asked[1] == track[64:128]

    vfs.warm_track_frames("/clip.mp4", track, 7, FPS)  # not a frame of the track
    assert len(asked) == 2


def test_the_cache_is_bounded_and_drops_the_oldest(monkeypatch):
    monkeypatch.setattr(vfs, "FRAME_CACHE_SIZE", 3)
    for n in range(4):
        vfs._store(("/clip.mp4", n, FPS), bytes([n]))
    assert vfs._cached(("/clip.mp4", 0, FPS)) is None
    assert vfs._cached(("/clip.mp4", 3, FPS)) == b"\x03"


def test_the_chunk_command_selects_by_offset_from_the_seek(tmp_path):
    cmd = vfs.frames_seek_cmd("ffmpeg", tmp_path / "c.mp4", 100, [0, 10, 25], 30.0, tmp_path)
    assert cmd[cmd.index("-ss") + 1] == f"{99.5 / 30.0:.6f}"
    assert cmd[cmd.index("-vf") + 1].startswith("select='eq(n\\,0)+eq(n\\,10)+eq(n\\,25)',")
    assert cmd[cmd.index("-frames:v") + 1] == "3"
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"


def test_a_decode_that_runs_past_the_timeout_is_given_up(monkeypatch, caplog):
    """A frame on a drive that costs 160 ms per read took over six
    minutes, and every request waiting on it held a browser connection
    until the viewer could not fetch its own file (2026-11-02). Past
    DECODE_TIMEOUT_S ffmpeg is stopped and the picture is missing, with
    a warning naming the likely cause; nothing hangs."""
    monkeypatch.setattr(vfs, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(vfs, "_frame_rate_of", lambda path: 10.0)

    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", slow)

    assert vfs.decode_frame_jpeg("/clip.mp4", 5, FPS) is None
    assert vfs.decode_frames_jpeg("/clip.mp4", [5, 6], FPS) == {}
    assert "took over" in caplog.text and "drive" in caplog.text
