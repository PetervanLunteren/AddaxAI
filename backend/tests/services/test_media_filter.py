"""The media filter decides which media a new analysis reads off disk.

`Project.media_filter` ("all" | "images" | "videos") exists because cameras
left in video mode by mistake produce videos that are noise and expensive to
analyse (a frame per second, each run through the detector).

Two things must agree on the rule: the counts announced in the run header
(sent before the scan, so they cannot count the filtered lists) and the file
lists handed to the detector. These tests pin that agreement — if they drift,
the header promises media the run silently skips.
"""

from app.workers.detection_worker import (
    _announced_media_counts,
    media_filter_allows,
)

# ----------------------------------------------------------------------
# media_filter_allows: the rule itself
# ----------------------------------------------------------------------


def test_all_allows_both():
    assert media_filter_allows("all", "image") is True
    assert media_filter_allows("all", "video") is True


def test_images_allows_only_images():
    assert media_filter_allows("images", "image") is True
    assert media_filter_allows("images", "video") is False


def test_videos_allows_only_videos():
    assert media_filter_allows("videos", "image") is False
    assert media_filter_allows("videos", "video") is True


def test_unknown_filter_allows_everything():
    """A filter we don't understand must not silently drop a user's files.

    Reaching this needs a value outside the Literal (a hand-edited DB, a
    future value on an older build). Dropping media on a typo is far worse
    than ignoring the setting.
    """
    assert media_filter_allows("photos", "image") is True
    assert media_filter_allows("", "video") is True


# ----------------------------------------------------------------------
# _announced_media_counts: the run header
# ----------------------------------------------------------------------


def test_announced_counts_zero_the_excluded_kind():
    """The header must not promise 12 videos and then process none."""
    assert _announced_media_counts("images", 12, 226) == (0, 226)
    assert _announced_media_counts("videos", 12, 226) == (12, 0)


def test_announced_counts_untouched_for_all():
    assert _announced_media_counts("all", 12, 226) == (12, 226)


def test_announced_counts_agree_with_the_rule():
    """The header and the file lists derive from one rule, not two.

    This is the drift guard: whatever `media_filter_allows` says about a
    kind, the announced count for that kind must follow.
    """
    for media_filter in ("all", "images", "videos"):
        videos, images = _announced_media_counts(media_filter, 7, 9)
        assert (videos > 0) is media_filter_allows(media_filter, "video")
        assert (images > 0) is media_filter_allows(media_filter, "image")
