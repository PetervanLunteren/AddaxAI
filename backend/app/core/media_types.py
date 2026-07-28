"""Canonical media file extension constants. Matches MegaDetector's path_utils.IMG_EXTENSIONS."""

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".gif", ".png", ".tif", ".tiff", ".bmp"})
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mpeg", ".mpg", ".mov", ".mkv", ".flv", ".m4v", ".wmv"}
)
