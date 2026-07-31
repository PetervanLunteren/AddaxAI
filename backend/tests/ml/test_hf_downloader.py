"""
Tests for the two options that make a targeted model update possible:
restricting which files are described, and overwriting a same-size file.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.ml.hf_downloader import HuggingFaceRepoDownloader

REPO_FILES = [
    "README.md",
    "inference.py",
    "taxonomy.csv",
    "weights.pt",
    "dinov2/__init__.py",
    "dinov2/layers/attention.py",
]


@pytest.fixture
def downloader() -> HuggingFaceRepoDownloader:
    return HuggingFaceRepoDownloader()


def test_include_restricts_the_per_file_metadata_calls(
    downloader: HuggingFaceRepoDownloader,
) -> None:
    """
    The filter has to run before the metadata loop, because that loop costs
    one HTTP call per file. Updating one file in a repo that vendors a whole
    source tree must not pay for all of it.
    """
    with (
        patch.object(downloader.api, "list_repo_files", return_value=REPO_FILES),
        patch.object(
            downloader.api,
            "get_paths_info",
            return_value=[SimpleNamespace(size=10)],
        ) as mock_paths,
    ):
        total, files_info = downloader.get_repo_info("repo", include={"inference.py"})

    assert [f["path"] for f in files_info] == ["inference.py"]
    assert mock_paths.call_count == 1
    assert total == 10


def test_no_include_describes_the_whole_repo(
    downloader: HuggingFaceRepoDownloader,
) -> None:
    with (
        patch.object(downloader.api, "list_repo_files", return_value=REPO_FILES),
        patch.object(
            downloader.api,
            "get_paths_info",
            return_value=[SimpleNamespace(size=10)],
        ) as mock_paths,
    ):
        _, files_info = downloader.get_repo_info("repo")

    assert [f["path"] for f in files_info] == REPO_FILES
    assert mock_paths.call_count == len(REPO_FILES)


def test_include_that_matches_nothing_yields_nothing(
    downloader: HuggingFaceRepoDownloader,
) -> None:
    with (
        patch.object(downloader.api, "list_repo_files", return_value=REPO_FILES),
        patch.object(downloader.api, "get_paths_info") as mock_paths,
    ):
        total, files_info = downloader.get_repo_info("repo", include={"nope.txt"})

    assert files_info == []
    assert total == 0
    mock_paths.assert_not_called()


def test_existing_file_of_the_same_size_is_skipped(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    (tmp_path / "inference.py").write_bytes(b"abcde")
    info = {"path": "inference.py", "size": 5, "url": "https://example.com/f"}

    with patch.object(downloader, "_download_stream") as mock_stream:
        assert downloader.download_file(info, tmp_path) is True

    mock_stream.assert_not_called()


def test_overwrite_fetches_a_same_size_file_anyway(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """
    Size equality does not mean the contents match. An upstream edit of the
    same byte length would otherwise be skipped by the very call that came to
    replace it, leaving an update prompt that can never be cleared.
    """
    (tmp_path / "inference.py").write_bytes(b"abcde")
    info = {"path": "inference.py", "size": 5, "url": "https://example.com/f"}

    def _write(url: str, temp_path: Path, *args: object, **kwargs: object) -> int:
        temp_path.write_bytes(b"vwxyz")
        return 5

    with patch.object(downloader, "_download_stream", side_effect=_write):
        assert downloader.download_file(info, tmp_path, overwrite=True) is True

    assert (tmp_path / "inference.py").read_bytes() == b"vwxyz"
