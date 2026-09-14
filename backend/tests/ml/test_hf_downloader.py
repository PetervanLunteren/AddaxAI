"""
Tests for the two options that make a targeted model update possible:
restricting which files are described, and overwriting a same-size file,
plus the per-file retry that keeps one transient failure from failing a
whole repo download.
"""

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import requests
from huggingface_hub.utils import HfHubHTTPError

from app.ml.hf_downloader import (
    _FILE_ATTEMPTS,
    HuggingFaceRepoDownloader,
    NetworkBlockedError,
)

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


def _info(size: int = 5) -> dict:
    return {"path": "inference.py", "size": size, "url": "https://example.com/f"}


def _write_after(failures: int, content: bytes = b"abcde"):
    """A _download_stream that raises `failures` times, then succeeds."""
    calls = {"n": 0}

    def _stream(url: str, temp_path: Path, *args: object, **kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] <= failures:
            # Half the bytes land before the connection dies, which is what
            # makes the progress bookkeeping below worth asserting.
            temp_path.write_bytes(content[: len(content) // 2])
            raise OSError("connection reset")
        temp_path.write_bytes(content)
        return len(content)

    return _stream, calls


def test_a_transient_failure_is_retried(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """
    The 2026-08-12 trigger. A 12 KB inference.py could not resolve
    huggingface.co while four other files in the same repo downloaded fine
    in the same second, and that one failure failed the whole 1.13 GB
    download.
    """
    stream, calls = _write_after(failures=1)

    with (
        patch.object(downloader, "_download_stream", side_effect=stream),
        patch("app.ml.hf_downloader.time.sleep") as mock_sleep,
    ):
        assert downloader.download_file(_info(), tmp_path) is True

    assert calls["n"] == 2
    assert (tmp_path / "inference.py").read_bytes() == b"abcde"
    mock_sleep.assert_called_once_with(1)


def test_a_persistent_failure_gives_up_after_the_attempt_budget(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    stream, calls = _write_after(failures=_FILE_ATTEMPTS)

    with (
        patch.object(downloader, "_download_stream", side_effect=stream),
        patch("app.ml.hf_downloader.time.sleep"),
    ):
        assert downloader.download_file(_info(), tmp_path) is False

    assert calls["n"] == _FILE_ATTEMPTS
    # Nothing half-written is left behind under the real filename.
    assert not (tmp_path / "inference.py").exists()
    assert list(tmp_path.iterdir()) == []


def test_a_retry_does_not_count_the_failed_attempt_twice(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """
    Bytes are counted as they are written, so without un-counting a dropped
    attempt the progress bar walks past 100% on every retry.
    """
    stream, _ = _write_after(failures=1)
    downloader.total_bytes = 5

    def _counting_stream(url: str, temp_path: Path, *a: object, **k: object) -> int:
        before = downloader.downloaded_bytes
        try:
            written = stream(url, temp_path, *a, **k)
        except OSError:
            downloader.update_progress(temp_path.stat().st_size)
            raise
        downloader.update_progress(written - (downloader.downloaded_bytes - before))
        return written

    with (
        patch.object(downloader, "_download_stream", side_effect=_counting_stream),
        patch("app.ml.hf_downloader.time.sleep"),
    ):
        assert downloader.download_file(_info(), tmp_path) is True

    assert downloader.downloaded_bytes == 5


def test_a_cancelled_download_is_not_retried(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """
    _download_stream returns None on cancel rather than raising, so this
    pins that the retry loop returns instead of treating it as a failure.
    """
    with patch.object(downloader, "_download_stream", return_value=None) as mock_stream:
        assert downloader.download_file(_info(), tmp_path, lambda: True) is False

    assert mock_stream.call_count == 1


def test_cancelling_during_the_backoff_stops_the_retries(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """A cancel pressed while a failed file waits to retry must not spend
    the rest of the attempt budget first."""
    # Without a budget to cut short there is nothing here to prove.
    assert _FILE_ATTEMPTS > 1
    stream, calls = _write_after(failures=_FILE_ATTEMPTS)

    with (
        patch.object(downloader, "_download_stream", side_effect=stream),
        patch("app.ml.hf_downloader.time.sleep") as mock_sleep,
    ):
        assert downloader.download_file(_info(), tmp_path, lambda: True) is False

    assert calls["n"] == 1
    mock_sleep.assert_not_called()


def test_no_authorization_header_without_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """huggingface.co serves our repos anonymously, so nothing is sent."""
    monkeypatch.delenv("ADDAXAI_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert "Authorization" not in HuggingFaceRepoDownloader().session.headers


def test_token_travels_on_the_file_downloads_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The metadata calls go through huggingface_hub, which reads the token
    itself; the file downloads are plain requests and have to be told.
    A private endpoint would otherwise list a repo and 401 every file.
    """
    monkeypatch.setenv("ADDAXAI_HF_TOKEN", "secret")
    downloader = HuggingFaceRepoDownloader()
    assert downloader.session.headers["Authorization"] == "Bearer secret"
    assert downloader.api.token == "secret"


# ---------------------------------------------------------------------------
# A network that answers with a block page.
#
# The 2026-09-11 shape: a US state network's web filter answered every call
# to huggingface.co with a 403 and a 101 KB HTML page. The hub never does
# that (its errors are small JSON bodies, and our repos are public), so the
# pair of status and content type is the whole signal.


def _hub_error(status: int, content_type: str) -> HfHubHTTPError:
    url = "https://huggingface.co/api/models/Addax-Data-Science/MD5A-0-0/tree/main"
    response = httpx.Response(
        status,
        headers={"content-type": content_type},
        request=httpx.Request("GET", url),
    )
    return HfHubHTTPError(f"{status} error", response=response)


def test_a_block_page_on_the_listing_is_reported_as_blocked(
    downloader: HuggingFaceRepoDownloader,
) -> None:
    with patch.object(
        downloader.api,
        "list_repo_files",
        side_effect=_hub_error(403, "text/html; charset=utf-8"),
    ):
        with pytest.raises(NetworkBlockedError) as exc:
            downloader.get_repo_info("Addax-Data-Science/MD5A-0-0")

    assert exc.value.host == "huggingface.co"
    # Written for the user, and names the host so IT gets it verbatim.
    assert "blocks huggingface.co" in str(exc.value)


def test_a_json_403_from_the_hub_is_an_ordinary_failure(
    downloader: HuggingFaceRepoDownloader,
) -> None:
    """The hub's own refusals are JSON. Those keep the generic wording:
    sending someone to IT over a gated repo points them the wrong way."""
    with patch.object(
        downloader.api,
        "list_repo_files",
        side_effect=_hub_error(403, "application/json; charset=utf-8"),
    ):
        with pytest.raises(RuntimeError) as exc:
            downloader.get_repo_info("Addax-Data-Science/MD5A-0-0")

    assert not isinstance(exc.value, NetworkBlockedError)


def _block_page_response(url: str) -> requests.Response:
    r = requests.Response()
    r.status_code = 403
    r.headers["content-type"] = "text/html; charset=utf-8"
    r.url = url
    r._content = b"<html>Blocked by policy</html>"
    # A real response carries its socket here; leaving the `with` block
    # closes it, and a None crashes that close before our error surfaces.
    r.raw = io.BytesIO(b"")
    return r


def test_a_block_page_on_a_file_is_not_retried(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """The files redirect to *.hf.co, so a filter that allows huggingface.co
    and blocks the CDN shows up here and nowhere else. Three attempts with a
    pause between them would only delay the same page."""
    cdn = "https://cdn-lfs.hf.co/repos/x/md_v5a.0.0.pt"
    with (
        patch.object(
            downloader.session, "get", return_value=_block_page_response(cdn)
        ) as mock_get,
        patch("app.ml.hf_downloader.time.sleep") as mock_sleep,
    ):
        with pytest.raises(NetworkBlockedError) as exc:
            downloader.download_file(_info(), tmp_path)

    assert exc.value.host == "cdn-lfs.hf.co"
    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_download_repo_raises_when_the_listing_is_blocked(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """False is what every other failure returns, and it is what turned the
    typed error back into "Download failed for <repo>"."""
    with patch.object(
        downloader, "get_repo_info", side_effect=NetworkBlockedError("huggingface.co")
    ):
        with pytest.raises(NetworkBlockedError):
            downloader.download_repo("Addax-Data-Science/MD5A-0-0", tmp_path)


def test_download_repo_raises_when_a_file_is_blocked(
    downloader: HuggingFaceRepoDownloader, tmp_path: Path
) -> None:
    """A file download runs in the executor, so its error surfaces through
    the future and has to be re-raised rather than counted as one failure."""
    files = [_info(), {**_info(), "path": "taxonomy.csv"}]
    with (
        patch.object(downloader, "get_repo_info", return_value=(10, files)),
        patch.object(
            downloader, "download_file", side_effect=NetworkBlockedError("cdn-lfs.hf.co")
        ),
    ):
        with pytest.raises(NetworkBlockedError):
            downloader.download_repo("Addax-Data-Science/MD5A-0-0", tmp_path)
