"""
The relay in front of huggingface.co, for networks that block it.

`ModelStorage.download_weights` downloads from HuggingFace like everyone
else, and retries the same download once through our relay
(infra/hf-relay/worker.js) only when the network answered HuggingFace with
a block page. Four rules, one test each:

- a block page sends the download to the relay, nothing else does
- a user who configured their own mirror never reaches the relay
- no relay configured means the old behaviour, the block propagates
- a relay that cannot help (blocked too, unreachable, over quota, any
  failure) raises the *original* error, so the wizard names
  huggingface.co, the host IT has to allow; only a cancel stays a cancel
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.core.job_cancellation import JobCancelledError
from app.ml.hf_downloader import HuggingFaceRepoDownloader, NetworkBlockedError
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest

RELAY = "https://addaxai-models.example.workers.dev"

MANIFEST_JSON = {
    "model_id": "TEST-v1",
    "friendly_name": "Test model",
    "env": "pytorch",
    "model_fname": "weights.pt",
    "description": "...",
    "developer": "x",
    "info_url": "https://example.org",
    "min_app_version": "7.0.1",
}


def _manifest() -> ModelManifest:
    m = ModelManifest(**MANIFEST_JSON)
    m.model_category = "classification"
    return m


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "cls" / "TEST-v1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(MANIFEST_JSON))
    return d


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> str:
    """A relay configured, and no mirror, so the fallback is armed."""
    monkeypatch.delenv("ADDAXAI_HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("ADDAXAI_HF_FALLBACK_ENDPOINT", RELAY)
    return RELAY


def _blocked_at_huggingface_then_relay_succeeds(model_dir: Path):
    """The 2026-09-11 shape: huggingface.co is filtered, the relay is not."""
    endpoints: list[str] = []

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        endpoints.append(self.endpoint)
        if self.endpoint == "https://huggingface.co":
            raise NetworkBlockedError("huggingface.co")
        (model_dir / "weights.pt").write_bytes(b"w" * 16)
        return True

    return run, endpoints


def test_a_block_page_sends_the_download_through_the_relay(
    model_dir: Path, relay: str
) -> None:
    run, endpoints = _blocked_at_huggingface_then_relay_succeeds(model_dir)
    messages: list[str] = []

    with patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run):
        path = ModelStorage(models_dir=model_dir.parent.parent).download_weights(
            _manifest(), progress_callback=lambda m, p: messages.append(m)
        )

    assert endpoints == ["https://huggingface.co", relay]
    assert path == model_dir
    assert (model_dir / "weights.pt").is_file()
    # The user sees that something different is being tried, not a
    # silent second attempt.
    assert any("relay" in m for m in messages)


def test_an_ordinary_failure_never_reaches_the_relay(model_dir: Path, relay: str) -> None:
    """False from the downloader (timeout, disk) is retried by the user, not by us."""
    endpoints: list[str] = []

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        endpoints.append(self.endpoint)
        return False

    with patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run):
        with pytest.raises(RuntimeError, match="Download failed"):
            ModelStorage(models_dir=model_dir.parent.parent).download_weights(_manifest())

    assert endpoints == ["https://huggingface.co"]


def test_a_configured_mirror_is_never_overridden_by_the_relay(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ADDAXAI_HF_ENDPOINT is a deliberate choice about where downloads may
    come from (a company repository manager). A block page from it must
    not be answered by routing the download somewhere IT never approved.
    """
    monkeypatch.setenv("ADDAXAI_HF_ENDPOINT", "https://artifactory.example/hf")
    monkeypatch.setenv("ADDAXAI_HF_FALLBACK_ENDPOINT", RELAY)
    endpoints: list[str] = []

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        endpoints.append(self.endpoint)
        raise NetworkBlockedError("artifactory.example")

    with patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run):
        with pytest.raises(NetworkBlockedError):
            ModelStorage(models_dir=model_dir.parent.parent).download_weights(_manifest())

    assert endpoints == ["https://artifactory.example/hf"]


def test_without_a_relay_the_block_propagates_as_before(
    model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build with no relay address behaves as it did before the relay existed."""
    monkeypatch.delenv("ADDAXAI_HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    no_relay = Settings(user_data_dir=tmp_path, hf_fallback_endpoint=None)
    endpoints: list[str] = []

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        endpoints.append(self.endpoint)
        raise NetworkBlockedError("huggingface.co")

    with (
        patch("app.ml.model_storage.get_settings", return_value=no_relay),
        patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run),
    ):
        with pytest.raises(NetworkBlockedError):
            ModelStorage(models_dir=model_dir.parent.parent).download_weights(_manifest())

    assert endpoints == ["https://huggingface.co"]


def test_a_blocked_relay_raises_the_original_host(model_dir: Path, relay: str) -> None:
    """
    The wizard's message tells the user which host to have IT allow. That
    is huggingface.co, the fix, not the relay host that also happened to
    be filtered. The relay's error stays attached as the cause.
    """

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        raise NetworkBlockedError(self.endpoint.removeprefix("https://"))

    with patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run):
        with pytest.raises(NetworkBlockedError) as info:
            ModelStorage(models_dir=model_dir.parent.parent).download_weights(_manifest())

    assert info.value.host == "huggingface.co"
    assert isinstance(info.value.__cause__, NetworkBlockedError)
    assert info.value.__cause__.host == relay.removeprefix("https://")


@pytest.mark.parametrize(
    "relay_outcome",
    [
        pytest.param(False, id="relay-returns-false"),
        pytest.param(RuntimeError("Error fetching repository info: 429"), id="relay-over-quota"),
        pytest.param(RuntimeError("Connection refused"), id="relay-unreachable"),
    ],
)
def test_a_relay_that_cannot_help_raises_the_original_host(
    model_dir: Path, relay: str, relay_outcome: bool | Exception
) -> None:
    """
    Found by the e2e hunt on 2026-09-15: with the relay unreachable, the
    wizard fell back to the generic "Download failed for <repo>" with no
    error_kind, the exact message that got a state laptop twelve retries.
    Whatever the relay's own problem, the user's problem is still the
    blocked host, and that is what they must be told.
    """

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        if self.endpoint == "https://huggingface.co":
            raise NetworkBlockedError("huggingface.co")
        if isinstance(relay_outcome, Exception):
            raise relay_outcome
        return relay_outcome

    with patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run):
        with pytest.raises(NetworkBlockedError) as info:
            ModelStorage(models_dir=model_dir.parent.parent).download_weights(_manifest())

    assert info.value.host == "huggingface.co"
    if isinstance(relay_outcome, Exception):
        assert info.value.__cause__ is relay_outcome


def test_a_cancel_during_the_relay_download_is_a_cancel(model_dir: Path, relay: str) -> None:
    """
    A cancel must stay a cancel, so `download_weights` runs its cancelled
    cleanup rather than reporting a blocked network the user never saw.
    """

    def run(self: HuggingFaceRepoDownloader, **kwargs: object) -> bool:
        if self.endpoint == "https://huggingface.co":
            raise NetworkBlockedError("huggingface.co")
        (model_dir / "weights.pt").write_bytes(b"partial")
        raise JobCancelledError("stopped")

    with patch.object(HuggingFaceRepoDownloader, "download_repo", autospec=True, side_effect=run):
        with pytest.raises(JobCancelledError):
            ModelStorage(models_dir=model_dir.parent.parent).download_weights(_manifest())

    # The cancelled-download rule: files gone, manifest kept.
    assert sorted(p.name for p in model_dir.iterdir()) == ["manifest.json"]


def test_the_relay_downloader_talks_to_the_relay_only(relay: str) -> None:
    """Both the metadata client and the file URLs must move, or the relay covers half."""
    downloader = HuggingFaceRepoDownloader(endpoint=relay + "/")
    assert downloader.endpoint == relay
    assert downloader.api.endpoint == relay
