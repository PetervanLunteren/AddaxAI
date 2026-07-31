"""
Tests for per-file staleness detection and targeted model updates.

The rule under test: every file in a model's HuggingFace repo that is not
stored in LFS and is not documentation must match the local copy byte for
byte, compared through the git blob SHA-1 that HuggingFace already reports.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from app.ml.catalog_updater import ModelCatalogUpdater
from app.ml.model_storage import ModelStorage, find_stale_files, git_blob_sha1
from app.ml.schemas.model_manifest import ModelManifest

REPO = "Addax-Data-Science/TEST-v1"


def _sibling(
    rfilename: str,
    blob_id: str | None = None,
    lfs: object | None = None,
    size: int = 0,
) -> SimpleNamespace:
    """One entry of HfApi.model_info(..., files_metadata=True).siblings."""
    return SimpleNamespace(rfilename=rfilename, blob_id=blob_id, lfs=lfs, size=size)


def _plant(model_dir: Path, rel: str, content: bytes) -> str:
    """Write a local file and return the blob_id upstream would report."""
    path = model_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return git_blob_sha1(path)


def _model_info(*siblings: SimpleNamespace) -> MagicMock:
    return MagicMock(siblings=list(siblings))


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "cls" / "TEST-v1"
    d.mkdir(parents=True)
    return d


def _manifest(**overrides: object) -> ModelManifest:
    data: dict = {
        "model_id": "TEST-v1",
        "friendly_name": "Test model",
        "env": "addaxai-base",
        "model_fname": "weights.pt",
        "description": "...",
        "developer": "x",
        "info_url": "https://example.com",
        "min_app_version": "0.1.0",
    }
    data.update(overrides)
    manifest = ModelManifest(**data)
    manifest.model_category = "classification"
    return manifest


# --- git_blob_sha1 ---------------------------------------------------------
# Both expected digests are what `git hash-object` produces. The framing is
# the only thing that can be wrong here, so it is pinned against git itself
# rather than against our own output.


def test_git_blob_sha1_matches_git(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello\n")
    assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_git_blob_sha1_of_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    assert git_blob_sha1(path) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


# --- find_stale_files ------------------------------------------------------


def test_returns_empty_when_everything_matches(model_dir: Path) -> None:
    """The zero-false-positive property, which is the whole point."""
    inference = _plant(model_dir, "inference.py", b"print('hi')\n")
    taxonomy = _plant(model_dir, "taxonomy.csv", b"model_class,class\n")
    info = _model_info(
        _sibling("inference.py", inference),
        _sibling("taxonomy.csv", taxonomy),
    )
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == []


def test_flags_modified_file(model_dir: Path) -> None:
    _plant(model_dir, "inference.py", b"old\n")
    info = _model_info(_sibling("inference.py", "0" * 40))
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == ["inference.py"]


def test_flags_edit_that_keeps_the_same_byte_length(model_dir: Path) -> None:
    """
    Content is compared, not size. Pairs with the downloader's `overwrite`
    flag: without both halves, changing 0.15 to 0.25 upstream would be
    invisible and then unfixable.
    """
    local = _plant(model_dir, "inference.py", b"threshold = 0.15\n")
    upstream = git_blob_sha1(model_dir / "inference.py")
    (model_dir / "other.py").write_bytes(b"threshold = 0.25\n")
    remote = git_blob_sha1(model_dir / "other.py")
    assert local == upstream and local != remote
    assert len(b"threshold = 0.15\n") == len(b"threshold = 0.25\n")

    info = _model_info(_sibling("inference.py", remote))
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == ["inference.py"]


def test_flags_missing_file(model_dir: Path) -> None:
    info = _model_info(_sibling("taxonomy.csv", "a" * 40))
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == ["taxonomy.csv"]


def test_never_flags_lfs_files(model_dir: Path) -> None:
    """The weights are versioned by model_id and must never be re-fetched."""
    (model_dir / "weights.pt").write_bytes(b"not the real weights")
    info = _model_info(
        _sibling("weights.pt", "b" * 40, lfs=SimpleNamespace(sha256="c" * 64))
    )
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == []


@pytest.mark.parametrize(
    "name",
    ["README.md", "LICENSE", "LICENSE.md", ".gitattributes", ".DS_Store", "manifest.json"],
)
def test_never_flags_ignored_files(model_dir: Path, name: str) -> None:
    (model_dir / name).write_bytes(b"local")
    info = _model_info(_sibling(name, "d" * 40))
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == []


def test_ignores_extra_local_files(model_dir: Path) -> None:
    """Never proposes a deletion."""
    inference = _plant(model_dir, "inference.py", b"x\n")
    _plant(model_dir, "leftover-from-an-older-version.csv", b"y\n")
    info = _model_info(_sibling("inference.py", inference))
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == []


def test_compares_nested_paths_and_ignores_nested_docs(model_dir: Path) -> None:
    """Mirrors the vendored dinov2 / dinov3 source trees."""
    good = _plant(model_dir, "dinov2/layers/attention.py", b"a\n")
    _plant(model_dir, "dinov2/models/vit.py", b"stale\n")
    (model_dir / "dinov2" / "README.md").write_bytes(b"docs")
    info = _model_info(
        _sibling("dinov2/layers/attention.py", good),
        _sibling("dinov2/models/vit.py", "e" * 40),
        _sibling("dinov2/README.md", "f" * 40),
    )
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == ["dinov2/models/vit.py"]


def test_skips_sibling_without_blob_id(model_dir: Path) -> None:
    """
    Nothing to compare against. Calling it stale would download it and flag
    it again next launch, leaving a prompt the user could never clear.
    """
    _plant(model_dir, "inference.py", b"x\n")
    info = _model_info(_sibling("inference.py", None))
    with patch("huggingface_hub.HfApi.model_info", return_value=info):
        assert find_stale_files(model_dir, REPO) == []


def test_returns_none_when_offline(model_dir: Path) -> None:
    with patch(
        "huggingface_hub.HfApi.model_info", side_effect=ConnectionError("offline")
    ):
        assert find_stale_files(model_dir, REPO) is None


def test_returns_none_for_private_repo(model_dir: Path) -> None:
    """A draft or private repo really does 404. It must not be reported."""
    with patch(
        "huggingface_hub.HfApi.model_info",
        side_effect=RepositoryNotFoundError("404", response=MagicMock()),
    ):
        assert find_stale_files(model_dir, REPO) is None


# --- ModelStorage.update_stale_files ---------------------------------------


def test_update_downloads_only_the_stale_paths(tmp_path: Path) -> None:
    """Must not wipe the directory and must not name the weights."""
    models_dir = tmp_path / "models"
    model_dir = models_dir / "cls" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.pt").write_bytes(b"weights")

    storage = ModelStorage(models_dir=models_dir)
    with (
        patch(
            "app.ml.model_storage.find_stale_files",
            return_value=["inference.py"],
        ),
        patch(
            "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
            return_value=True,
        ) as mock_download,
    ):
        assert storage.update_stale_files(_manifest()) == ["inference.py"]

    mock_download.assert_called_once()
    kwargs = mock_download.call_args.kwargs
    assert kwargs["include"] == {"inference.py"}
    assert kwargs["overwrite"] is True
    assert kwargs["local_dir"] == model_dir


def test_update_is_a_no_op_when_in_sync(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "cls" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.pt").write_bytes(b"weights")

    storage = ModelStorage(models_dir=models_dir)
    with (
        patch("app.ml.model_storage.find_stale_files", return_value=[]),
        patch(
            "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo"
        ) as mock_download,
    ):
        assert storage.update_stale_files(_manifest()) == []
    mock_download.assert_not_called()


def test_update_raises_when_model_not_installed(tmp_path: Path) -> None:
    storage = ModelStorage(models_dir=tmp_path / "models")
    with pytest.raises(FileNotFoundError):
        storage.update_stale_files(_manifest())


def test_update_raises_connection_error_when_undecidable(tmp_path: Path) -> None:
    """Distinct from a download failure, because they mean different things."""
    models_dir = tmp_path / "models"
    model_dir = models_dir / "cls" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.pt").write_bytes(b"weights")

    storage = ModelStorage(models_dir=models_dir)
    with patch("app.ml.model_storage.find_stale_files", return_value=None):
        with pytest.raises(ConnectionError):
            storage.update_stale_files(_manifest())


def test_update_leaves_weights_and_extra_files_untouched(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "cls" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.pt").write_bytes(b"the real weights")
    (model_dir / "notes.txt").write_bytes(b"mine")
    weights_stat = (model_dir / "weights.pt").stat()

    storage = ModelStorage(models_dir=models_dir)
    with (
        patch(
            "app.ml.model_storage.find_stale_files", return_value=["inference.py"]
        ),
        patch(
            "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
            return_value=True,
        ),
    ):
        storage.update_stale_files(_manifest())

    assert (model_dir / "weights.pt").read_bytes() == b"the real weights"
    assert (model_dir / "weights.pt").stat().st_mtime == weights_stat.st_mtime
    assert (model_dir / "notes.txt").read_bytes() == b"mine"


def test_update_raises_runtime_error_when_download_fails(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "cls" / "TEST-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.pt").write_bytes(b"weights")

    storage = ModelStorage(models_dir=models_dir)
    with (
        patch(
            "app.ml.model_storage.find_stale_files", return_value=["inference.py"]
        ),
        patch(
            "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
            return_value=False,
        ),
    ):
        with pytest.raises(RuntimeError):
            storage.update_stale_files(_manifest())


# --- the manifest must stay equal to its catalog entry ---------------------


def _catalog_entry() -> dict:
    return {
        "model_id": "TEST-v1",
        "friendly_name": "Test model",
        "env": "addaxai-base",
        "model_fname": "weights.pt",
        "description": "...",
        "developer": "x",
        "info_url": "https://example.com",
        "min_app_version": "0.1.0",
    }


def test_downloading_does_not_change_the_manifest(tmp_path: Path) -> None:
    """
    Regression test for the bug this feature was built on top of.

    Downloading used to write an hf_revision_sha into the local manifest.
    The catalog entry has no such key, so write_manifest found a difference
    on every single launch, rewrote the file, dropped the key, and reported
    the model as refreshed forever. Staleness detection depended on that
    key, so it never fired for anyone.
    """
    models_dir = tmp_path / "models"
    updater = ModelCatalogUpdater(models_dir=models_dir)
    entry = _catalog_entry()

    with patch.object(updater, "download_taxonomy"):
        assert updater.write_manifest("cls", entry) == "created"

        manifest_path = models_dir / "cls" / "TEST-v1" / "manifest.json"

        def _fake_download(**kwargs: object) -> bool:
            (models_dir / "cls" / "TEST-v1" / "weights.pt").write_bytes(b"w")
            return True

        with patch(
            "app.ml.hf_downloader.HuggingFaceRepoDownloader.download_repo",
            side_effect=_fake_download,
        ):
            ModelStorage(models_dir=models_dir).download_weights(_manifest())

        assert json.loads(manifest_path.read_text()) == entry
        assert updater.write_manifest("cls", entry) == "unchanged"


def test_legacy_manifest_with_revision_sha_heals_once(tmp_path: Path) -> None:
    """An install upgraded from the old scheme drops the stray key and settles."""
    models_dir = tmp_path / "models"
    updater = ModelCatalogUpdater(models_dir=models_dir)
    entry = _catalog_entry()

    manifest_path = models_dir / "cls" / "TEST-v1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({**entry, "hf_revision_sha": "a" * 40}))

    with patch.object(updater, "download_taxonomy"):
        assert updater.write_manifest("cls", entry) == "updated"
        assert updater.write_manifest("cls", entry) == "unchanged"

    assert "hf_revision_sha" not in json.loads(manifest_path.read_text())
