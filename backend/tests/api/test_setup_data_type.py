"""First-run setup installs for the data type the user picked.

The wizard asks once whether the user works on camera traps or
underwater footage, and that answer decides two things: which detector
is downloaded, and which environments are built. Getting it wrong is
expensive in both directions, a 268 MB detector nobody will pick or a
1.9 GB environment nobody needs, so each half is pinned here.

The base environment is in every list on purpose: the app's ffmpeg
lives in it, so it decodes every video frame a user looks at.
"""

from types import SimpleNamespace

import pytest

from app.api.routers import setup as setup_router


@pytest.fixture(autouse=True)
def clean_install_state():
    """Module-level state leaks between tests otherwise."""
    setup_router._install_state.finish(None)
    yield
    setup_router._install_state.finish(None)


@pytest.fixture
def own_data_dir(tmp_path, monkeypatch):
    """A throwaway data dir, so no test reads or writes the real one."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    return tmp_path


def _write_weights(models_dir, data_type: str) -> None:
    """Put every default weight of one data type on disk."""
    for spec in setup_router._DEFAULT_MODELS[data_type]:
        weight = (
            models_dir / spec["type_dir"] / spec["model_id"] / spec["model_fname"]
        )
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"weights")


class _FakeStorage:
    """Records what would have been downloaded."""

    downloaded: list[str] = []

    def __init__(self, models_dir):
        self.models_dir = models_dir

    def download_weights(self, manifest, cb):
        _FakeStorage.downloaded.append(manifest.model_id)


def _stub_install(monkeypatch, present_envs: tuple[str, ...] = ()):
    """Run setup against fakes: nothing is built and nothing is fetched.

    Returns the two lists the assertions read, envs created and models
    downloaded, in the order run_setup queued them.
    """
    created: list[str] = []
    _FakeStorage.downloaded = []

    def get_or_create_env(manifest, cb):
        created.append(manifest.env)

    monkeypatch.setattr(
        setup_router,
        "_get_env_manager",
        lambda: SimpleNamespace(get_or_create_env=get_or_create_env),
    )
    monkeypatch.setattr(
        setup_router, "_env_present", lambda name=None: name in present_envs
    )
    monkeypatch.setattr(setup_router, "ModelStorage", _FakeStorage)
    return created, _FakeStorage.downloaded


def test_underwater_installs_the_fish_detector_and_the_marine_env(
    own_data_dir, monkeypatch
) -> None:
    created, downloaded = _stub_install(monkeypatch)

    setup_router.run_setup(lambda msg, prog: None, data_type="underwater")

    assert created == ["addaxai-base", "marine"]
    assert downloaded == ["CFD-NANO-1-0", "DINOV2-VITS14"]
    assert "MD5A-0-0" not in downloaded


def test_camera_trap_is_unchanged_and_is_the_default(
    own_data_dir, monkeypatch
) -> None:
    created, downloaded = _stub_install(monkeypatch)

    setup_router.run_setup(lambda msg, prog: None)

    assert created == ["addaxai-base"]
    assert downloaded == ["MD5A-0-0", "DINOV2-VITS14"]
    assert "marine" not in created


def test_the_marine_env_is_skipped_when_it_is_already_there(
    own_data_dir, monkeypatch
) -> None:
    created, _ = _stub_install(
        monkeypatch, present_envs=("addaxai-base", "marine")
    )

    setup_router.run_setup(lambda msg, prog: None, data_type="underwater")

    assert created == []


def test_status_reads_the_models_of_the_data_type_it_is_asked_about(
    client, own_data_dir, monkeypatch
) -> None:
    """An underwater install is ready without MegaDetector, and a camera
    trap install is not ready with only the fish detector."""
    monkeypatch.setattr(setup_router, "_env_present", lambda name=None: True)
    _write_weights(own_data_dir / "models", "underwater")

    underwater = client.get("/api/setup/status?data_type=underwater").json()
    camera_trap = client.get("/api/setup/status").json()

    assert underwater["models_installed"] is True
    assert underwater["ready"] is True
    assert camera_trap["models_installed"] is False
    assert camera_trap["ready"] is False


def test_status_refuses_a_data_type_that_is_not_one_of_the_two(client) -> None:
    assert client.get("/api/setup/status?data_type=aerial").status_code == 422


def test_install_refuses_a_data_type_that_is_not_one_of_the_two(client) -> None:
    response = client.post(
        "/api/setup/install-env", json={"data_type": "aerial"}
    )
    assert response.status_code == 422


def test_the_data_type_reaches_run_setup(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        setup_router,
        "run_setup",
        lambda cb, force_envs=(), data_type="camera_trap": seen.append(data_type),
    )

    setup_router._install_env_blocking((), "underwater")

    assert seen == ["underwater"]


def test_the_disk_preflight_asks_for_more_when_a_second_env_is_needed() -> None:
    """Underwater builds two envs. A disk that fits one need not fit two,
    and the whole point of the pre-flight is to say so before the
    download starts rather than after."""
    assert setup_router._required_free_bytes(
        "underwater"
    ) > setup_router._required_free_bytes("camera_trap")
