"""
Tests for Settings path resolution.

ADDAXAI_USER_DATA_DIR must relocate the whole app: database_url and
models_dir derive from user_data_dir unless set explicitly, and the ML
managers resolve their directories through settings instead of
hardcoding the home folder. These tests pin that contract, including
the pydantic behavior it relies on (env-sourced values appear in
model_fields_set).
"""

from pathlib import Path

import pytest

from app.core.config import Settings


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove the path env vars so each test controls them explicitly."""
    for name in (
        "ADDAXAI_USER_DATA_DIR",
        "ADDAXAI_DATABASE_URL",
        "ADDAXAI_MODELS_DIR",
        "ADDAXAI_HF_ENDPOINT",
        "HF_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_user_data_dir_env_derives_database_url_and_models_dir(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    settings = Settings()
    assert settings.user_data_dir == tmp_path
    assert settings.database_url == f"sqlite:///{tmp_path / 'addaxai.db'}"
    assert settings.models_dir == tmp_path / "models"
    # The mkdir side effect must follow the derived path too.
    assert settings.models_dir.is_dir()


def test_explicit_database_url_env_wins(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = f"sqlite:///{tmp_path / 'elsewhere.db'}"
    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path / "data"))
    clean_env.setenv("ADDAXAI_DATABASE_URL", explicit)
    settings = Settings()
    assert settings.database_url == explicit
    assert settings.models_dir == tmp_path / "data" / "models"


def test_explicit_models_dir_env_wins(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path / "data"))
    clean_env.setenv("ADDAXAI_MODELS_DIR", str(tmp_path / "elsewhere"))
    settings = Settings()
    assert settings.models_dir == tmp_path / "elsewhere"
    assert settings.database_url == f"sqlite:///{tmp_path / 'data' / 'addaxai.db'}"


def test_explicit_kwargs_win(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pins the construction path tests/db uses to build isolated Settings.
    explicit_url = f"sqlite:///{tmp_path / 'kw.db'}"
    settings = Settings(
        user_data_dir=tmp_path / "data",
        database_url=explicit_url,
        models_dir=tmp_path / "kw-models",
    )
    assert settings.database_url == explicit_url
    assert settings.models_dir == tmp_path / "kw-models"


def test_default_home_layout(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With no env vars everything derives from <home>/AddaxAI. Path.home
    # is patched so the test never touches the real home folder.
    clean_env.setattr(Path, "home", lambda: tmp_path)
    settings = Settings()
    assert settings.user_data_dir == tmp_path / "AddaxAI"
    assert settings.database_url == f"sqlite:///{tmp_path / 'AddaxAI' / 'addaxai.db'}"
    assert settings.models_dir == tmp_path / "AddaxAI" / "models"


def test_blank_env_values_treated_as_unset(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A defined-but-empty env var (half-filled GPO entry) must behave
    # exactly like an absent one, matching Electron's falsy check.
    clean_env.setattr(Path, "home", lambda: tmp_path)
    clean_env.setenv("ADDAXAI_USER_DATA_DIR", "")
    clean_env.setenv("ADDAXAI_DATABASE_URL", "   ")
    clean_env.setenv("ADDAXAI_MODELS_DIR", "")
    settings = Settings()
    assert settings.user_data_dir == tmp_path / "AddaxAI"
    assert settings.database_url == f"sqlite:///{tmp_path / 'AddaxAI' / 'addaxai.db'}"
    assert settings.models_dir == tmp_path / "AddaxAI" / "models"


def test_relative_user_data_dir_rejected(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Relative paths resolve against the working directory, which
    # differs per process, so they are refused instead of guessed at.
    for value in ("relative-dir", "~/somewhere"):
        clean_env.setenv("ADDAXAI_USER_DATA_DIR", value)
        with pytest.raises(Exception, match="absolute"):
            Settings()


def test_environment_manager_paths_follow_user_data_dir(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.ml.environment_manager import EnvironmentManager

    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    # Pre-create the micromamba binary; a missing one triggers a real
    # network download at construction time.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("micromamba", "micromamba.exe"):
        (bin_dir / name).touch()
    manager = EnvironmentManager()
    assert manager.envs_dir == tmp_path / "envs"
    assert manager.micromamba_path.parent == bin_dir


def test_model_managers_follow_models_dir(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.ml.catalog_updater import ModelCatalogUpdater
    from app.ml.manifest_manager import ManifestManager
    from app.ml.model_storage import ModelStorage

    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    assert ModelStorage().models_dir == tmp_path / "models"
    assert ModelCatalogUpdater().models_dir == tmp_path / "models"
    assert ManifestManager().models_dir == tmp_path / "models"

    # An explicit MODELS_DIR moves all three the same way.
    clean_env.setenv("ADDAXAI_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert ModelStorage().models_dir == tmp_path / "elsewhere"
    assert ModelCatalogUpdater().models_dir == tmp_path / "elsewhere"
    assert ManifestManager().models_dir == tmp_path / "elsewhere"


def test_hf_endpoint_prefixed_var_and_fallback(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    # Unset: no mirror.
    assert Settings().hf_endpoint is None
    # The ecosystem-standard name still works (early China adopters).
    clean_env.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    assert Settings().hf_endpoint == "https://hf-mirror.com"
    # The prefixed name is the documented one and wins.
    clean_env.setenv("ADDAXAI_HF_ENDPOINT", "https://mirror.example")
    assert Settings().hf_endpoint == "https://mirror.example"


def test_unprefixed_vars_are_ignored(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Generic names like DATABASE_URL are common in other tooling; a
    # stray export must not redirect the app.
    clean_env.setattr(Path, "home", lambda: tmp_path)
    clean_env.setenv("USER_DATA_DIR", str(tmp_path / "stray"))
    clean_env.setenv("DATABASE_URL", "postgresql://stray/db")
    settings = Settings()
    assert settings.user_data_dir == tmp_path / "AddaxAI"
    assert settings.database_url == f"sqlite:///{tmp_path / 'AddaxAI' / 'addaxai.db'}"


def test_prefixed_vars_without_a_field_do_not_crash(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Electron sets its own ADDAXAI_-prefixed vars (ADDAXAI_BACKEND_PORT,
    # ADDAXAI_SLOW_NOTICE_MS) in the environment the backend inherits.
    # pydantic-settings must keep ignoring prefixed env vars that have no
    # Settings field; a pydantic-settings bump that starts rejecting them
    # would kill every packaged launch.
    clean_env.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    clean_env.setenv("ADDAXAI_BACKEND_PORT", "8123")
    clean_env.setenv("ADDAXAI_SLOW_NOTICE_MS", "60000")
    settings = Settings()
    assert settings.user_data_dir == tmp_path


def test_path_home_only_in_allowed_modules() -> None:
    # The home folder may only be resolved in config.py (the single
    # source of truth) and legacy_install.py (which targets the legacy
    # v6 install locations on purpose). Anything else must go through
    # settings, or USER_DATA_DIR silently stops relocating the app.
    allowed = {
        Path("app/core/config.py"),
        Path("app/services/legacy_install.py"),
    }
    backend_root = Path(__file__).resolve().parents[1]
    offenders = []
    for py_file in (backend_root / "app").rglob("*.py"):
        rel = py_file.relative_to(backend_root)
        if rel in allowed:
            continue
        text = py_file.read_text(encoding="utf-8")
        if "Path.home()" in text or 'expanduser("~' in text:
            offenders.append(str(rel))
    assert not offenders, (
        f"home folder resolved outside config/legacy_install: {offenders}"
    )
