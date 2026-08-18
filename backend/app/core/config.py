"""
Application configuration.

Following DEVELOPERS.md principles:
- Sensible defaults for bundled mode (PyInstaller)
- Can be overridden via environment variables
- Crash early if configuration is invalid
- Type hints everywhere

Every setting reads its env var with the ADDAXAI_ prefix
(ADDAXAI_USER_DATA_DIR, ADDAXAI_DATABASE_URL, ...). Unprefixed names
like DATABASE_URL are common enough in other tooling that a stray
export would silently redirect the app.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_user_data_dir() -> Path:
    """Get default user data directory."""
    return Path.home() / "AddaxAI"


def get_default_database_url() -> str:
    """Get default database URL in user's home directory."""
    db_path = get_default_user_data_dir() / "addaxai.db"
    return f"sqlite:///{db_path}"


def get_default_models_dir() -> Path:
    """Get default models directory."""
    return get_default_user_data_dir() / "models"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Provides sensible defaults for bundled mode but can be overridden.
    Crashes if directories cannot be created.
    """

    model_config = SettingsConfigDict(
        env_prefix="ADDAXAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",  # Crash if unknown env vars are provided
    )

    # Application
    app_name: str = "AddaxAI"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True
    # SQL statement echo is separate from `debug`: it logs every query +
    # params, which floods the log (and stalls the single-process server
    # on its synchronous writes) during bulk work like an analysis. Off
    # by default; flip it on only when actually debugging SQL.
    sql_echo: bool = False

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Database - defaults to local SQLite in working directory
    database_url: str = Field(default_factory=get_default_database_url)

    # User data directory - defaults to ~/AddaxAI
    user_data_dir: Path = Field(default_factory=get_default_user_data_dir)

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379

    # Models directory
    models_dir: Path = Field(default_factory=get_default_models_dir)

    # Model catalog sync
    model_catalog_url: str = Field(
        default="https://raw.githubusercontent.com/PetervanLunteren/AddaxAI/main/models.json",
        description="URL to fetch model catalog from"
    )
    disable_model_updates: bool = Field(
        default=False,
        description="Disable automatic model catalog sync on startup"
    )

    # HuggingFace mirror (mainland China). ADDAXAI_HF_ENDPOINT is the
    # documented name; the ecosystem-standard HF_ENDPOINT still works as
    # a fallback because huggingface_hub honours it natively and early
    # adopters configured it per the original docs.
    hf_endpoint: str | None = Field(
        default=None,
        description="Base URL of a HuggingFace mirror, e.g. https://hf-mirror.com"
    )

    # Bearer token for an endpoint that will not serve anonymously. Our
    # repos on huggingface.co are public, so this is only ever needed for
    # a company repository manager (Artifactory, Nexus) proxying them.
    # HF_TOKEN is the ecosystem name and works as a fallback, like
    # HF_ENDPOINT above.
    hf_token: str | None = Field(
        default=None,
        description="Bearer token for a HuggingFace endpoint that requires auth"
    )

    # PyTorch wheel index mirror (mainland China). pip has no index
    # priority, so a mirror added through pip.ini competes with the
    # download.pytorch.org entry baked into the env YAMLs and can lose.
    # Only we can take that entry out, so this replaces it.
    pytorch_index_url: str | None = Field(
        default=None,
        description=(
            "Base URL of a PyTorch wheel index mirror, without the CUDA "
            "suffix, e.g. https://mirror.nju.edu.cn/pytorch/whl"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_values(cls, data: dict) -> dict:
        """
        Treat blank strings as unset. An env var that is defined but
        empty (a half-filled GPO entry, `ADDAXAI_USER_DATA_DIR= app` in
        a shell script) would otherwise resolve paths against the
        working directory. Electron treats blank as unset too, so this
        keeps the two processes in agreement.
        """
        if isinstance(data, dict):
            return {
                k: v
                for k, v in data.items()
                if not (isinstance(v, str) and not v.strip())
            }
        return data

    @model_validator(mode="after")
    def _derive_from_user_data_dir(self) -> "Settings":
        """
        Derive database_url and models_dir from user_data_dir unless they
        were set explicitly (env var, .env file, or constructor argument).
        This is what makes ADDAXAI_USER_DATA_DIR alone relocate the
        whole app; without it the two defaults resolve to ~/AddaxAI
        regardless of the override.
        """
        # A relative path would silently resolve against the working
        # directory, which differs between Electron and the backend, so
        # the two processes would stop agreeing on where markers live.
        # Electron ignores non-absolute values; here we crash early for
        # anyone running the backend or CLI directly.
        if not self.user_data_dir.is_absolute():
            raise ValueError(
                f"ADDAXAI_USER_DATA_DIR must be an absolute path, "
                f"got {str(self.user_data_dir)!r}"
            )
        if "database_url" not in self.model_fields_set:
            self.database_url = f"sqlite:///{self.user_data_dir / 'addaxai.db'}"
        if "models_dir" not in self.model_fields_set:
            self.models_dir = self.user_data_dir / "models"
        if self.hf_endpoint is None:
            self.hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip() or None
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN", "").strip() or None
        return self

    @property
    def hf_base_url(self) -> str:
        """
        Base URL for every HuggingFace request: the mirror when one is
        configured, the real host otherwise. One rule, so the mirror
        cannot cover part of the traffic and quietly miss the rest.
        """
        return (self.hf_endpoint or "https://huggingface.co").rstrip("/")

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize settings and validate critical paths exist or can be created.

        Crashes immediately if required directories cannot be set up.
        """
        super().__init__(**kwargs)

        # Ensure user data directory exists
        if not self.user_data_dir.exists():
            try:
                self.user_data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create user data directory at {self.user_data_dir}: {e}"
                ) from e

        # Ensure models directory exists
        if not self.models_dir.exists():
            try:
                self.models_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create models directory at {self.models_dir}: {e}"
                ) from e


def get_settings() -> Settings:
    """
    Get application settings.

    Will crash if required environment variables are not set.
    This is intentional - we want to fail fast in development.
    """
    return Settings()
