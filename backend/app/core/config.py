"""
Application configuration.

Following DEVELOPERS.md principles:
- Sensible defaults for bundled mode (PyInstaller)
- Can be overridden via environment variables
- Crash early if configuration is invalid
- Type hints everywhere
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
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


def get_default_model_manifests_dir() -> Path:
    """Get default model manifests directory."""
    return get_default_models_dir() / "manifests"


def get_default_model_weights_dir() -> Path:
    """Get default model weights directory."""
    return get_default_models_dir() / "weights"


def get_default_model_environments_dir() -> Path:
    """Get default model environments directory."""
    return get_default_user_data_dir() / "environments"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Provides sensible defaults for bundled mode but can be overridden.
    Crashes if directories cannot be created.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",  # Crash if unknown env vars are provided
    )

    # Application
    app_name: str = "AddaxAI"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True

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

    # Models - all default to subdirectories of user_data_dir
    models_dir: Path = Field(default_factory=get_default_models_dir)
    model_manifests_dir: Path = Field(default_factory=get_default_model_manifests_dir)
    model_weights_dir: Path = Field(default_factory=get_default_model_weights_dir)
    model_environments_dir: Path = Field(default_factory=get_default_model_environments_dir)

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

        # Ensure models directories exist
        for directory in [
            self.models_dir,
            self.model_manifests_dir,
            self.model_weights_dir,
            self.model_environments_dir,
        ]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to create models directory at {directory}: {e}"
                    ) from e


def get_settings() -> Settings:
    """
    Get application settings.

    Will crash if required environment variables are not set.
    This is intentional - we want to fail fast in development.
    """
    return Settings()
