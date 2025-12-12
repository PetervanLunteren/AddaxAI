"""
Application configuration.

Following DEVELOPERS.md principles:
- No defaults - all required config must be explicitly set
- Crash early if configuration is missing
- Type hints everywhere
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    No defaults for critical paths - will crash if not provided.
    This follows the "crash early and loudly" principle.
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

    # Database - NO DEFAULT PATH, must be explicitly set
    database_url: str

    # User data directory - NO DEFAULT, must be explicitly set
    user_data_dir: Path

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379

    # Models
    models_dir: Path
    model_manifests_dir: Path
    model_weights_dir: Path
    model_environments_dir: Path

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
