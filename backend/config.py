"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment / .env with sane defaults."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="FT_", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8000
    db_path: str = "finetune_tracker.db"
    refresh_interval: int = 10
    stale_minutes: int = 30
    api_url: str = "http://127.0.0.1:8000"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
