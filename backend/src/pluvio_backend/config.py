"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import pathlib
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for the backend's tunables."""

    model_config = SettingsConfigDict(
        env_prefix="PLUVIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    cache_root: pathlib.Path = Field(default=pathlib.Path("./var/forecasts"))
    cache_stale_after_seconds: int = Field(default=900)

    # Upstream
    kmi_base_url: str = Field(default="https://app.meteo.be/services/appv4/")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="")

    # Model
    model_version: str = Field(default="stub-0.1")

    @property
    def cors_origin_list(self) -> list[str]:
        if not self.cors_origins.strip():
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings cache. Tests can call `get_settings.cache_clear()`."""
    return Settings()
