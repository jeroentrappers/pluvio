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
    # 45 min: the forecast's issue time is the OPERA analysis time, which lags
    # wall-clock by radar cadence + collection + producer latency. Observed
    # worst case is ~40 min (a 15:00Z frame isn't on the box until ~15:20Z, and
    # the producer only refreshes every 10 min), so a tighter threshold flaps
    # "degraded" on fresh-but-normal data. Above this means collection is broken.
    cache_stale_after_seconds: int = Field(default=2700)

    # Upstream
    kmi_base_url: str = Field(default="https://app.meteo.be/services/appv4/")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="")

    # Model
    model_version: str = Field(default="stub-0.1")
    # Public scoreboard (tools/scoreboard.py writes <dir>/index.html and
    # <dir>/YYYY/MM/DD.json on the storage box; the container sees it here).
    scoreboard_dir: pathlib.Path = Field(default=pathlib.Path("/storagebox/scoreboard"))

    @property
    def cors_origin_list(self) -> list[str]:
        if not self.cors_origins.strip():
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings cache. Tests can call `get_settings.cache_clear()`."""
    return Settings()
