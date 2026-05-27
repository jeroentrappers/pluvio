"""HTTP API exposed to the Pluvio Flutter app."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import schedules
from .cache import ForecastCache
from .config import Settings, get_settings

LOG = logging.getLogger("pluvio.api")


class FrameDto(BaseModel):
    """One lead-time of the forecast at a specific location."""

    band: schedules.BandName
    lead_min: int
    valid_time: datetime
    rate_mm_per_h: float
    overlay_url: str


class ForecastDto(BaseModel):
    issued_at: datetime
    location: dict[str, float]
    model_version: str
    horizon_min: int
    frames: list[FrameDto]


class HealthDto(BaseModel):
    status: str
    snapshot: str | None
    issued_at: datetime | None
    age_seconds: float | None
    model_version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    cache = ForecastCache(settings.cache_root)

    app = FastAPI(
        title="Pluvio Forecast API",
        version="0.1.0",
        description="Precipitation forecast cache for Belgium.",
    )

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    @app.get("/healthz", response_model=HealthDto)
    def healthz() -> HealthDto:
        snap = cache.latest_snapshot()
        if snap is None:
            return HealthDto(
                status="empty",
                snapshot=None,
                issued_at=None,
                age_seconds=None,
                model_version=settings.model_version,
            )
        meta = cache.latest_metadata() or {}
        issued = meta.get("issued_at")
        try:
            issued_dt = datetime.fromisoformat(issued.replace("Z", "+00:00")) if issued else None
        except (AttributeError, ValueError):
            issued_dt = None
        age = (
            (datetime.now(UTC) - issued_dt).total_seconds()
            if issued_dt is not None
            else None
        )
        degraded = age is not None and age > settings.cache_stale_after_seconds
        return HealthDto(
            status="degraded" if degraded else "ok",
            snapshot=snap.name,
            issued_at=issued_dt,
            age_seconds=age,
            model_version=meta.get("model_version", settings.model_version),
        )

    @app.get("/v1/forecast", response_model=ForecastDto)
    def forecast(
        lat: Annotated[float, Query(ge=-90, le=90)],
        lon: Annotated[float, Query(ge=-180, le=180)],
        horizon_min: Annotated[int, Query(gt=0, le=14400)] = 24 * 60,
    ) -> ForecastDto:
        snap = cache.latest_snapshot()
        if snap is None:
            raise HTTPException(status_code=503, detail="cache is empty; worker hasn't run yet")
        meta = cache.latest_metadata() or {}
        issued_at_raw = meta.get("issued_at")
        issued_at = (
            datetime.fromisoformat(issued_at_raw.replace("Z", "+00:00"))
            if isinstance(issued_at_raw, str)
            else datetime.now(UTC)
        )

        # Validate the location is inside the served grid before looking up
        # a shard — `latlon_to_cell` raises ValueError when out of bounds,
        # which we surface as a 400 (vs. a 503 for "cache not ready").
        try:
            cache.grid.latlon_to_cell(lat, lon)
            point_df = cache.read_point(lat, lon)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if point_df is None or point_df.empty:
            raise HTTPException(
                status_code=503,
                detail="no point shard for the requested location yet",
            )

        frames: list[FrameDto] = []
        for _, row in point_df.iterrows():
            if int(row["lead_min"]) > horizon_min:
                continue
            valid = issued_at.replace(microsecond=0) + _minutes(int(row["lead_min"]))
            frames.append(
                FrameDto(
                    band=row["band"],
                    lead_min=int(row["lead_min"]),
                    valid_time=valid,
                    rate_mm_per_h=float(row["rate_mm_per_h"]),
                    overlay_url=f"/v1/overlay/{row['band']}/{int(row['lead_min'])}.png?t={snap.name}",
                )
            )

        return ForecastDto(
            issued_at=issued_at,
            location={"lat": lat, "lon": lon},
            model_version=meta.get("model_version", settings.model_version),
            horizon_min=horizon_min,
            frames=frames,
        )

    @app.get("/v1/overlay/{band}/{lead_min}.png")
    def overlay(band: schedules.BandName, lead_min: int) -> FileResponse:
        if band not in schedules.BANDS:
            raise HTTPException(status_code=404, detail=f"unknown band {band!r}")
        path = cache.overlay_url_path(band, lead_min)
        if path is None:
            raise HTTPException(status_code=404, detail="overlay not in cache")
        return FileResponse(
            path,
            media_type="image/png",
            headers={
                "Cache-Control": f"public, max-age={schedules.band(band).refresh_seconds - 10}"
            },
        )

    @app.get("/v1/animation/manifest.json")
    def animation_manifest(
        band: schedules.BandName = "nowcast",
    ) -> Response:
        snap = cache.latest_snapshot()
        if snap is None:
            raise HTTPException(status_code=503, detail="cache is empty")
        meta = cache.latest_metadata() or {}
        b = schedules.band(band)
        issued_at_raw = meta.get("issued_at", datetime.now(UTC).isoformat())
        try:
            issued_at = datetime.fromisoformat(issued_at_raw.replace("Z", "+00:00"))
        except ValueError:
            issued_at = datetime.now(UTC)

        frames = []
        for lead in b.leads_min:
            path = cache.overlay_url_path(band, lead)
            if path is None:
                continue
            valid = (issued_at + _minutes(lead)).isoformat().replace("+00:00", "Z")
            frames.append(
                {
                    "lead_min": lead,
                    "valid_time": valid,
                    "url": f"/v1/overlay/{band}/{lead}.png?t={snap.name}",
                }
            )
        body = {
            "snapshot": snap.name,
            "band": band,
            "bounds": meta.get("grid", {}).get("bounds"),
            "frames": frames,
            "model_version": meta.get("model_version", settings.model_version),
        }
        return Response(
            content=__import__("json").dumps(body),
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=60"},
        )

    return app


def _minutes(n: int):
    from datetime import timedelta
    return timedelta(minutes=n)


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Pluvio API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    uvicorn.run(
        "pluvio_backend.api:create_app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        factory=True,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
