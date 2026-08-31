"""HTTP API exposed to the Pluvio Flutter app."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import schedules
from .cache import ForecastCache
from .config import Settings, get_settings
from . import history

LOG = logging.getLogger("pluvio.api")

# How often the server checks whether a new snapshot was published, to push it
# to connected clients. Cheap (a symlink stat); snapshots advance ~every 5 min.
_SNAPSHOT_POLL_SECONDS = 5.0


class _WSManager:
    """Tracks live WebSocket clients and broadcasts new-prediction notices."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self._clients):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 — a dead socket shouldn't block others
                self._clients.discard(ws)


def _snapshot_message(cache: ForecastCache) -> dict | None:
    """The 'new prediction' notice: snapshot id + issue time, so the client can
    refetch (one /v1/forecast + one sprite) instead of polling."""
    snap = cache.latest_snapshot()
    if snap is None:
        return None
    meta = cache.latest_metadata() or {}
    return {
        "type": "snapshot",
        "snapshot": snap.name,
        "issued_at": meta.get("issued_at"),
        "model_version": meta.get("model_version"),
    }


async def _watch_snapshots(cache: ForecastCache, manager: _WSManager) -> None:
    """Background task: poll the published snapshot and broadcast when it changes
    (the worker advances it ~every 5 min). Server-side only — clients never poll."""
    last: str | None = None
    while True:
        try:
            snap = cache.latest_snapshot()
            name = snap.name if snap else None
            if name and name != last:
                last = name
                msg = _snapshot_message(cache)
                if msg:
                    await manager.broadcast(msg)
                    LOG.info("ws: broadcast new snapshot %s to clients", name)
        except Exception:  # noqa: BLE001 — keep the watcher alive across hiccups
            LOG.exception("ws: snapshot watcher error")
        await asyncio.sleep(_SNAPSHOT_POLL_SECONDS)


class FrameDto(BaseModel):
    """One lead-time of the forecast at a specific location."""

    band: schedules.BandName
    lead_min: int
    valid_time: datetime
    rate_mm_per_h: float
    overlay_url: str
    # Provenance (rec #4): where this lead's number came from and how confident
    # we are. Null when the band is stub-served (no cube provenance).
    source: str | None = None
    confidence: float | None = None
    # Index of this frame's tile in the sprite sheet (see ForecastDto.sprite), so
    # the client renders it by cropping rather than fetching a per-frame PNG.
    sprite_index: int | None = None


class ForecastDto(BaseModel):
    issued_at: datetime
    location: dict[str, float]
    model_version: str
    horizon_min: int
    frames: list[FrameDto]
    # Per-band {source, confidence, producer}, so the client can honestly label
    # each horizon and widen its uncertainty band with lead.
    provenance: dict[str, dict] | None = None
    # Sprite sheet: one image with every frame tiled, so the client animates the
    # whole horizon with a single download. {url, tile_w, tile_h, cols, rows}.
    sprite: dict | None = None
    # Grid bounds [west, east, south, north] for placing the overlay/sprite.
    bounds: dict[str, float] | None = None


class HistoryFrameDto(BaseModel):
    """One observed radar frame at a specific location (history mode)."""

    minutes_ago: int  # 0 = newest observation, negative going back
    valid_time: datetime
    rate_mm_per_h: float
    overlay_url: str
    sprite_index: int | None = None


class HistoryDto(BaseModel):
    observed_at: datetime  # time of the newest frame — the mode's "now"
    location: dict[str, float]
    span_min: int
    frames: list[HistoryFrameDto]
    sprite: dict | None = None
    bounds: dict[str, float] | None = None


class HealthDto(BaseModel):
    status: str
    snapshot: str | None
    issued_at: datetime | None
    age_seconds: float | None
    model_version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    cache = ForecastCache(settings.cache_root)
    ws_manager = _WSManager()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        watcher = asyncio.create_task(_watch_snapshots(cache, ws_manager))
        try:
            yield
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    app = FastAPI(
        title="Pluvio Forecast API",
        version="0.1.0",
        description="Precipitation forecast cache for Belgium.",
        lifespan=lifespan,
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
        age = (datetime.now(UTC) - issued_dt).total_seconds() if issued_dt is not None else None
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

        provenance = meta.get("provenance") or {}
        sprite = meta.get("sprite") or {}
        sprite_index = sprite.get("index", {})
        frames: list[FrameDto] = []
        for _, row in point_df.iterrows():
            if int(row["lead_min"]) > horizon_min:
                continue
            band = row["band"]
            lead = int(row["lead_min"])
            prov = provenance.get(band, {})
            valid = issued_at.replace(microsecond=0) + _minutes(lead)
            frames.append(
                FrameDto(
                    band=band,
                    lead_min=lead,
                    valid_time=valid,
                    rate_mm_per_h=float(row["rate_mm_per_h"]),
                    overlay_url=f"/v1/overlay/{band}/{lead}.png?t={snap.name}",
                    source=prov.get("source"),
                    confidence=prov.get("confidence"),
                    sprite_index=sprite_index.get(f"{band}:{lead}"),
                )
            )

        sprite_dto = None
        if sprite:
            sprite_dto = {
                "url": f"/v1/sprite.png?t={snap.name}",
                "tile_w": sprite.get("tile_w"),
                "tile_h": sprite.get("tile_h"),
                "cols": sprite.get("cols"),
                "rows": sprite.get("rows"),
            }
        return ForecastDto(
            issued_at=issued_at,
            location={"lat": lat, "lon": lon},
            model_version=meta.get("model_version", settings.model_version),
            horizon_min=horizon_min,
            frames=frames,
            provenance=provenance or None,
            sprite=sprite_dto,
            bounds=meta.get("grid", {}).get("bounds"),
        )

    @app.websocket("/v1/ws")
    async def updates(websocket: WebSocket) -> None:
        """Push a notice whenever a new prediction is published, so the client
        refetches on demand instead of polling. On connect we send the current
        snapshot immediately; thereafter the server-side watcher broadcasts."""
        await ws_manager.connect(websocket)
        try:
            current = _snapshot_message(cache)
            if current:
                await websocket.send_json(current)
            # We don't expect client messages; receiving just detects the close.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001 — never let one client crash the route
            LOG.debug("ws: client error", exc_info=True)
        finally:
            ws_manager.disconnect(websocket)

    @app.get("/v1/sprite.png")
    def sprite() -> FileResponse:
        """The published snapshot's sprite sheet — one image with every frame
        tiled. The client downloads it once per prediction and crops tiles to
        animate, so scrubbing hits no network. Immutable per snapshot (the URL
        carries ?t=<snapshot>), so it caches for a long time."""
        path = cache.sprite_path()
        if path is None:
            raise HTTPException(status_code=404, detail="no sprite in cache")
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
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

    # ── Observed rainfall (history mode) ─────────────────────────────────
    # Mirrors the forecast serving shape so the client reuses its animation
    # pipeline with negative lead times. Backed by observed.npz from the
    # gauge-validated QPE chain (research/model/produce_observed.py).

    @app.get("/v1/history", response_model=HistoryDto)
    def history_frames(
        lat: Annotated[float, Query(ge=-90, le=90)],
        lon: Annotated[float, Query(ge=-180, le=180)],
        span_min: Annotated[int, Query(gt=0, le=360)] = 180,
    ) -> HistoryDto:
        data = history._load()
        if data is None:
            raise HTTPException(status_code=503, detail="no observed rainfall yet")
        try:
            pts = history.point_frames(lat, lon, span_min)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        info = history.sprite_info() or {}
        index = info.get("index", {})
        newest = int(data["times"][-1])
        frames = [
            HistoryFrameDto(
                minutes_ago=int(round((t - newest) / 60)),
                valid_time=datetime.fromtimestamp(t, tz=UTC),
                rate_mm_per_h=rate,
                overlay_url=f"/v1/history/overlay/{t}.png?t={info.get('mtime', 0)}",
                sprite_index=index.get(t),
            )
            for t, rate in pts
        ]
        sprite_dto = None
        if info:
            sprite_dto = {
                "url": f"/v1/history/sprite.png?t={info['mtime']}",
                "tile_w": info["tile_w"],
                "tile_h": info["tile_h"],
                "cols": info["cols"],
                "rows": info["rows"],
            }
        return HistoryDto(
            observed_at=datetime.fromtimestamp(newest, tz=UTC),
            location={"lat": lat, "lon": lon},
            span_min=span_min,
            frames=frames,
            sprite=sprite_dto,
            bounds=data["bounds"],
        )

    @app.get("/v1/history/tiles")
    def history_tiles() -> dict:
        info = history.tiles_info()
        if info is None:
            raise HTTPException(status_code=404, detail="no hi-res observed cube")
        return info

    @app.get("/v1/history/tile/{tx}/{ty}/sprite.png")
    def history_tile_sprite(tx: int, ty: int) -> FileResponse:
        path = history.tile_sprite_png_path(tx, ty)
        if path is None:
            raise HTTPException(status_code=404, detail="tile unavailable")
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=300"})

    @app.get("/v1/history/sprite.png")
    def history_sprite() -> FileResponse:
        path = history.sprite_png_path()
        if path is None:
            raise HTTPException(status_code=404, detail="no observed sprite")
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.get("/v1/history/overlay/{epoch}.png")
    def history_overlay(epoch: int) -> Response:
        png = history.overlay_png(epoch)
        if png is None:
            raise HTTPException(status_code=404, detail="frame not in window")
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
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
