"""Forecast cache layout + atomic refresh.

Each refresh writes a *new* directory and, when complete, atomically swaps
the ``latest`` symlink to point at it. Readers always see a consistent
snapshot — never half-written zarr or PNG files.

Directory layout (one per refresh; keep last N for animation history):

    <root>/
        2026-05-26T12-05-00Z/
            grid.json                 ← grid metadata (bounds, shape, model_version)
            bands/
                nowcast.zarr/         ← (n_leads, H, W) mm/h
                short.zarr/
                medium.zarr/
                long.zarr/
            overlays/
                nowcast/
                    0.png             ← 0-minute lead overlay
                    10.png
                    ...
                short/
                    120.png
                    ...
            points/
                bbox_lat=508_lon=43.parquet   ← coarse-bucket point indices
        latest -> 2026-05-26T12-05-00Z

Atomicity:
- A refresh writes everything under a sibling timestamp dir.
- A small ``status.json`` is the *last* file written.
- `swap_latest` only swaps the symlink if ``status.json`` exists.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import zarr

from . import schedules

LOG = logging.getLogger("pluvio.cache")

# Geographic bounds match the Flutter app's `Env.radarBounds*`.
DEFAULT_BOUNDS: dict[str, float] = {"west": 1.5, "east": 7.5, "south": 48.9, "north": 52.5}
DEFAULT_GRID_SHAPE: tuple[int, int] = (100, 100)


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """Static description of the spatial grid every cache uses."""

    bounds: dict[str, float]  # keys: west, east, south, north (degrees)
    shape: tuple[int, int]  # (height, width)

    def to_dict(self) -> dict:
        return {"bounds": dict(self.bounds), "shape": list(self.shape)}

    def latlon_to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert (lat, lon) → (row, col) on the grid, clamped to bounds."""
        h, w = self.shape
        if lon < self.bounds["west"] or lon > self.bounds["east"]:
            raise ValueError(f"lon={lon} outside [{self.bounds['west']}, {self.bounds['east']}]")
        if lat < self.bounds["south"] or lat > self.bounds["north"]:
            raise ValueError(f"lat={lat} outside [{self.bounds['south']}, {self.bounds['north']}]")
        # row 0 = north, row h-1 = south.
        col = int(
            (lon - self.bounds["west"]) / (self.bounds["east"] - self.bounds["west"]) * (w - 1)
        )
        row = int(
            (self.bounds["north"] - lat) / (self.bounds["north"] - self.bounds["south"]) * (h - 1)
        )
        return row, col


DEFAULT_GRID = GridSpec(bounds=DEFAULT_BOUNDS, shape=DEFAULT_GRID_SHAPE)


class ForecastCache:
    """One-stop API to read/write forecast snapshots."""

    def __init__(self, root: pathlib.Path, grid: GridSpec = DEFAULT_GRID):
        self.root = pathlib.Path(root)
        self.grid = grid
        self.root.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────── writer side

    def new_snapshot_dir(self, issued_at: datetime | None = None) -> pathlib.Path:
        ts = (issued_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H-%M-%SZ")
        d = self.root / ts
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_grid_metadata(
        self, snapshot_dir: pathlib.Path, model_version: str, extras: dict | None = None
    ) -> None:
        body = {
            "grid": self.grid.to_dict(),
            "model_version": model_version,
            "issued_at": self._stamp_from_dir(snapshot_dir),
            **(extras or {}),
        }
        (snapshot_dir / "grid.json").write_text(json.dumps(body, indent=2), encoding="utf-8")

    def write_band(
        self,
        snapshot_dir: pathlib.Path,
        band_name: schedules.BandName,
        rates_mm_per_h: np.ndarray,
    ) -> pathlib.Path:
        """Persist a (n_leads, H, W) tensor as zarr."""
        band = schedules.band(band_name)
        expected = (band.n_leads, *self.grid.shape)
        if rates_mm_per_h.shape != expected:
            raise ValueError(
                f"band {band_name}: expected shape {expected}, got {rates_mm_per_h.shape}"
            )
        path = snapshot_dir / "bands" / f"{band_name}.zarr"
        path.parent.mkdir(parents=True, exist_ok=True)
        z = zarr.open_array(
            store=str(path),
            mode="w",
            shape=expected,
            chunks=(1, *self.grid.shape),
            dtype="float32",
        )
        z[:] = rates_mm_per_h.astype("float32", copy=False)
        z.attrs["leads_min"] = band.leads_min
        z.attrs["band"] = band_name
        return path

    def _render_shape(self) -> tuple[int, int]:
        """Overlay render resolution: the radar composite's angular pixel
        density (wide serving grid: 1907 px / 41 deg lon, 2627 px / 35.5 deg
        lat) applied to this grid's bounds — so forecast frames match the
        measured composite's scale exactly when the timeline crosses t=0.
        Override with PLUVIO_OVERLAY_PXDEG="lon,lat"; values <= native
        resolution disable upscaling."""
        import os

        try:
            px_lon, px_lat = (float(x) for x in
                              os.environ.get("PLUVIO_OVERLAY_PXDEG", "46.51,74.0").split(","))
        except ValueError:
            px_lon, px_lat = 46.51, 74.0
        b = self.grid.bounds
        th = round((b["north"] - b["south"]) * px_lat)
        tw = round((b["east"] - b["west"]) * px_lon)
        return max(th, self.grid.shape[0]), max(tw, self.grid.shape[1])

    def write_overlays(
        self,
        snapshot_dir: pathlib.Path,
        band_name: schedules.BandName,
        rates_mm_per_h: np.ndarray,
    ) -> int:
        """Render one PNG per lead step. Returns the number of files written."""
        from .tiler import render_overlay_to_path

        band = schedules.band(band_name)
        shape = self._render_shape()
        n_written = 0
        for i, lead in enumerate(band.leads_min):
            target = snapshot_dir / "overlays" / band_name / f"{lead}.png"
            render_overlay_to_path(rates_mm_per_h[i], target, target_hw=shape)
            n_written += 1
        return n_written

    def write_point_shards(
        self,
        snapshot_dir: pathlib.Path,
        all_bands: dict[schedules.BandName, np.ndarray],
        bucket_step: float = 0.1,
    ) -> int:
        """Write per-bucket Parquet shards for fast point lookups.

        Bucket key: round(lat / bucket_step), round(lon / bucket_step).
        Each shard holds, for every cell within that bucket and every
        (band, lead_min), the precipitation rate.
        """
        h, w = self.grid.shape
        west, east = self.grid.bounds["west"], self.grid.bounds["east"]
        south, north = self.grid.bounds["south"], self.grid.bounds["north"]
        cols = np.linspace(west, east, w)
        rows = np.linspace(north, south, h)  # row 0 = north

        records: list[dict] = []
        for band_name, arr in all_bands.items():
            band = schedules.band(band_name)
            for i, lead in enumerate(band.leads_min):
                grid = arr[i]
                for r in range(h):
                    for c in range(w):
                        records.append(
                            {
                                "lat": float(rows[r]),
                                "lon": float(cols[c]),
                                "band": band_name,
                                "lead_min": lead,
                                "rate_mm_per_h": float(grid[r, c]),
                            }
                        )

        df = pd.DataFrame(records)
        df["lat_bucket"] = (df["lat"] / bucket_step).round().astype(int)
        df["lon_bucket"] = (df["lon"] / bucket_step).round().astype(int)
        n_written = 0
        points_dir = snapshot_dir / "points"
        points_dir.mkdir(parents=True, exist_ok=True)
        for (lat_b, lon_b), sub in df.groupby(["lat_bucket", "lon_bucket"]):
            target = points_dir / f"bbox_lat={lat_b}_lon={lon_b}.parquet"
            sub.drop(columns=["lat_bucket", "lon_bucket"]).to_parquet(
                target, compression="zstd", index=False
            )
            n_written += 1
        return n_written

    def write_sprite(
        self,
        snapshot_dir: pathlib.Path,
        all_bands: dict[schedules.BandName, np.ndarray],
        cols: int = 12,
    ) -> dict:
        """Compose one sprite-sheet PNG of every (band, lead) frame and return
        its layout. The client downloads this single image and scrubs by tile,
        so the whole animation is one request instead of one-per-frame.

        Tiles are ordered by lead-time across all bands (the same order the
        client sorts frames), and ``index`` maps "band:lead" → tile number.
        """
        from .tiler import render_sprite

        ordered: list[tuple[int, str, np.ndarray]] = []
        for band_name, arr in all_bands.items():
            band = schedules.band(band_name)
            for i, lead in enumerate(band.leads_min):
                ordered.append((lead, band_name, arr[i]))
        ordered.sort(key=lambda t: t[0])

        shape = self._render_shape()
        png, rows, cols = render_sprite([t[2] for t in ordered], cols=cols, target_hw=shape)
        (snapshot_dir / "sprite.png").write_bytes(png)

        h, w = shape
        return {
            "tile_w": w,
            "tile_h": h,
            "cols": cols,
            "rows": rows,
            "count": len(ordered),
            "index": {f"{band}:{lead}": i for i, (lead, band, _) in enumerate(ordered)},
        }

    def sprite_path(self) -> pathlib.Path | None:
        """The published snapshot's sprite, if present."""
        snap = self.latest_snapshot()
        if snap is None:
            return None
        p = snap / "sprite.png"
        return p if p.exists() else None

    def mark_complete(self, snapshot_dir: pathlib.Path, summary: dict | None = None) -> None:
        """Drop a `status.json` flag — must be the last file we write."""
        body = {
            "completed_at": datetime.now(UTC).isoformat(),
            "ok": True,
            **(summary or {}),
        }
        (snapshot_dir / "status.json").write_text(json.dumps(body, indent=2), encoding="utf-8")

    def swap_latest(self, snapshot_dir: pathlib.Path) -> None:
        """Atomically point ``latest`` at ``snapshot_dir``.

        Refuses to swap unless ``status.json`` exists, so a half-written
        snapshot can never become visible.
        """
        if not (snapshot_dir / "status.json").exists():
            raise RuntimeError(f"snapshot {snapshot_dir} is missing status.json; refusing to swap")
        link = self.root / "latest"
        tmp = self.root / "latest.tmp"
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(snapshot_dir.name, tmp)
        os.replace(tmp, link)
        LOG.info("swapped %s → %s", link, snapshot_dir.name)

    def prune(self, keep: int = 24) -> int:
        """Delete all but the `keep` most-recent complete snapshots.

        Returns the number of directories removed.
        """
        import shutil

        snapshots = sorted(
            (
                p
                for p in self.root.iterdir()
                if p.is_dir() and (p / "status.json").exists() and p.name != "latest"
            ),
            key=lambda p: p.name,
            reverse=True,
        )
        removed = 0
        for stale in snapshots[keep:]:
            shutil.rmtree(stale)
            removed += 1
        return removed

    # ──────────────────────────────────────────────────────────── reader side

    def latest_snapshot(self) -> pathlib.Path | None:
        link = self.root / "latest"
        if not link.exists():
            return None
        target = link.resolve()
        if not target.exists():
            return None
        return target

    def latest_metadata(self) -> dict | None:
        snap = self.latest_snapshot()
        if snap is None:
            return None
        meta = snap / "grid.json"
        if not meta.exists():
            return None
        return json.loads(meta.read_text(encoding="utf-8"))

    def read_band(self, band_name: schedules.BandName) -> np.ndarray | None:
        snap = self.latest_snapshot()
        if snap is None:
            return None
        path = snap / "bands" / f"{band_name}.zarr"
        if not path.exists():
            return None
        return np.asarray(zarr.open_array(store=str(path), mode="r")[:])

    def complete_snapshots(self) -> list[pathlib.Path]:
        """All complete snapshot dirs, newest first."""
        return sorted(
            (
                p
                for p in self.root.iterdir()
                if p.is_dir() and p.name != "latest" and (p / "status.json").exists()
            ),
            key=lambda p: p.name,
            reverse=True,
        )

    def read_band_any(self, band_name: schedules.BandName) -> np.ndarray | None:
        """Freshest array for `band_name` across all complete snapshots.

        Bands refresh on different cadences, so the long-range outlook may live
        in an older snapshot than the latest nowcast. This lets the worker fold
        every band's freshest data into the snapshot it's about to publish.
        """
        expected = (schedules.band(band_name).n_leads, *self.grid.shape)
        for snap in self.complete_snapshots():
            path = snap / "bands" / f"{band_name}.zarr"
            if not path.exists():
                continue
            arr = np.asarray(zarr.open_array(store=str(path), mode="r")[:])
            # Skip arrays written under an older band definition (different lead
            # count) — folding them in would mislabel leads. They age out via
            # prune as fresh ticks rewrite the band.
            if arr.shape == expected:
                return arr
        return None

    def read_point(self, lat: float, lon: float, bucket_step: float = 0.1) -> pd.DataFrame | None:
        snap = self.latest_snapshot()
        if snap is None:
            return None
        lat_b = round(lat / bucket_step)
        lon_b = round(lon / bucket_step)
        path = snap / "points" / f"bbox_lat={lat_b}_lon={lon_b}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        # Pick the row whose (lat, lon) is closest to the request.
        d2 = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
        nearest_lat = df.loc[d2.idxmin(), "lat"]
        nearest_lon = df.loc[d2.idxmin(), "lon"]
        out = df[(df["lat"] == nearest_lat) & (df["lon"] == nearest_lon)].copy()
        return out.sort_values(["band", "lead_min"]).reset_index(drop=True)

    def overlay_url_path(self, band_name: schedules.BandName, lead_min: int) -> pathlib.Path | None:
        # Prefer the published snapshot, but fall back to the most recent
        # snapshot that has this overlay — longer bands refresh on their own
        # cadence and their PNGs may live in an older snapshot than the latest
        # nowcast one (the worker folds their data forward, not their files).
        snap = self.latest_snapshot()
        if snap is not None:
            path = snap / "overlays" / band_name / f"{lead_min}.png"
            if path.exists():
                return path
        for older in self.complete_snapshots():
            path = older / "overlays" / band_name / f"{lead_min}.png"
            if path.exists():
                return path
        return None

    # ─────────────────────────────────────────────────────────────── helpers

    @staticmethod
    def _stamp_from_dir(snapshot_dir: pathlib.Path) -> str:
        # "2026-05-26T12-05-00Z" → "2026-05-26T12:05:00Z"
        # The date half keeps its dashes; only the time half (after T) gets
        # its dashes turned into colons.
        date_part, _, time_part = snapshot_dir.name.partition("T")
        return f"{date_part}T{time_part.replace('-', ':')}"
