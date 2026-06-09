"""Build the static channels (elevation / landmask / distance-to-coast) on
the same 100×100 analysis grid the model uses for the time-varying channels.

One-off — the output `static.npz` is checked into the data dir and reused
for every training run. Re-run only if the analysis grid changes.

Sources (no key, FOSS):
- **OpenTopoData REST API** for elevation. Hits the public ETOPO1 dataset.
  Rate-limited to ~1 req/sec; we batch 100 grid cells per request so the
  full 100×100 grid is 100 calls, ~2 min wall time.
- Landmask = elevation > 0 (good enough for Benelux — the Polders and the
  Westerschelde are sub-cell scale on this grid).
- Distance-to-coast = scipy distance transform on the landmask, then
  scaled by the grid cell pitch in km.

Output (`research/data/static.npz`):

    elevation_m         (100, 100) float32   metres above sea level
    landmask            (100, 100) float32   0/1
    distance_to_coast_km(100, 100) float32   km from each cell to the nearest
                                              non-land cell (0 over sea)

These are saved un-normalised; the dataset loader is the right place to
centre/scale into ~O(1) for the model.

Run with:

    python -m model.build_static --out data/static.npz
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import httpx
import numpy as np

from . import geo

LOG = logging.getLogger("pluvio.build_static")

# OpenTopoData public mirror — supports several datasets via /v1/<name>.
# etopo1 is global at 1 arc-min ≈ 1.8 km, plenty for our 100×100 grid.
TOPO_URL = "https://api.opentopodata.org/v1/etopo1"
BATCH = 100   # API caps each request at 100 locations.
COOLDOWN = 1.0  # public API rate-limits to ~1 req/sec.


def fetch_elevation(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return an (H, W) float32 elevation array in metres."""
    h, w = lat.shape
    flat_lat = lat.reshape(-1)
    flat_lon = lon.reshape(-1)
    out = np.zeros(flat_lat.size, dtype="float32")

    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        for start in range(0, flat_lat.size, BATCH):
            end = min(start + BATCH, flat_lat.size)
            locs = "|".join(
                f"{flat_lat[i]:.5f},{flat_lon[i]:.5f}" for i in range(start, end)
            )
            r = client.get(TOPO_URL, params={"locations": locs})
            if r.status_code == 429:
                LOG.warning("rate-limited; waiting 5s")
                time.sleep(5)
                r = client.get(TOPO_URL, params={"locations": locs})
            r.raise_for_status()
            payload = r.json()
            for i, result in enumerate(payload["results"]):
                e = result.get("elevation")
                out[start + i] = float(e) if e is not None else 0.0
            LOG.info("  %d/%d points", end, flat_lat.size)
            time.sleep(COOLDOWN)

    return out.reshape(h, w)


def grid_pitch_km(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Approximate cell pitch in km. Used to convert pixel distances to km."""
    # Mean cell pitch in the row (north-south) and column (east-west) direction.
    # Both are nearly uniform on the polar-stereographic grid for our latitudes.
    R = 6371.0  # mean Earth radius in km
    # Latitude pitch: difference between adjacent rows, in degrees, * km/deg.
    d_lat = float(np.mean(np.abs(np.diff(lat, axis=0))))
    pitch_y = d_lat * (np.pi / 180.0) * R
    # Longitude pitch: depends on cos(lat).
    mean_lat = float(np.mean(lat))
    d_lon = float(np.mean(np.abs(np.diff(lon, axis=1))))
    pitch_x = d_lon * (np.pi / 180.0) * R * np.cos(np.radians(mean_lat))
    return pitch_x, pitch_y


def distance_to_coast_km(landmask: np.ndarray, pitch_x: float, pitch_y: float) -> np.ndarray:
    """For each cell, distance in km to the nearest non-land cell.

    Uses scipy's exact Euclidean distance transform; the input mask treats
    sea (False) as the background, so distances over the sea are 0.
    """
    from scipy.ndimage import distance_transform_edt

    # sampling = pitch per axis, so the transform result is in km, not pixels.
    sea = ~landmask.astype(bool)
    # distance_transform_edt(mask) returns distance from each True cell to the
    # nearest False cell — so we pass landmask=True and get land-→-coast.
    dist = distance_transform_edt(~sea, sampling=(pitch_y, pitch_x))
    return dist.astype("float32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/static.npz"))
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch elevation even if --out exists.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.out.exists() and not args.force:
        LOG.info("%s exists — pass --force to rebuild", args.out)
        return 0

    lat, lon = geo.grid_latlon()
    LOG.info("grid %s, bbox=%s", lat.shape, geo.bbox())

    LOG.info("fetching elevation from OpenTopoData (etopo1)…")
    elev = fetch_elevation(lat, lon)
    LOG.info("elevation range: [%.1f, %.1f] m", elev.min(), elev.max())

    landmask = (elev > 0).astype("float32")
    LOG.info("landmask: %.1f%% land", 100.0 * landmask.mean())

    pitch_x, pitch_y = grid_pitch_km(lat, lon)
    LOG.info("grid pitch: %.2f × %.2f km", pitch_x, pitch_y)

    dist = distance_to_coast_km(landmask.astype(bool), pitch_x, pitch_y)
    LOG.info(
        "distance to coast: min=%.1f, mean=%.1f, max=%.1f km",
        dist[landmask > 0].min() if landmask.any() else 0,
        dist[landmask > 0].mean() if landmask.any() else 0,
        dist.max(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        elevation_m=elev,
        landmask=landmask,
        distance_to_coast_km=dist,
        grid_lat=lat,
        grid_lon=lon,
        grid_pitch_km=np.array([pitch_x, pitch_y], dtype="float32"),
    )
    LOG.info("wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
