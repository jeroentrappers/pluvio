"""Stub inference path: turn KMI's `getForecasts` into our cache shape.

This is what the inference worker calls before the trained CorrDiff model
exists. The API is identical to what the real model will eventually use,
so swapping in the model is a one-function change.

Strategy:
- Nowcast band (0–120 min, 10-min steps): hit KMI `getForecasts` once per
  centroid in a small set of representative locations, then build a 2-D
  field by inverse-distance-weighted interpolation onto the grid.
- Short/medium/long bands: pull `for.hourly` (49 entries, 1-h resolution),
  spread the point value uniformly across the grid for now. The eventual
  model produces real fields; the stub just gives the API something
  realistic to serve while we wait.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np

from . import schedules
from .cache import GridSpec
from .kmi_signing import sign

LOG = logging.getLogger("pluvio.stubs")
USER_AGENT = "pluvio-backend/0.1 (+https://github.com/appmire/pluvio)"

# Sentinel locations the stub queries to build the field via IDW. Enough
# to cover Belgium with sensible diversity.
STUB_QUERY_POINTS: dict[str, tuple[float, float]] = {
    "brussels": (50.8503, 4.3517),
    "antwerp": (51.2194, 4.4025),
    "ghent": (51.0543, 3.7174),
    "liege": (50.6326, 5.5797),
    "charleroi": (50.4108, 4.4446),
    "bruges": (51.2093, 3.2247),
    "arlon": (49.6839, 5.8167),
    "ostend": (51.2247, 2.9069),
}


def _km_5min_to_mm_per_h(v: float) -> float:
    """KMI's `animation.sequence[].value` is mm/10min → multiply by 6 for mm/h."""
    return float(v) * 6.0


def fetch_getforecasts(
    client: httpx.Client,
    base_url: str,
    lat: float,
    lon: float,
) -> dict:
    params = {
        "s": "getForecasts",
        "k": sign("getForecasts"),
        "lat": round(lat, 6),
        "long": round(lon, 6),
    }
    r = client.get(base_url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def _idw_field(
    values_by_point: dict[str, tuple[float, float, float]],
    grid: GridSpec,
    power: float = 2.0,
) -> np.ndarray:
    """Inverse-distance weighting onto the grid.

    ``values_by_point`` is ``{name: (lat, lon, mm_per_h)}``.
    """
    h, w = grid.shape
    west, east = grid.bounds["west"], grid.bounds["east"]
    south, north = grid.bounds["south"], grid.bounds["north"]
    cols = np.linspace(west, east, w)
    rows = np.linspace(north, south, h)

    sample_lats = np.array([v[0] for v in values_by_point.values()], dtype="float32")
    sample_lons = np.array([v[1] for v in values_by_point.values()], dtype="float32")
    sample_vals = np.array([v[2] for v in values_by_point.values()], dtype="float32")

    grid_lat, grid_lon = np.meshgrid(rows, cols, indexing="ij")
    # Pairwise (lat,lon) → squared great-circle approximation (flat-earth
    # is fine over Belgium); add a small ε so points exactly on a sample
    # don't divide by zero.
    dy = grid_lat[..., None] - sample_lats
    dx = grid_lon[..., None] - sample_lons
    d2 = dy * dy + dx * dx + 1e-9
    weights = 1.0 / np.power(d2, power / 2.0)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights * sample_vals).sum(axis=-1).astype("float32")


def stub_nowcast(
    client: httpx.Client,
    base_url: str,
    grid: GridSpec,
) -> tuple[np.ndarray, datetime]:
    """Return a (n_leads, H, W) array for the nowcast band + issue time."""
    band = schedules.band("nowcast")
    leads = band.leads_min
    responses: dict[str, dict] = {}
    for name, (lat, lon) in STUB_QUERY_POINTS.items():
        try:
            responses[name] = fetch_getforecasts(client, base_url, lat, lon)
        except httpx.HTTPError as exc:
            LOG.warning("nowcast stub: skip %s (%s)", name, exc)

    if not responses:
        raise RuntimeError("nowcast stub: every upstream call failed")

    issued_at = datetime.now(UTC).replace(microsecond=0)
    frames = np.zeros((band.n_leads, *grid.shape), dtype="float32")
    for k, lead in enumerate(leads):
        values_for_grid: dict[str, tuple[float, float, float]] = {}
        for name, resp in responses.items():
            seq = resp.get("animation", {}).get("sequence") or []
            issue_time = (
                datetime.fromisoformat(seq[0]["time"]).astimezone(UTC)
                if seq
                else issued_at
            )
            target = issue_time + timedelta(minutes=lead)
            nearest = min(
                seq,
                key=lambda f: abs(datetime.fromisoformat(f["time"]).astimezone(UTC) - target),
                default=None,
            )
            if nearest is None:
                continue
            lat, lon = STUB_QUERY_POINTS[name]
            values_for_grid[name] = (lat, lon, _km_5min_to_mm_per_h(nearest.get("value", 0.0)))
        if values_for_grid:
            frames[k] = _idw_field(values_for_grid, grid)
    return frames, issued_at


def stub_band(
    client: httpx.Client,
    base_url: str,
    grid: GridSpec,
    band_name: schedules.BandName,
) -> tuple[np.ndarray, datetime]:
    """Return a (n_leads, H, W) array for any non-nowcast band.

    Uses `for.hourly` (49 entries) keyed off Brussels — good enough for the
    stub. The model that replaces this writes real per-pixel fields.
    """
    if band_name == "nowcast":
        return stub_nowcast(client, base_url, grid)
    band = schedules.band(band_name)

    lat, lon = STUB_QUERY_POINTS["brussels"]
    resp = fetch_getforecasts(client, base_url, lat, lon)
    hourly = resp.get("for", {}).get("hourly") or []

    issued_at = datetime.now(UTC).replace(microsecond=0)
    frames = np.zeros((band.n_leads, *grid.shape), dtype="float32")
    if not hourly:
        return frames, issued_at

    # Each hourly entry has 'hour' as wall-clock hour-of-day in local time;
    # we treat them as a sequential 1-h series starting from issue time.
    for k, lead in enumerate(band.leads_min):
        idx = min(int(lead // 60), len(hourly) - 1)
        precip = hourly[idx].get("precipQuantity")
        try:
            value = float(precip) if precip is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        frames[k] = value  # broadcast scalar — flat field
    return frames, issued_at
