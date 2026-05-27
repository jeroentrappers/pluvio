"""Build aligned auxiliary-channel grids for the training window.

Produces ``data/knmi/aux_cache.npz`` with arrays aligned 1:1 to the forecast
timestamps in ``frames_cache.npz`` (same order), so the dataset can index
aux channels with the same ``issue_idx`` it uses for radar.

v1 sources:
  - KMI AWS 10-min → IDW onto the grid: pressure, temperature, humidity,
    wind speed. Each forecast time T uses the latest AWS frame ≤ T (the
    surface state actually available at issue time).

Channels are normalised to ~O(1) here so the UNet's first conv sees inputs
on a comparable scale to the radar mm/h channels.

    python -m model.build_aux --data data/knmi --start 2026-05-09 --end 2026-05-12T03:00
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collectors.fetch_kmi_aws import fetch_all, flatten  # noqa: E402
from model.geo import grid_latlon  # noqa: E402

LOG = logging.getLogger("pluvio.build_aux")

# (column, normaliser): centre/scale chosen to map typical values to ~O(1).
AWS_CHANNELS: dict[str, tuple[float, float]] = {
    "pressure": (1013.0, 20.0),
    "temp_dry_shelter_avg": (10.0, 10.0),
    "humidity_rel_shelter_avg": (70.0, 30.0),
    "wind_speed_10m": (4.0, 4.0),
}


def _idw(lats, lons, vals, glat, glon, power=2.0) -> np.ndarray:
    """Inverse-distance weighting of point values onto the grid."""
    dy = glat[..., None] - lats  # (H, W, n)
    dx = glon[..., None] - lons
    d2 = dy * dy + dx * dx + 1e-9
    w = 1.0 / np.power(d2, power / 2.0)
    w /= w.sum(axis=-1, keepdims=True)
    return (w * vals).sum(axis=-1).astype("float32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/knmi"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Forecast timestamps (the alignment target) — read from the frame cache.
    # The dataset writes it next to the forecast files (fc_dir.parent).
    cache = args.data / "radar_forecast" / "frames_cache.npz"
    if not cache.exists():
        LOG.error("%s not found; build the dataset first.", cache)
        return 2
    d = np.load(cache, allow_pickle=False)
    fc_ts = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in d["ts_epoch"]]
    LOG.info("aligning to %d forecast timestamps", len(fc_ts))

    glat, glon = grid_latlon()

    # Pull AWS for the window, day by day (BETWEEN handles each chunk).
    import httpx

    start = _iso(args.start)
    end = _iso(args.end)
    rows: list[dict] = []
    with httpx.Client(headers={"User-Agent": "pluvio-build-aux/0.1"}) as client:
        cur = start
        while cur < end:
            nxt = min(cur + pd.Timedelta(days=1).to_pytimedelta(), end)
            cql = (
                f"timestamp BETWEEN '{cur.isoformat().replace('+00:00', 'Z')}' "
                f"AND '{nxt.isoformat().replace('+00:00', 'Z')}'"
            )
            page = fetch_all(client, 200_000, cql)
            rows.extend(flatten(f) for f in page.get("features", []) or [])
            cur = nxt
    aws = pd.DataFrame(rows)
    aws["timestamp"] = pd.to_datetime(aws["timestamp"], utc=True, errors="coerce")
    # Drop tz so comparisons against the tz-naive forecast datetime64 work.
    aws["timestamp"] = aws["timestamp"].dt.tz_localize(None)
    aws = aws.dropna(subset=["timestamp"]).sort_values("timestamp")
    LOG.info("pulled %d AWS rows, %d timesteps, %d stations",
             len(aws), aws["timestamp"].nunique(), aws["code"].nunique())

    aws_times = np.array(sorted(aws["timestamp"].unique()))

    # Build a (n_fc, H, W) grid per channel: for each forecast ts, IDW the
    # latest AWS frame at or before it.
    h, w = glat.shape
    out: dict[str, np.ndarray] = {
        ch: np.zeros((len(fc_ts), h, w), dtype="float32") for ch in AWS_CHANNELS
    }
    grids_cache: dict[np.datetime64, dict[str, np.ndarray]] = {}

    def grids_for(ts64) -> dict[str, np.ndarray]:
        if ts64 in grids_cache:
            return grids_cache[ts64]
        frame = aws[aws["timestamp"] == ts64]
        lats = frame["lat"].to_numpy(dtype="float64")
        lons = frame["lon"].to_numpy(dtype="float64")
        result: dict[str, np.ndarray] = {}
        for ch, (centre, scale) in AWS_CHANNELS.items():
            vals = frame[ch].to_numpy(dtype="float64")
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                result[ch] = np.zeros((h, w), dtype="float32")
                continue
            g = _idw(lats[mask], lons[mask], vals[mask], glat, glon)
            result[ch] = ((g - centre) / scale).astype("float32")
        grids_cache[ts64] = result
        return result

    n_aligned = 0
    for i, ts in enumerate(fc_ts):
        ts64 = np.datetime64(ts.replace(tzinfo=None))
        prior = aws_times[aws_times <= ts64]
        if len(prior) == 0:
            continue  # no AWS yet — leave zeros
        chosen = prior[-1]
        if (ts64 - chosen) > np.timedelta64(30, "m"):
            continue  # too stale
        g = grids_for(chosen)
        for ch in AWS_CHANNELS:
            out[ch][i] = g[ch]
        n_aligned += 1

    LOG.info("aligned AWS for %d/%d forecast timestamps", n_aligned, len(fc_ts))
    out_path = args.data / "radar_forecast" / "aux_cache.npz"
    np.savez_compressed(out_path, **out, channel_order=np.array(list(AWS_CHANNELS)))
    LOG.info("wrote %s (%d channels)", out_path, len(AWS_CHANNELS))
    return 0


def _iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
