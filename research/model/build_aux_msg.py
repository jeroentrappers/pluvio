"""Add a Meteosat convective-signal channel to the aux cache.

EUMETView WMS returns *rendered* imagery, not raw values. For the RDT
(Rapid Developing Thunderstorms) product that's fine: storm cells are
drawn as opaque polygons over a transparent background, so the alpha
channel gives a clean 0/1 "convective cell present" mask. We max-pool it
to a ~convection-proximity field, reproject onto the analysis grid, and
align to forecast timestamps — then merge into aux_cache.npz alongside the
AWS channels.

    python -m model.build_aux_msg --data data/knmi --start 2026-05-09 --end 2026-05-12T03:00
"""

from __future__ import annotations

import argparse
import io
import logging
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.geo import bbox, grid_latlon  # noqa: E402

LOG = logging.getLogger("pluvio.build_aux_msg")

WMS_URL = "https://view.eumetsat.int/geoserver/wms"
LAYER = "msg_fes:rdt"
FRAME_PX = 256
STEP_MIN = 15  # Meteosat cadence
PROXIMITY_RADIUS = 4  # grid cells of max-pool dilation → "convection nearby"


def _frame_to_grid_index() -> tuple[np.ndarray, np.ndarray]:
    """Precompute, for each grid cell, the frame pixel (row, col) to sample.

    The WMS frame is plate-carrée (EPSG:4326) over the grid's lon/lat bbox;
    the grid is stereographic. Over the Benelux the curvature error is small,
    so nearest-pixel sampling by lat/lon is adequate for a coarse mask.
    """
    glat, glon = grid_latlon()
    w, s, e, n = bbox()
    col = ((glon - w) / (e - w) * (FRAME_PX - 1)).round().astype("int32")
    row = ((n - glat) / (n - s) * (FRAME_PX - 1)).round().astype("int32")
    return np.clip(row, 0, FRAME_PX - 1), np.clip(col, 0, FRAME_PX - 1)


def _maxpool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square max-pool dilation so sparse storm cells become a vicinity field."""
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out = np.maximum(out, np.roll(np.roll(mask, dy, axis=0), dx, axis=1))
    return out


def fetch_mask(client: httpx.Client, when: datetime) -> np.ndarray | None:
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap",
        "layers": LAYER, "styles": "", "format": "image/png", "transparent": "true",
        "crs": "EPSG:4326",
        "bbox": f"{bbox()[1]},{bbox()[0]},{bbox()[3]},{bbox()[2]}",
        "width": str(FRAME_PX), "height": str(FRAME_PX),
        "time": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    try:
        r = client.get(WMS_URL, params=params, timeout=60)
        r.raise_for_status()
        alpha = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))[..., 3]
        return (alpha > 0).astype("float32")
    except (httpx.HTTPError, OSError) as exc:
        LOG.warning("RDT frame %s failed: %s", when, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/knmi"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    aux_path = args.data / "radar_forecast" / "aux_cache.npz"
    frames_cache = args.data / "radar_forecast" / "frames_cache.npz"
    if not frames_cache.exists():
        LOG.error("%s not found; build the dataset first.", frames_cache)
        return 2

    d = np.load(frames_cache, allow_pickle=False)
    fc_ts = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in d["ts_epoch"]]
    LOG.info("aligning RDT to %d forecast timestamps", len(fc_ts))

    row_idx, col_idx = _frame_to_grid_index()
    h, w = row_idx.shape

    # Download RDT at 15-min cadence across the window, build a {ts: grid} map.
    start = _iso(args.start)
    end = _iso(args.end)
    # Snap to the 15-min grid.
    t = start.replace(minute=(start.minute // STEP_MIN) * STEP_MIN, second=0, microsecond=0)
    masks: dict[datetime, np.ndarray] = {}
    with httpx.Client(headers={"User-Agent": "pluvio-build-aux-msg/0.1"}) as client:
        while t <= end:
            m = fetch_mask(client, t)
            if m is not None:
                grid_mask = m[row_idx, col_idx]  # reproject to grid
                masks[t] = _maxpool(grid_mask, PROXIMITY_RADIUS)
            t += timedelta(minutes=STEP_MIN)
    LOG.info("downloaded %d RDT frames", len(masks))

    mask_times = np.array(sorted(masks))
    rdt = np.zeros((len(fc_ts), h, w), dtype="float32")
    n_aligned = 0
    for i, ts in enumerate(fc_ts):
        ts_naive = ts.replace(tzinfo=None)
        prior = [mt for mt in mask_times if mt.replace(tzinfo=None) <= ts_naive]
        if not prior:
            continue
        chosen = prior[-1]
        if (ts_naive - chosen.replace(tzinfo=None)) > timedelta(minutes=20):
            continue
        rdt[i] = masks[chosen]
        n_aligned += 1
    LOG.info("aligned RDT for %d/%d forecast timestamps", n_aligned, len(fc_ts))

    # Merge into the existing aux cache (preserve AWS channels).
    if aux_path.exists():
        existing = np.load(aux_path, allow_pickle=True)
        order = [str(c) for c in existing["channel_order"]]
        data = {ch: existing[ch] for ch in order}
    else:
        order, data = [], {}
    if "msg_rdt" not in order:
        order.append("msg_rdt")
    data["msg_rdt"] = rdt
    np.savez_compressed(aux_path, **data, channel_order=np.array(order))
    LOG.info("wrote %s (channels: %s)", aux_path, order)
    return 0


def _iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
