"""Low-latency nowcast issue (TODO 4.1): infer at any 5-minute time from OUR
composite, not from the 30-minute KNMI issue.

Measured 2026-09-04: KNMI's RAC_FM files are 30-min issues that land ~30 min
after their issue time, so the served nowcast is 30–60 min stale, while the
QPE composite is 5-min cadence and ~20 min old. This builds the model input
for a time ``t`` as:

  history frames   composite at t, t-step, … (history_steps), resampled onto
                   the model grid (bilinear at cell centres)
  nowcast @ lead   the newest KNMI nowcast issue i0 <= t, time-shifted: its
                   lead nearest to (t - issue(i0)) + lead, clamped to its range
  aux              carried forward from issue i0 (unchanged for < 30 min)
  statics          from the store

and writes the same npz `infer_latest` writes, to a SIDE path. Nothing here
is served until the benchmark shows the composite-driven input does not
hurt skill (distribution shift: composite vs KNMI analysis) — see
``--evaluate`` in TODO 4.1.

    python -m tools.lowlatency_infer --zarr … --qpe-root /mnt/storagebox/qpe \\
        --checkpoint … --out /opt/pluvio/serve/lowlatency_nowcast.npz [--at 2026-09-04T09:35:00Z]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.grid import Grid, GridContractError  # noqa: E402
from model.infer_latest import (  # noqa: E402
    LEADS, LEGACY_AUX_CHANNELS, dataset_for_checkpoint, to_serving_grid, write_nowcast_npz,
)
from model.zarr_dataset import _normalise, assemble_input  # noqa: E402
from tools.scoreboard import QpeTruth  # noqa: E402

LOG = logging.getLogger("pluvio.lowlatency")
MAX_COMPOSITE_RATE_MM_H = 150.0


def sample_regular_raster(rate: np.ndarray, bounds, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear sample of an edge-referenced regular lat/lon raster (row 0 =
    north) at the points (lat, lon). NaN source cells count as 0; points
    outside the raster → 0 (dry, not NaN — the model input is nan_to_num'd
    anyway, and 'outside the composite' is 'no observed rain' for a nowcast)."""
    from scipy.ndimage import map_coordinates

    w, s, e, n = (float(x) for x in bounds)
    h, wd = rate.shape
    rows = (n - lat) / (n - s) * h - 0.5
    cols = (lon - w) / (e - w) * wd - 0.5
    out = map_coordinates(np.nan_to_num(rate, nan=0.0), [rows, cols], order=1, mode="constant", cval=0.0)
    # The 1-km composite carries clutter/hail spikes (365 mm/h seen live) the
    # KNMI analysis the model trained on never shows; cap like a radar product.
    return np.clip(out, 0.0, MAX_COMPOSITE_RATE_MM_H).astype("float32")


def model_grid_latlon(store_root):
    """(lat, lon) of the model grid's cell centres — Grid contract if present,
    else the legacy KNMI stereographic analysis grid."""
    try:
        grid = Grid.from_zarr(store_root)
    except GridContractError:
        attrs = dict(store_root.attrs)
        hw = tuple(int(x) for x in store_root["radar"].shape[-2:])
        if attrs.get("bounds") is not None and len(attrs["bounds"]) == 4:
            # v3-style store without the full Grid contract: a regular
            # lat/lon box whose `bounds` are the cell-centre envelope.
            grid = Grid.regular(tuple(float(x) for x in attrs["bounds"]), hw)
        else:
            from model.geo import grid_latlon
            lat, lon = grid_latlon()
            if lat.shape != hw:
                raise GridContractError(f"legacy grid {lat.shape} != store radar {hw}")
            return None, lat, lon
    lat, lon = grid.latlon()
    return grid, lat, lon


def shifted_lead_index(leads_min, age_min: float, lead_min: int) -> int:
    """Index of the KNMI lead nearest to (age + lead), clamped to the issue's range."""
    target = age_min + lead_min
    leads = np.asarray(leads_min, dtype="float64")
    return int(np.argmin(np.abs(leads - min(target, leads.max()))))


def build_lowlatency_input(ds, store_root, qpe: QpeTruth, t_epoch: int, lead_min: int,
                           lat: np.ndarray, lon: np.ndarray, aux_names) -> tuple[np.ndarray, dict]:
    """(input, info) for time ``t_epoch`` and ``lead_min``."""
    if ds.lagrangian_channels:
        raise NotImplementedError("low-latency path does not build Lagrangian planes yet")
    step_s = ds.history_step_min * 60
    frames = []
    for k in range(ds.history_steps):
        te = t_epoch - (ds.history_steps - 1 - k) * step_s
        got = qpe.frame(te)
        if got is None:
            raise RuntimeError(f"no composite frame at {dt.datetime.fromtimestamp(te, dt.UTC).isoformat()}")
        rate, bounds = got
        frames.append(sample_regular_raster(rate, bounds, lat, lon))
    epochs = ds._issue_epoch
    i0 = int(np.searchsorted(epochs, t_epoch, side="right") - 1)
    if i0 < 0:
        raise RuntimeError("no store issue at or before the requested time")
    age_min = (t_epoch - int(epochs[i0])) / 60.0
    store_leads = [int(x) for x in np.asarray(store_root["leads_min"][:])]
    li = shifted_lead_index(store_leads, age_min, lead_min)
    nowcast = np.asarray(store_root["radar"][i0, li], dtype="float32")
    aux_raw = [np.asarray(store_root[name][i0]) for name in aux_names]
    statics = [_normalise(n, np.asarray(store_root[n][:])) for n in ds.static_channels]
    valid = dt.datetime.fromtimestamp(t_epoch, dt.UTC) + dt.timedelta(minutes=lead_min)
    x = assemble_input(frames, np.nan_to_num(nowcast, nan=0.0), lead_min, valid, aux_names, aux_raw, statics)
    info = {"issue_idx": i0, "issue_epoch": int(epochs[i0]), "age_min": age_min,
            "knmi_lead_used": store_leads[li]}
    return x, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", required=True)
    p.add_argument("--qpe-root", default="/mnt/storagebox/qpe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--at", default=None, help="ISO UTC time; default: newest composite slot")
    p.add_argument("--max-composite-age-min", type=float, default=40.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import torch
    import zarr

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ds = dataset_for_checkpoint(args.zarr, ckpt)
    root = zarr.open_group(args.zarr, mode="r")
    grid, lat, lon = model_grid_latlon(root)
    qpe = QpeTruth(args.qpe_root)

    if args.at:
        t = int(dt.datetime.fromisoformat(args.at.replace("Z", "+00:00")).timestamp())
    else:
        now = int(dt.datetime.now(dt.UTC).timestamp())
        t = None
        for back in range(0, int(args.max_composite_age_min * 60) + 1, 300):
            cand = (now - back) // 300 * 300
            if qpe.frame(cand) is not None:
                t = cand
                break
        if t is None:
            LOG.error("no composite frame within %.0f min", args.max_composite_age_min)
            return 1
    aux_names = ds.aux_channels if ckpt.get("channel_recipe") else list(LEGACY_AUX_CHANNELS)

    from model.unet import PluvioUNet
    quantiles = ckpt.get("quantiles")
    model = PluvioUNet(in_channels=ckpt["in_channels"], base_channels=ckpt.get("base_channels", 32),
                       out_channels=len(quantiles) if quantiles else 1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    median_index = list(quantiles).index(0.5) if quantiles else 0

    rates = np.zeros((len(LEADS) + 1, *lat.shape), dtype="float32")
    info = None
    with torch.no_grad():
        for i, lead in enumerate(LEADS, start=1):
            x, info = build_lowlatency_input(ds, root, qpe, t, lead, lat, lon, aux_names)
            if i == 1:
                rates[0] = x[ds.history_steps - 1]           # newest composite frame = lead 0
            pred = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            pred = np.sort(pred, axis=0)[median_index] if quantiles else pred[0]
            rates[i] = np.clip(pred, 0.0, None)
    out_rates, out_bounds = to_serving_grid(rates, grid, None if grid is not None else (lat, lon))
    out = write_nowcast_npz(pathlib.Path(args.out), (0, *LEADS), out_rates, out_bounds, t,
                            {"source": np.asarray("lowlatency-composite")})
    LOG.info("wrote %s — issue %s (composite), KNMI issue %s is %.0f min old (its lead %d fed our lead %d), "
             "max %.2f mm/h", out, dt.datetime.fromtimestamp(t, dt.UTC).isoformat(),
             dt.datetime.fromtimestamp(info["issue_epoch"], dt.UTC).isoformat(), info["age_min"],
             info["knmi_lead_used"], LEADS[-1], float(rates.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
