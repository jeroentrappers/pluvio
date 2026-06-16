"""Produce the seamless serving artifact `model_forecast.npz` for the backend.

This generalises `infer_latest.py` (nowcast-only, learned-UNet) to the full
0 → 240 h cube and, crucially, supports **two producers** (recommendation #4):

    --producer classical   # pysteps⊕AIFS — the product baseline, ships TODAY
    --producer model       # the learned SeamlessNet — a GATED upgrade

The default is `classical`: the PWA gets a credible, source-tagged, confidence-
banded forecast with zero dependence on the research model. The learned model is
swapped in only after it beats this baseline on the champion/challenger gate
(docs/plan_overview.md §5) — `--producer model` is what the gate promotes.

Output npz (`/opt/pluvio/serve/model_forecast.npz`) — a superset of the old
`model_nowcast.npz`, so `backend/.../model.py` reads it for *all* bands:

    leads        int16   (n_lead,)            lead minutes (0 … 14400)
    rates        float32 (n_lead, H, W)       mm/h on the Belgium serving grid
    source       <U8     (n_lead,)            "nowcast" | "blend" | "nwp"
    confidence   float32 (n_lead,)            0–1, widening with lead
    bounds       float64 (4,)                 [W, S, E, N] of the serving grid
    issue_epoch  int64                        UTC seconds
    producer     <U16                         "classical" | "model"
    engine       <U24                         nowcast engine (pysteps/fallback/net)

The serving lead set mirrors the backend bands (schedules.py): 0–120 @10, then
hourly to 24 h, 3-hourly to 240 h.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys
import tempfile

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("pluvio.produce_forecast")

# Serving leads = the union of the backend's band lead sets (schedules.py).
SERVE_LEADS: tuple[int, ...] = (
    tuple(range(0, 121, 10)) + tuple(range(180, 24 * 60, 60)) + tuple(range(24 * 60, 240 * 60 + 1, 180))
)

# Belgium serving grid — must match backend cache.DEFAULT_BOUNDS / shape and
# infer_latest.py.
BE_BOUNDS = (1.5, 48.9, 7.5, 52.5)  # W, S, E, N
BE_SHAPE = (100, 100)

OPERA_DT_MIN = 15  # OPERA analysis cadence


def _analysis_to_belgium(rates_analysis: np.ndarray) -> np.ndarray:
    """Reproject (n, H, W) fields from the KNMI-stereo analysis grid onto the
    backend's regular Belgium lat/lon grid (same transform as infer_latest)."""
    from scipy.interpolate import griddata

    from model.geo import grid_latlon

    glat, glon = grid_latlon()
    bw, bs, be, bn = BE_BOUNDS
    bh, bwid = BE_SHAPE
    be_lon = np.linspace(bw, be, bwid)
    be_lat = np.linspace(bn, bs, bh)  # row 0 = north
    be_LON, be_LAT = np.meshgrid(be_lon, be_lat)
    pts = np.column_stack([glon.ravel(), glat.ravel()])
    out = np.zeros((rates_analysis.shape[0], bh, bwid), dtype="float32")
    for i in range(rates_analysis.shape[0]):
        g = griddata(pts, rates_analysis[i].ravel(), (be_LON, be_LAT),
                     method="linear", fill_value=0.0)
        out[i] = np.clip(np.nan_to_num(g, nan=0.0), 0.0, None).astype("float32")
    return out


def _opera_history(storage: pathlib.Path, n_frames: int, max_age_min: int):
    """Most-recent ``n_frames`` OPERA RATE analyses reprojected to the analysis
    grid (oldest→newest), plus the issue datetime. Returns (history, issue_dt)."""
    from tools.build_seamless_zarr import _index_tiffs
    from model.nwp_regrid import reproject_to_analysis_grid

    idx = _index_tiffs(storage / "opera" / "RATE")
    if len(idx) < n_frames:
        raise RuntimeError(f"need {n_frames} OPERA frames, found {len(idx)}")
    recent = idx[-n_frames:]
    issue_dt = recent[-1][0]
    age_min = (dt.datetime.now(dt.UTC) - issue_dt).total_seconds() / 60
    if age_min > max_age_min:
        raise RuntimeError(f"latest OPERA {issue_dt.isoformat()} is {age_min:.0f} min old")
    hist = np.stack([np.nan_to_num(reproject_to_analysis_grid(p), nan=0.0) for _, p in recent])
    return hist.astype("float32"), issue_dt


def _aifs_cube(storage: pathlib.Path, leads_min, issue_dt) -> np.ndarray | None:
    """Build a (n_lead, H, W) raw-AIFS precip cube on the analysis grid for the
    given leads, or None if AIFS isn't available yet. Best-effort: the product
    degrades to nowcast-only rather than failing."""
    try:
        from model.aifs_cube import build_aifs_cube  # optional helper
    except Exception:
        LOG.info("no AIFS cube helper / data — outlook degrades to nowcast carry-forward")
        return None
    try:
        return build_aifs_cube(storage / "aifs", leads_min, issue_dt)
    except Exception as exc:
        LOG.warning("AIFS cube build failed (%s) — nowcast-only", exc)
        return None


def produce_classical(storage: pathlib.Path, leads, max_age_min: int):
    from model.classical import seamless_cube

    hist, issue_dt = _opera_history(storage, n_frames=6, max_age_min=max_age_min)
    aifs = _aifs_cube(storage, leads, issue_dt)
    fc = seamless_cube(hist, leads, dt_min=OPERA_DT_MIN, aifs_rates=aifs)
    return fc.rates, fc.source, fc.confidence, fc.engine, issue_dt


def produce_model(zarr_path: str, leads, max_age_min: int, ckpt: str):
    """The learned SeamlessNet producer (gated upgrade).

    Assembles inputs through ``SeamlessDataset.build_input`` (the same code path
    training uses) so the channel layout — history + AIFS + every aux + static —
    matches the checkpoint exactly. Hand-stacking only the radar+AIFS channels
    (as a quick draft did) silently mismatches any model trained with aux/static
    inputs; routing through the dataset removes that whole failure class.
    """
    import torch
    from datetime import datetime, timezone

    from model.classical import confidence_for_leads, source_for_lead
    from model.seamless import SeamlessNet
    from model.seamless_dataset import SeamlessDataset

    ds = SeamlessDataset(zarr_path, leads_min=leads, build_index=False)
    issue_idx = ds.latest_issue_idx()
    issue_dt = datetime.fromtimestamp(int(ds._issue_epoch[issue_idx]), tz=timezone.utc)
    age_min = (datetime.now(timezone.utc) - issue_dt).total_seconds() / 60
    if age_min > max_age_min:
        raise RuntimeError(f"latest issue {issue_dt.isoformat()} is {age_min:.0f} min old")

    # Most-recent K analyses stepping back from the issue time (mirrors _build_index).
    issue_e = int(ds._issue_epoch[issue_idx])
    step = ds.history_step_min * 60
    history_idx = tuple(ds._lookup(issue_e - k * step) for k in range(ds.history_steps - 1, -1, -1))
    if any(h is None for h in history_idx):
        raise RuntimeError("no radar history for the latest issue — cannot infer")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    quantiles = ck.get("quantiles")  # None for a deterministic checkpoint
    net = SeamlessNet(in_channels=ck["in_channels"], base_channels=ck["base_channels"],
                      quantiles=tuple(quantiles) if quantiles else None)
    net.load_state_dict(ck["model"]); net.eval()

    H, W = ds.build_input(issue_idx, leads[0], history_idx).shape[-2:]
    rates = np.empty((len(leads), H, W), dtype="float32")
    with torch.no_grad():
        for i, lead in enumerate(leads):
            x = torch.from_numpy(ds.build_input(issue_idx, lead, history_idx)[None])
            cond = torch.from_numpy(ds.cond_for(issue_idx, lead)[None]).float()
            out = net(x, cond)
            rates[i] = np.clip(net.median(out).squeeze().numpy(), 0.0, None)  # point = median quantile
    source = [source_for_lead(int(l)) for l in leads]
    return rates, source, confidence_for_leads(leads), f"seamless:{pathlib.Path(ckpt).stem}", issue_dt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--producer", choices=["classical", "model"], default="classical")
    p.add_argument("--storage", default="/mnt/storagebox",
                   help="root with opera/ aifs/ (classical producer)")
    p.add_argument("--zarr", default="/opt/pluvio/zarr/seamless.zarr",
                   help="built seamless store (model producer — channel-faithful input)")
    p.add_argument("--out", default="/opt/pluvio/serve/model_forecast.npz")
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "pluvio_seamless.pt"))
    p.add_argument("--max-age-min", type=int, default=90)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    storage = pathlib.Path(args.storage)
    leads = list(SERVE_LEADS)
    if args.producer == "classical":
        rates_a, source, conf, engine, issue_dt = produce_classical(storage, leads, args.max_age_min)
    else:
        rates_a, source, conf, engine, issue_dt = produce_model(
            args.zarr, leads, args.max_age_min, args.checkpoint)

    be_rates = _analysis_to_belgium(rates_a)
    LOG.info("producer=%s engine=%s issue=%s leads=%d max=%.2f mm/h",
             args.producer, engine, issue_dt.isoformat(), len(leads), float(be_rates.max()))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    np.savez(
        tmp,
        leads=np.asarray(leads, dtype="int16"),
        rates=be_rates,
        source=np.asarray(source),
        confidence=conf.astype("float32"),
        bounds=np.asarray(BE_BOUNDS, dtype="float64"),
        issue_epoch=np.int64(issue_dt.timestamp()),
        producer=np.asarray(args.producer),
        engine=np.asarray(engine),
    )
    tmp.replace(out)
    out.chmod(0o644)
    LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
