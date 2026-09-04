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
import functools
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


@functools.lru_cache(maxsize=1)
def _belgium_triangulation():
    """Delaunay triangulation of the analysis grid + the Belgium target points.

    Both grids are FIXED for a given PLUVIO_GRID_N, so the triangulation is
    constant across leads and across runs. `griddata` rebuilds it on every call,
    which cost 68 s of a 88 s producer run (107 leads x one triangulation of 65k
    points). Building it once and reusing it via LinearNDInterpolator is the same
    interpolation, ~30x faster.
    """
    from scipy.spatial import Delaunay

    from model.geo import grid_latlon

    glat, glon = grid_latlon()
    bw, bs, be, bn = BE_BOUNDS
    bh, bwid = BE_SHAPE
    be_lon = np.linspace(bw, be, bwid)
    be_lat = np.linspace(bn, bs, bh)  # row 0 = north
    be_LON, be_LAT = np.meshgrid(be_lon, be_lat)
    pts = np.column_stack([glon.ravel(), glat.ravel()])
    tri = Delaunay(pts)
    targets = np.column_stack([be_LON.ravel(), be_LAT.ravel()])
    return tri, targets, (bh, bwid)


def _analysis_to_belgium(rates_analysis: np.ndarray) -> np.ndarray:
    """Reproject (n, H, W) fields from the KNMI-stereo analysis grid onto the
    backend's regular Belgium lat/lon grid (same transform as infer_latest)."""
    from scipy.interpolate import LinearNDInterpolator

    tri, targets, (bh, bwid) = _belgium_triangulation()
    out = np.zeros((rates_analysis.shape[0], bh, bwid), dtype="float32")
    for i in range(rates_analysis.shape[0]):
        interp = LinearNDInterpolator(tri, rates_analysis[i].ravel(), fill_value=0.0)
        g = interp(targets).reshape(bh, bwid)
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


_classical_inputs: dict = {}   # hist/aifs of the last produce_classical(return_inputs=True) call


def produce_classical(storage: pathlib.Path, leads, max_age_min: int, *, return_inputs: bool = False):
    from model.classical import seamless_cube

    hist, issue_dt = _opera_history(storage, n_frames=6, max_age_min=max_age_min)
    aifs = _aifs_cube(storage, leads, issue_dt)
    fc = seamless_cube(hist, leads, dt_min=OPERA_DT_MIN, aifs_rates=aifs)
    _classical_inputs.update({"hist": hist, "aifs": aifs}) if return_inputs else None
    if fc.phase_offset_px is not None:
        LOG.info("classical: NWP phase offset at handoff dy=%.1f dx=%.1f px",
                 fc.phase_offset_px[0], fc.phase_offset_px[1])
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

    # Rebuild the EXACT input recipe the checkpoint was trained with. Checkpoints
    # written since 2026-08 record it; without this the dataset re-derives the layout
    # (auto-discovering every aux and static array in the live zarr) and silently
    # assembles a different channel stack than training used — which is how a served
    # model can look healthy and forecast nonsense.
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ds = SeamlessDataset(
        zarr_path, leads_min=leads, build_index=False,
        history_steps=ck.get("history_steps", 6),
        aux_channels=ck.get("aux_channels"),          # None on a legacy ckpt → auto
        include_static=bool(ck.get("static_channels")) if "static_channels" in ck else True,
        use_advection=bool(ck.get("advection", False)))
    if ds.n_channels != ck["in_channels"]:
        raise RuntimeError(
            f"channel mismatch: checkpoint expects {ck['in_channels']}, this zarr assembles "
            f"{ds.n_channels} (aux={ds.aux_channels}, static={ds.static_channels}, "
            f"history={ds.history_steps}, advection={ds.use_advection}). "
            f"Checkpoint recipe: aux={ck.get('aux_channels')}, "
            f"static={ck.get('static_channels')}, history={ck.get('history_steps')}, "
            f"advection={ck.get('advection')}.")
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



def produce_hybrid(storage: pathlib.Path, zarr_path: str, leads, max_age_min: int, ckpt: str):
    """Learned nowcast spliced into the classical full-horizon cube.

    c17-C is a **nowcast** model: trained on leads 0-120 min and gated only there
    (it beats real pysteps on MAE, CSI@0.1, CSI@1 and FSS at 3/9/15 km at all 12
    served leads). It has never seen a lead beyond 2 h. Serving it as the whole
    0-240 h cube would claim day-5 skill it was never evaluated for -- and in
    practice just crashes, because the live zarr carries only the nowcast leads
    (`KeyError: 180`).

    So: take the classical pysteps⊕AIFS cube for the full horizon, then overwrite
    the leads the model actually covers. Every lead is then served by whichever
    producer is verified for it, and `source` stays honest per lead.
    """
    rates, source, conf, engine_c, issue_dt = produce_classical(
        storage, leads, max_age_min, return_inputs=True)

    import zarr
    zleads = {int(x) for x in zarr.open_group(zarr_path, mode="r")["leads_min"][:]}
    model_leads = [l for l in leads if l in zleads]
    if not model_leads:
        LOG.warning("live zarr covers none of the serving leads — classical only")
        return rates, source, conf, f"{engine_c}(model:none)", issue_dt

    m_rates, m_source, m_conf, engine_m, m_issue = produce_model(
        zarr_path, model_leads, max_age_min, ckpt)

    # Both producers must describe the SAME analysis, or we would splice a learned
    # nowcast from one issue time onto an outlook from another.
    skew_min = abs((m_issue - issue_dt).total_seconds()) / 60
    if skew_min > OPERA_DT_MIN:
        raise RuntimeError(
            f"issue-time skew {skew_min:.0f} min between model ({m_issue.isoformat()}) "
            f"and classical ({issue_dt.isoformat()}) — refusing to splice")

    idx = {l: i for i, l in enumerate(leads)}
    for j, l in enumerate(model_leads):
        i = idx[l]
        rates[i] = m_rates[j]
        source[i] = m_source[j]
        conf[i] = m_conf[j]

    # Continue the SERVED nowcast into the blend (2.8): the classical cube's
    # 2-6 h radar arm was pysteps' own extrapolation from t0 — a second
    # nowcast of the same rain, 20+ cells away from the model's 120-min field
    # on 2026-09-04, so the timeline jumped at the 120→180 seam.
    from model.classical import anchored_blend, global_motion_robust
    hist = _classical_inputs.get("hist")
    anchor_lead = max(model_leads)
    if hist is not None and len(hist) >= 2:
        motion = global_motion_robust(hist[-2], hist[-1])
        rates, offset = anchored_blend(
            rates, source, leads, anchor_field=rates[idx[anchor_lead]],
            anchor_lead_min=anchor_lead, motion_per_frame=motion, dt_min=OPERA_DT_MIN,
            aifs_rates=_classical_inputs.get("aifs"))
        LOG.info("hybrid: blend re-anchored on the model's %d-min field; motion/frame dy=%.1f dx=%.1f px; "
                 "NWP phase offset %s", anchor_lead, motion[0], motion[1], offset)
    LOG.info("hybrid: model on %d leads (%d-%d min), classical on the remaining %d",
             len(model_leads), min(model_leads), max(model_leads), len(leads) - len(model_leads))
    return rates, source, conf, f"hybrid:{engine_m}+{engine_c}", issue_dt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--producer", choices=["classical", "model", "hybrid"], default="classical",
                   help="hybrid = learned nowcast (0-120 min, the gated regime) spliced "
                        "into the classical cube for the rest of the 0-240 h horizon")
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
    elif args.producer == "hybrid":
        rates_a, source, conf, engine, issue_dt = produce_hybrid(
            storage, args.zarr, leads, args.max_age_min, args.checkpoint)
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
