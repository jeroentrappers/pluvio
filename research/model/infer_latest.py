"""Run the trained UNet on the LATEST issue-time and write a precomputed
nowcast field for the backend to serve.

This is the production inference step: the forward collectors + `build_zarr
--append` keep `timeseries.zarr` current; this reads the newest issue-time,
assembles the model input (reusing ZarrCorrectionDataset.build_input), runs
the model at leads 30/60/90/120, and writes an npz. Lead 0 is the current
radar analysis (the interpolation anchor).

Channel contract (2.3): the input layout comes from the CHECKPOINT's
`channel_recipe` (see `dataset_for_checkpoint`), not from whatever the store
happens to hold now — including whether the Lagrangian persistence channels
are part of the stack.

Grid contract (1.1): the store's georeference is READ from its zarr attrs
via `model.grid.Grid.from_zarr`, never assumed.
  * v3 store (regular lat/lon, e.g. build_store_v3 output): the model already
    runs on the serving grid, so the field is emitted as-is — no reprojection
    — with `bounds`/shape taken from the store's Grid.
  * legacy v2 store (no Grid attrs): falls back unchanged to today's
    behaviour — KNMI-stereo analysis grid, scipy griddata reprojection onto
    the hardcoded Belgium box — with a warning that the store lacks a Grid
    contract.

Output npz (`/opt/pluvio/serve/model_nowcast.npz`):
    leads        int16 [0, 30, 60, 90, 120]
    rates        float32 (5, H, W) mm/h on the store's serving grid
    bounds       float64 (west, south, east, north) of that grid
    issue_epoch  int64 UTC seconds
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("pluvio.infer_latest")

LEADS = (30, 60, 90, 120)


def dataset_for_checkpoint(zarr_path, ckpt: dict, leads_min=LEADS, *,
                           lagrangian_channels: int | None = None):
    """Build the ZarrCorrectionDataset that reproduces the input a checkpoint
    was TRAINED on, from the ``channel_recipe`` train.py stored alongside it.

    Without this, serving re-derives the channel layout from whatever the
    store holds today: adding one aux channel to the store would silently
    renumber every channel under a trained model, and turning the Lagrangian
    channels (2.3) on in training would leave inference feeding a model that
    expects them a stack that has none. Both are shape- or worse
    silence-level bugs, so the recipe is authoritative and the resulting
    channel count is cross-checked against ``in_channels``.

    A checkpoint from before the recipe existed carries none — every key
    falls back to the dataset's own default, i.e. exactly today's behaviour.
    ``lagrangian_channels`` overrides the recipe (the CLI flag), for
    ablating a channel set against a store by hand.
    """
    from model.zarr_dataset import ZarrCorrectionDataset

    recipe = dict(ckpt.get("channel_recipe") or {})
    kwargs: dict = {}
    if "history_steps" in recipe:
        kwargs["history_steps"] = int(recipe["history_steps"])
    if "history_step_min" in recipe:
        kwargs["history_step_min"] = int(recipe["history_step_min"])
    if "history_tolerance_s" in recipe:
        kwargs["history_tolerance_s"] = int(recipe["history_tolerance_s"])
    if recipe.get("aux_channels") is not None:
        kwargs["aux_channels"] = list(recipe["aux_channels"])
    lag = (int(lagrangian_channels) if lagrangian_channels is not None
           else int(recipe.get("lagrangian_channels", 0)))

    ds = ZarrCorrectionDataset(zarr_path, leads_min=tuple(leads_min), build_index=False,
                               lagrangian_channels=lag, **kwargs)

    # Statics are auto-discovered from the store, so they are checked rather
    # than forced: a static RENAMED between training and serving keeps the
    # channel count intact and would otherwise load in silence, feeding the
    # model a different plane at that index.
    recipe_static = recipe.get("static_channels")
    if recipe_static is not None and list(recipe_static) != list(ds.static_channels):
        raise ValueError(
            f"checkpoint was trained on static channels {list(recipe_static)} but "
            f"this store discovers {list(ds.static_channels)} — same count or not, "
            "the planes at those indices are not the ones the model learned"
        )

    expected = ckpt.get("in_channels", recipe.get("n_channels"))
    if expected is not None and int(expected) != ds.n_channels:
        raise ValueError(
            f"checkpoint expects {int(expected)} input channels but this store "
            f"resolves to {ds.n_channels} (recipe={recipe or None}, "
            f"aux={ds.aux_channels}, static={ds.static_channels}, "
            f"lagrangian={ds.lagrangian_channels}) — the store's channel set "
            "has drifted from the one the model was trained on"
        )
    return ds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", default="/opt/pluvio/zarr/timeseries.zarr")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "pluvio_unet.pt"))
    parser.add_argument("--out", default="/opt/pluvio/serve/model_nowcast.npz")
    parser.add_argument("--max-age-min", type=int, default=90,
                        help="Skip (don't overwrite) if the latest issue is staler than this.")
    parser.add_argument("--lagrangian-channels", type=int, default=None, choices=(0, 1, 2),
                        help="Override the checkpoint's Lagrangian channel count (2.3). "
                             "Default: whatever the checkpoint's channel recipe recorded "
                             "(0 for any checkpoint trained before the recipe existed).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import torch
    from model.geo import grid_latlon, log_resolved_geometry
    from model.grid import Grid, GridContractError

    log_resolved_geometry()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ds = dataset_for_checkpoint(args.zarr, ckpt, LEADS,
                                lagrangian_channels=args.lagrangian_channels)
    issue_idx = ds.latest_issue_idx()
    issue_epoch = int(ds._issue_epoch[issue_idx])
    issue_dt = datetime.fromtimestamp(issue_epoch, tz=timezone.utc)
    age_min = (datetime.now(timezone.utc) - issue_dt).total_seconds() / 60
    LOG.info("latest issue %s (%.0f min old)", issue_dt.isoformat(), age_min)
    if age_min > args.max_age_min:
        LOG.warning("latest issue too stale (%.0f > %d min) — not updating serve file",
                    age_min, args.max_age_min)
        return 1

    hist = ds.history_for(issue_idx)
    if hist is None:
        LOG.error("no radar history for the latest issue — cannot infer"); return 1

    from model.unet import PluvioUNet
    model = PluvioUNet(in_channels=ckpt.get("in_channels", ds.n_channels),
                       base_channels=ckpt.get("base_channels", 32))
    model.load_state_dict(ckpt["model"])
    model.eval()
    LOG.info("loaded %s (val_rmse=%.4f, %d ch, lagrangian=%d)", args.checkpoint,
             ckpt.get("val_rmse", float("nan")), ckpt.get("in_channels", ds.n_channels),
             ds.lagrangian_channels)

    # Store Grid (1.1): read it, never assume it. A v3 (regular lat/lon) store
    # carries its own georeference — the model runs and serves on that grid
    # directly, no reprojection. A legacy v2 store has no Grid attrs; fall
    # back to today's KNMI-stereo + griddata-onto-Belgium-box behaviour.
    store_root = ds._open()
    try:
        store_grid = Grid.from_zarr(store_root)
    except GridContractError as exc:
        store_grid = None
        LOG.warning("store %s has no Grid contract (legacy v2 store): %s", args.zarr, exc)

    if store_grid is not None:
        H, W = store_grid.shape
    else:
        glat, glon = grid_latlon()
        H, W = glat.shape
        if (H, W) != tuple(ds.grid_hw):
            raise GridContractError(
                f"legacy fallback assumes geo.GRID {(H, W)} but store radar is "
                f"{tuple(ds.grid_hw)} — rebuild the store with a Grid contract"
            )

    rates = np.zeros((len(LEADS) + 1, H, W), dtype="float32")
    # lead 0 = current radar analysis (anchor); read from the issue block
    radar = store_root["radar"]
    rates[0] = np.nan_to_num(np.asarray(radar[issue_idx, 0]), nan=0.0).astype("float32")
    with torch.no_grad():
        for i, lead in enumerate(LEADS, start=1):
            x = torch.from_numpy(ds.build_input(issue_idx, lead, hist)).unsqueeze(0)
            pred = model(x).squeeze().numpy()
            rates[i] = np.clip(pred, 0.0, None).astype("float32")  # rain ≥ 0

    if store_grid is not None:
        # v3: the store is already the regular lat/lon serving grid — emit it
        # as-is, bounds/shape from the Grid contract.
        out_rates = rates
        out_bounds = np.asarray(store_grid.bounds, dtype="float64")
    else:
        # legacy v2: reproject the KNMI-stereo fields onto the backend's
        # regular Belgium grid here (scipy is available in this venv) so the
        # backend stays slim. Must match backend cache.DEFAULT_BOUNDS /
        # DEFAULT_GRID_SHAPE.
        from scipy.interpolate import griddata
        BE_W, BE_S, BE_E, BE_N = 1.5, 48.9, 7.5, 52.5
        BE_H = BE_WID = 100
        be_lon = np.linspace(BE_W, BE_E, BE_WID)
        be_lat = np.linspace(BE_N, BE_S, BE_H)        # row 0 = north (backend convention)
        be_LON, be_LAT = np.meshgrid(be_lon, be_lat)
        pts = np.column_stack([glon.ravel(), glat.ravel()])
        be_rates = np.zeros((rates.shape[0], BE_H, BE_WID), dtype="float32")
        for i in range(rates.shape[0]):
            g = griddata(pts, rates[i].ravel(), (be_LON, be_LAT), method="linear", fill_value=0.0)
            be_rates[i] = np.clip(np.nan_to_num(g, nan=0.0), 0.0, None).astype("float32")
        out_rates = be_rates
        out_bounds = np.asarray([BE_W, BE_S, BE_E, BE_N], dtype="float64")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # atomic write (temp + rename) so the backend never reads a half-written file
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    np.savez(tmp, leads=np.asarray((0, *LEADS), dtype="int16"), rates=out_rates,
             bounds=out_bounds, issue_epoch=np.int64(issue_epoch))
    tmp.replace(out)
    out.chmod(0o644)  # the backend worker container reads this as a different uid
    LOG.info("wrote %s — leads=%s, grid=%s max=%.2f mm/h, issue=%s",
             out, (0, *LEADS), out_rates.shape[-2:], float(out_rates.max()), issue_dt.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
