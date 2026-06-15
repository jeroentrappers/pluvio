"""Run the trained UNet on the LATEST issue-time and write a precomputed
nowcast field for the backend to serve.

This is the production inference step: the forward collectors + `build_zarr
--append` keep `timeseries.zarr` current; this reads the newest issue-time,
assembles the 33-channel input (reusing ZarrCorrectionDataset.build_input), runs
the model at leads 30/60/90/120, and writes an npz the backend reprojects onto
its Belgium grid. Lead 0 is the current radar analysis (the interpolation anchor).

Output npz (`/opt/pluvio/serve/model_nowcast.npz`):
    leads        int16 [0, 30, 60, 90, 120]
    rates        float32 (5, 100, 100) mm/h on the KNMI-stereo analysis grid
    lat, lon     float64 (100, 100) grid coordinates (for reprojection)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", default="/opt/pluvio/zarr/timeseries.zarr")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "pluvio_unet.pt"))
    parser.add_argument("--out", default="/opt/pluvio/serve/model_nowcast.npz")
    parser.add_argument("--max-age-min", type=int, default=90,
                        help="Skip (don't overwrite) if the latest issue is staler than this.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import torch
    from model.zarr_dataset import ZarrCorrectionDataset
    from model.geo import grid_latlon

    ds = ZarrCorrectionDataset(args.zarr, leads_min=LEADS, build_index=False)
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

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from model.unet import PluvioUNet
    model = PluvioUNet(in_channels=ckpt.get("in_channels", ds.n_channels),
                       base_channels=ckpt.get("base_channels", 32))
    model.load_state_dict(ckpt["model"])
    model.eval()
    LOG.info("loaded %s (val_rmse=%.4f, %d ch)", args.checkpoint,
             ckpt.get("val_rmse", float("nan")), ckpt.get("in_channels", ds.n_channels))

    glat, glon = grid_latlon()
    H, W = glat.shape
    rates = np.zeros((len(LEADS) + 1, H, W), dtype="float32")
    # lead 0 = current radar analysis (anchor); read from the issue block
    radar = ds._open()["radar"]
    rates[0] = np.nan_to_num(np.asarray(radar[issue_idx, 0]), nan=0.0).astype("float32")
    with torch.no_grad():
        for i, lead in enumerate(LEADS, start=1):
            x = torch.from_numpy(ds.build_input(issue_idx, lead, hist)).unsqueeze(0)
            pred = model(x).squeeze().numpy()
            rates[i] = np.clip(pred, 0.0, None).astype("float32")  # rain ≥ 0

    # Reproject the KNMI-stereo fields onto the backend's regular Belgium grid
    # here (scipy is available in this venv) so the backend stays slim. Must match
    # backend cache.DEFAULT_BOUNDS / DEFAULT_GRID_SHAPE.
    from scipy.interpolate import griddata
    BE_W, BE_S, BE_E, BE_N = 1.5, 48.9, 7.5, 52.5
    BE_H = BE_WID = 100
    be_lon = np.linspace(BE_W, BE_E, BE_WID)
    be_lat = np.linspace(BE_N, BE_S, BE_H)            # row 0 = north (backend convention)
    be_LON, be_LAT = np.meshgrid(be_lon, be_lat)
    pts = np.column_stack([glon.ravel(), glat.ravel()])
    be_rates = np.zeros((rates.shape[0], BE_H, BE_WID), dtype="float32")
    for i in range(rates.shape[0]):
        g = griddata(pts, rates[i].ravel(), (be_LON, be_LAT), method="linear", fill_value=0.0)
        be_rates[i] = np.clip(np.nan_to_num(g, nan=0.0), 0.0, None).astype("float32")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # atomic write (temp + rename) so the backend never reads a half-written file
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    np.savez(tmp, leads=np.asarray((0, *LEADS), dtype="int16"), rates=be_rates,
             bounds=np.asarray([BE_W, BE_S, BE_E, BE_N], dtype="float64"),
             issue_epoch=np.int64(issue_epoch))
    tmp.replace(out)
    out.chmod(0o644)  # the backend worker container reads this as a different uid
    LOG.info("wrote %s — leads=%s, Belgium-grid max=%.2f mm/h, issue=%s",
             out, (0, *LEADS), float(be_rates.max()), issue_dt.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
