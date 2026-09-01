"""Archive the CONTINENTAL composite for future training — the serving frames are
already computed; not saving them discards them after the 3-h window.

The QPE archive covers the Benelux research box (training truth today). The
continental cube (Gibraltar->North Cape) exists only as the producer's rolling
frame store: this job appends each new SCAN frame (never interpolants) to one
daily zarr, block-mean downsampled to ~3 km — patch-crop training resolution,
~0.6 GB/day, bounded. Idempotent per stamp; run hourly.

Usage: PYTHONPATH=... python -m tools.wide_archive [--store DIR] [--out-root DIR] [--ds 2]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
LOG = logging.getLogger("pluvio.wide_archive")


def downsample(arr: np.ndarray, ds: int) -> np.ndarray:
    h, w = arr.shape
    ph, pw = -h % ds, -w % ds
    a = np.pad(arr.astype("float32"), ((0, ph), (0, pw)), constant_values=np.nan)
    return np.nanmean(a.reshape((h + ph) // ds, ds, (w + pw) // ds, ds),
                      axis=(1, 3)).astype("float16")


def main(argv=None) -> int:
    import zarr

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="/opt/pluvio/serve/observed_frames")
    p.add_argument("--out-root", default="/mnt/storagebox/qpe_wide")
    p.add_argument("--ds", type=int, default=2,
                   help="block-mean factor on the serving grid (2 -> ~3 km)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    store = pathlib.Path(args.store)
    bounds = None
    # serving bounds ride with the npz next to the store
    npz = pathlib.Path("/opt/pluvio/serve/observed.npz")
    if npz.exists():
        with np.load(npz) as z:
            bounds = [float(x) for x in z["bounds"]]

    added = 0
    for f in sorted(store.glob("*.npy")):
        stamp = f.stem
        try:
            ts = dt.datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        day_dir = pathlib.Path(args.out_root) / f"{ts:%Y/%m}"
        day_dir.mkdir(parents=True, exist_ok=True)
        zp = day_dir / f"{ts:%d}.zarr"
        root = zarr.open_group(str(zp), mode="a")
        epoch = int(ts.timestamp())
        if "times" in root.array_keys():
            times = list(int(x) for x in root["times"][:])
            if epoch in times:
                continue
        else:
            times = []
        arr = downsample(np.load(f).astype("float32"), args.ds)
        if "rate" not in root.array_keys():
            root.create_array("rate", shape=(0, *arr.shape), dtype="float16",
                              chunks=(1, *arr.shape))
            root.create_array("times", shape=(0,), dtype="int64", chunks=(512,))
            if bounds:
                root.attrs["bounds"] = bounds
                root.attrs["ds"] = args.ds
        n = root["rate"].shape[0]
        root["rate"].resize((n + 1, *arr.shape))
        root["rate"][n] = arr
        root["times"].resize((n + 1,))
        root["times"][n] = epoch
        added += 1
    LOG.info("wide archive: %d new frames appended", added)
    return 0


if __name__ == "__main__":
    sys.exit(main())
