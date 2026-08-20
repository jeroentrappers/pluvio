"""Append the static terrain channels to an existing seamless zarr, in place.

Adds elevation / landmask / distance-to-coast from `static.npz` to a store that was
built without them:

    static_elevation_m          (H, W) float32   metres above sea level
    static_landmask             (H, W) float32   0/1
    static_distance_to_coast_km (H, W) float32   km to nearest sea cell

Why this exists: `tools/build_seamless_zarr.py` only writes `static_*` when a
static.npz is found, and until 2026-08 it looked in `model/` while
`model/build_static.py` writes to `data/` — and it skipped in **silence** on a
miss. So every run through c16 trained with no terrain channels. Rebuilding an
11 GB, 14-month zarr just to add three 2-D arrays would cost hours of
reprojection for no reason; this adds them to the existing store in seconds.

`SeamlessDataset` picks `static_*` up automatically (`_discover(root, False)` with
``include_static=True``), so after running this the channel count rises by 3.
⚠️ That means **existing checkpoints no longer match this store** — a 9-channel
c16 checkpoint will fail `load_state_dict` against a now-12-channel dataset. Pass
``include_static=False`` (or `eval_nowcast.py --no-static`) to re-evaluate an
older checkpoint.

    python -m tools.add_static_channels --zarr nowcast_mm_c15_0724_v2.zarr
    python -m tools.add_static_channels --zarr <z> --static ../data/static.npz
    python -m tools.add_static_channels --zarr <z> --force        # overwrite
    python -m tools.add_static_channels --zarr <z> --dry-run      # report only
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.add_static_channels")

# The terrain channels we actually want as model inputs. build_static.py ALSO
# writes provenance/metadata into the same npz — grid_lat, grid_lon (both (H, W))
# and grid_pitch_km (shape (2,)). Iterating over `npz.files` blindly would turn
# the coordinate grids into input channels and trip over the 1-D pitch array, so
# the channel set is an explicit allow-list, not "whatever is in the file".
TERRAIN_KEYS = ("elevation_m", "landmask", "distance_to_coast_km")
# Recognised metadata keys — skipped silently rather than warned about.
META_KEYS = ("grid_lat", "grid_lon", "grid_pitch_km")


def _resolve_static(explicit: pathlib.Path | None) -> pathlib.Path:
    """Same search order as tools/build_seamless_zarr._resolve_static, but this tool
    exists only to add static channels, so a miss is a hard error, not a warning."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"--static {explicit} does not exist")
        return explicit
    repo = pathlib.Path(__file__).resolve().parents[1]
    for cand in (repo / "data" / "static.npz", repo / "model" / "static.npz"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "no static.npz found in data/ or model/ — build it first with "
        "`PLUVIO_GRID_N=<n> python -m model.build_static --out data/static.npz`")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", required=True)
    p.add_argument("--static", default="", help="path to static.npz; default = data/ then model/")
    p.add_argument("--keys", default=",".join(TERRAIN_KEYS),
                   help="comma list of static.npz keys to add as channels; default = the three "
                        f"terrain fields ({', '.join(TERRAIN_KEYS)}). The file also holds "
                        "grid_lat/grid_lon/grid_pitch_km metadata, which are NOT channels.")
    p.add_argument("--force", action="store_true", help="overwrite existing static_* arrays")
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    static = _resolve_static(pathlib.Path(args.static) if args.static else None)
    d = np.load(static)
    want = [k.strip() for k in args.keys.split(",") if k.strip()]
    absent = [k for k in want if k not in d.files]
    if absent:
        LOG.error("static.npz %s lacks requested keys %s (has: %s)",
                  static, absent, ", ".join(sorted(d.files)))
        return 1
    skipped = [k for k in d.files if k not in want]
    LOG.info("static.npz=%s | channels=%s | skipped=%s",
             static, ", ".join(want), ", ".join(sorted(skipped)) or "none")

    import zarr
    root = zarr.open_group(args.zarr, mode="r" if args.dry_run else "r+")
    H, W = root["opera_rate"].shape[1:]
    n = root["opera_rate"].shape[0]
    existing = set(root.array_keys())
    already = sorted(k for k in existing if k.startswith("static_"))
    LOG.info("zarr=%s n=%d grid=%dx%d existing static=%s", args.zarr, n, H, W, already or "none")

    if already and not args.force:
        LOG.error("static channels already present (%s) — use --force to overwrite. Aborting.",
                  ", ".join(already))
        return 1

    # A grid mismatch would write garbage channels that train fine and verify
    # nonsensically — the exact class of silent bug that cost us c15/c16. Fail hard.
    for k in want:
        if tuple(d[k].shape) != (H, W):
            LOG.error("%s:%s has shape %s but the zarr grid is %s — rebuild static.npz with "
                      "PLUVIO_GRID_N=%d.", static, k, d[k].shape, (H, W), H)
            return 1

    if args.dry_run:
        LOG.info("dry-run: would add %s (each %dx%d float32); dataset channels +%d",
                 ", ".join(f"static_{k}" for k in want), H, W, len(want))
        return 0

    for k in want:
        arr = np.asarray(d[k], dtype="float32")
        if not np.isfinite(arr).all():
            LOG.warning("static_%s contains non-finite values — zeroing them", k)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Constant channels carry no signal and silently waste a conv input — the
        # dead-li_flash failure mode from c15. Warn rather than abort (a landmask
        # over an all-land domain is legitimately constant).
        if float(arr.min()) == float(arr.max()):
            LOG.warning("static_%s is CONSTANT (%.4g) — it will contribute nothing",
                        k, float(arr.min()))
        root.create_dataset(f"static_{k}", shape=(H, W), dtype="float32", overwrite=True)[:] = arr
        LOG.info("  static_%-24s range [%.3g, %.3g] mean %.3g",
                 k, float(arr.min()), float(arr.max()), float(arr.mean()))

    LOG.info("done: added %d static channels to %s — dataset channel count is now +%d "
             "(existing checkpoints need include_static=False to re-evaluate)",
             len(want), args.zarr, len(want))
    return 0


if __name__ == "__main__":
    sys.exit(main())
