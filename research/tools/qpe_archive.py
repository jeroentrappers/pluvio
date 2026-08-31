"""Continuous QPE archive: composite raw volumes into a permanent store, then let the
raw data go.

The raw captures grow without bound (~35 GB/day for the OPERA single-site feed alone,
plus DWD sweeps and KNMI volumes) while the thing every downstream use actually needs —
serving history, verification, and retraining the nowcast on gauge-validated truth — is
the champion composite. One archived day at 1 km is ~150 MB compressed against ~35 GB
raw: a ~99% reduction, and more importantly a bounded steady state.

⚠️ Compositing is LOSSY. A better clutter classifier or VPR cannot be re-run on deleted
raw. Retention policy, deliberately conservative:

  * composites: kept forever (they are the product);
  * raw: kept RETAIN_DAYS (default 10) as a re-processing window;
  * deletion only for days whose archive coverage passes MIN_COVERAGE — a day the
    archiver missed keeps its raw until it is archived, indefinitely;
  * --prune is explicit; without it this never deletes anything.

Store layout: one zarr group per day, /mnt/storagebox/qpe/YYYY/MM/DD.zarr with
    rate     float16 (288, H, W)   mm/h, NaN = not covered by any radar
    quality  uint8   (288, H, W)   composite Q_r scaled 0-250
    n_radars uint8   (288,)        contributing radars per slot
    present  uint8   (288,)        1 = slot archived (idempotency)
on the research analysis grid (PLUVIO_GRID_N=768, ~1 km). Each timer tick archives the
newest missing stamps (bounded per run); a daily pass backfills stragglers before the
pruner may touch that day's raw.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pathlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("pluvio.qpe_archive")

ARCHIVE_ROOT = pathlib.Path(os.environ.get("PLUVIO_QPE_ROOT", "/mnt/storagebox/qpe"))
RADARS = tuple(os.environ.get(
    "PLUVIO_QPE_RADARS",
    "nlhrw,nldhl,behel,bejab,bewid,deess,denhb,deasb").split(","))
RETAIN_DAYS = int(os.environ.get("PLUVIO_QPE_RETAIN_DAYS", "5"))
MIN_COVERAGE = float(os.environ.get("PLUVIO_QPE_MIN_COVERAGE", "0.90"))
SLOTS_PER_DAY = 288                       # 5-min cadence
MIN_RADARS = 3
LAG_MIN = 15                              # freshest stamp worth attempting

# True raw: unrecoverable once the upstream rolling window moves on. Retention =
# RETAIN_DAYS. Measured 2026-08-31: radar_volumes 56 GB/day, dwd_vol ~11 GB/day —
# at 67 GB/day a 10-day window (~670 GB) does not fit the 1 TB box, 5 days does.
RAW_STORES = (                            # (root, layout) — see prune_raw
    (pathlib.Path("/mnt/storagebox/radar_volumes"), "daydir"),
    (pathlib.Path("/mnt/storagebox/dwd_vol"), "stampfile"),
)
# Cache: re-downloadable from upstream at any time (KNMI archives to 2019), so it gets
# a much shorter window. Measured at 107 GB after one evaluation campaign.
CACHE_STORES = ((pathlib.Path("/mnt/storagebox/knmi_vol"), "stampfile"),)
CACHE_RETAIN_DAYS = int(os.environ.get("PLUVIO_QPE_CACHE_RETAIN_DAYS", "2"))


def _env():
    os.environ["PLUVIO_GRID_N"] = "768"
    os.environ.setdefault("PLUVIO_SWEEP_MERGE", "local")
    os.environ.setdefault("PLUVIO_LOCAL_DH", "1200")


def _slot(stamp: str) -> int:
    return (int(stamp[9:11]) * 60 + int(stamp[11:13])) // 5


def _one(args):
    """Worker: one stamp -> (stamp, rate f16, quality u8, n_radars) or Nones."""
    stamp, radars = args
    import warnings
    warnings.filterwarnings("ignore")
    _env()
    from model.geo import GRID, bbox
    from tools.radar_single_site import polar_to_grid
    from tools import rtcor_chain as rc

    bounds = bbox()
    per = []
    for r in radars:
        try:
            sw = rc.read_sweeps_any(r, stamp)
            if sw:
                per.append(rc.single_radar_h(sw, GRID, bounds, polar_to_grid))
        except Exception:
            pass
    if len(per) < MIN_RADARS:
        return stamp, None, None, len(per)
    rate, q = rc.composite_by_height(per, GRID)
    q8 = np.clip(np.nan_to_num(q, nan=0.0) * 250.0, 0, 250).astype("uint8")
    return stamp, rate.astype("float16"), q8, len(per)


def _open_day(day: dt.date, shape):
    import zarr

    path = ARCHIVE_ROOT / f"{day:%Y/%m}" / f"{day:%d}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="a")
    have = set(getattr(root, "array_keys", lambda: [])())
    def create(name, shp, dtype, chunks, fill):
        if name in have:
            return root[name]
        try:                                # zarr 3
            return root.create_array(name, shape=shp, dtype=dtype,
                                     chunks=chunks, fill_value=fill)
        except (AttributeError, TypeError):  # zarr 2
            return root.create_dataset(name, shape=shp, dtype=dtype,
                                       chunks=chunks, fill_value=fill)
    h, w = shape
    return path, {
        "rate": create("rate", (SLOTS_PER_DAY, h, w), "float16", (12, h, w), np.nan),
        "quality": create("quality", (SLOTS_PER_DAY, h, w), "uint8", (12, h, w), 0),
        "n_radars": create("n_radars", (SLOTS_PER_DAY,), "uint8", (SLOTS_PER_DAY,), 0),
        "present": create("present", (SLOTS_PER_DAY,), "uint8", (SLOTS_PER_DAY,), 0),
    }


def day_coverage(day: dt.date) -> float:
    import zarr

    path = ARCHIVE_ROOT / f"{day:%Y/%m}" / f"{day:%d}.zarr"
    if not path.exists():
        return 0.0
    try:
        root = zarr.open_group(str(path), mode="r")
        return float(np.asarray(root["present"][:]).mean())
    except Exception:
        return 0.0


def archive(day: dt.date, max_stamps: int, workers: int) -> int:
    """Composite up to max_stamps missing slots of `day` into the archive."""
    _env()
    from model.geo import GRID

    path, arrs = _open_day(day, GRID)
    present = np.asarray(arrs["present"][:])
    now = dt.datetime.now(dt.UTC)
    stamps = []
    for s in range(SLOTS_PER_DAY - 1, -1, -1):          # newest first
        if present[s]:
            continue
        t = dt.datetime(day.year, day.month, day.day, s * 5 // 60, s * 5 % 60,
                        tzinfo=dt.UTC)
        if now - t < dt.timedelta(minutes=LAG_MIN):
            continue
        stamps.append(t.strftime("%Y%m%dT%H%M"))
        if len(stamps) >= max_stamps:
            break
    if not stamps:
        return 0
    done = 0
    with ProcessPoolExecutor(max_workers=min(workers, len(stamps))) as ex:
        for stamp, rate, q8, nrad in ex.map(_one, [(s, list(RADARS)) for s in stamps]):
            i = _slot(stamp)
            if rate is None:
                # too few radars — mark present so we do not retry forever, with n_radars
                # recording why (prune still guards on coverage of USABLE frames below).
                arrs["n_radars"][i] = nrad
                arrs["present"][i] = 1 if nrad == 0 else 0  # 0 radars: dead slot; else retry later
                continue
            arrs["rate"][i] = rate
            arrs["quality"][i] = q8
            arrs["n_radars"][i] = nrad
            arrs["present"][i] = 1
            done += 1
    LOG.info("%s: archived %d/%d attempted (coverage now %.1f%%)",
             day, done, len(stamps), 100 * day_coverage(day))
    return done


def prune_raw(today: dt.date) -> None:
    """Delete raw volumes for days old enough AND fully archived. Explicit only."""
    for age in range(RETAIN_DAYS, RETAIN_DAYS + 30):
        day = today - dt.timedelta(days=age)
        cov = day_coverage(day)
        daydir = RAW_STORES[0][0] / f"{day:%Y/%m/%d}"
        if not daydir.exists() and not any(
                _stampfiles(root, day) for root, kind in RAW_STORES[1:] if kind == "stampfile"):
            continue
        if cov < MIN_COVERAGE:
            LOG.warning("keeping raw for %s: archive coverage %.1f%% < %.0f%%",
                        day, 100 * cov, 100 * MIN_COVERAGE)
            continue
        if daydir.exists():
            LOG.info("pruning %s (%s archived %.1f%%)", daydir, day, 100 * cov)
            shutil.rmtree(daydir, ignore_errors=True)
        for root, kind in RAW_STORES[1:]:
            if kind != "stampfile":
                continue
            for f in _stampfiles(root, day):
                f.unlink(missing_ok=True)
    # cache stores: re-downloadable upstream, so age alone (plus the same archive
    # guard, out of caution) is enough
    for age in range(CACHE_RETAIN_DAYS, CACHE_RETAIN_DAYS + 60):
        day = today - dt.timedelta(days=age)
        if day_coverage(day) < MIN_COVERAGE:
            continue
        for root, _ in CACHE_STORES:
            n = 0
            for f in _stampfiles(root, day):
                f.unlink(missing_ok=True)
                n += 1
            if n:
                LOG.info("cache-pruned %d files for %s from %s", n, day, root)


def _stampfiles(root: pathlib.Path, day: dt.date):
    """Files in flat per-radar dirs whose name carries this day's date."""
    tokens = (f"{day:%Y%m%d}", f"-{day:%Y%m%d}")
    out = []
    if not root.exists():
        return out
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            name = f.name
            if any(tok in name for tok in tokens):
                out.append(f)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day", help="YYYY-MM-DD to (back)fill; default: today+yesterday catchup")
    p.add_argument("--max-stamps", type=int, default=8,
                   help="bound per run so the timer tick stays short")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prune", action="store_true",
                   help="after archiving, delete raw for old fully-archived days")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    today = dt.datetime.now(dt.UTC).date()
    if args.day:
        archive(dt.date.fromisoformat(args.day), args.max_stamps, args.workers)
    else:
        left = args.max_stamps
        for day in (today, today - dt.timedelta(days=1)):
            if left <= 0:
                break
            left -= archive(day, left, args.workers)
    if args.prune:
        prune_raw(today)
    return 0


if __name__ == "__main__":
    sys.exit(main())
