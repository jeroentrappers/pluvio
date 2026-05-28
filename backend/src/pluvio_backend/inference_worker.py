"""The scheduled worker that refreshes the forecast cache.

Two run modes:

  pluvio-worker tick --band nowcast        # one-shot, ideal for cron
  pluvio-worker schedule                   # in-process scheduler (APScheduler)

Cron mode is recommended in production. The scheduler is convenient for
local development and inside a single container.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import numpy as np

from . import schedules
from .cache import ForecastCache, GridSpec
from .config import get_settings
from .stubs import stub_band

LOG = logging.getLogger("pluvio.worker")

# A reference to the inference function for each band. Swap stub_band for
# the trained CorrDiff model when it lands — same signature.
BandInference = Callable[
    [httpx.Client, str, GridSpec, schedules.BandName],
    tuple[np.ndarray, datetime],
]


def run_tick(band_name: schedules.BandName, infer: BandInference = stub_band) -> dict:
    """One refresh of one band. Returns a small summary dict."""
    settings = get_settings()
    cache = ForecastCache(settings.cache_root)
    grid = cache.grid

    LOG.info("tick band=%s starting", band_name)
    with httpx.Client() as client:
        rates, issued_at = infer(client, settings.kmi_base_url, grid, band_name)

    snap = _reuse_or_new_snapshot(cache, issued_at, band_name)

    cache.write_band(snap, band_name, rates)
    cache.write_overlays(snap, band_name, rates)
    cache.write_grid_metadata(snap, model_version=settings.model_version)
    # `points` is built once we have at least one band on disk; rebuild it
    # every tick so the bucket file is always coherent across bands.
    all_bands = _collect_all_bands(cache, snap)
    cache.write_point_shards(snap, all_bands)
    cache.mark_complete(snap, summary={"refreshed_band": band_name})
    cache.swap_latest(snap)

    removed = cache.prune(keep=24)
    summary = {
        "snapshot": snap.name,
        "band": band_name,
        "n_leads": rates.shape[0],
        "max_mm_per_h": float(rates.max()),
        "pruned": removed,
    }
    LOG.info("tick band=%s done %s", band_name, summary)
    return summary


def _reuse_or_new_snapshot(
    cache: ForecastCache, issued_at: datetime, band_name: schedules.BandName
) -> pathlib.Path:
    """If a recent enough snapshot already exists, append our band to it.

    Each band runs on its own cadence, but they all live in the same
    snapshot directory so the API can read a consistent set. We "join" the
    current refresh window: bucket the issue time to the band's cadence.
    """
    # Bucket the issue time to the nowcast cadence — the smallest unit.
    bucket = (
        issued_at.replace(second=0, microsecond=0).timestamp()
        // schedules.band("nowcast").refresh_seconds
    )
    bucket_dt = datetime.fromtimestamp(bucket * schedules.band("nowcast").refresh_seconds, tz=UTC)
    existing = cache.latest_snapshot()
    if existing is not None and existing.name.startswith(bucket_dt.strftime("%Y-%m-%dT%H-%M-")):
        return existing
    return cache.new_snapshot_dir(bucket_dt)


def _collect_all_bands(
    cache: ForecastCache, snap: pathlib.Path
) -> dict[schedules.BandName, np.ndarray]:
    """Read whichever band zarrs exist in `snap`. Missing bands are skipped."""
    import zarr

    out: dict[schedules.BandName, np.ndarray] = {}
    bands_dir = snap / "bands"
    if not bands_dir.exists():
        return out
    for band in schedules.all_bands():
        path = bands_dir / f"{band.name}.zarr"
        if path.exists():
            out[band.name] = np.asarray(zarr.open_array(store=str(path), mode="r")[:])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tick = sub.add_parser("tick", help="Refresh a single band and exit.")
    tick.add_argument("--band", choices=sorted(schedules.BANDS.keys()), required=True)

    schedule = sub.add_parser(
        "schedule", help="Run all bands on a long-lived in-process scheduler."
    )
    schedule.add_argument("--bands", default=",".join(schedules.BANDS.keys()))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "tick":
        run_tick(args.band)
        return 0

    if args.command == "schedule":
        return _run_scheduler(args.bands.split(","))

    raise SystemExit(2)


def _run_scheduler(band_names: list[str]) -> int:
    """Long-running APScheduler loop. Use cron in production instead."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler(timezone="UTC")
    for name in band_names:
        if name not in schedules.BANDS:
            LOG.warning("unknown band %s; skipping", name)
            continue
        band = schedules.band(name)
        trigger = CronTrigger.from_crontab(band.cron_expression, timezone="UTC")
        sched.add_job(run_tick, trigger=trigger, args=[name], name=f"band-{name}")
        LOG.info("scheduled band=%s cron=%s", name, band.cron_expression)

    # Kick every band once at startup so we land in a coherent state.
    for name in band_names:
        if name in schedules.BANDS:
            try:
                run_tick(name)
            except Exception:
                LOG.exception("startup tick %s failed", name)

    sched.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
