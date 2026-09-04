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
from .model import band_provenance, model_band
from .stubs import stub_band  # noqa: F401  (kept; model_band falls back to it)

LOG = logging.getLogger("pluvio.worker")

# A reference to the inference function for each band. `model_band` serves the
# trained UNet for the nowcast band and transparently falls back to the KMI stub
# (for the longer bands, and whenever the model field is missing/stale). The
# third return value is the GridSpec the rates were actually produced on —
# an npz's own `bounds`/shape when it carries one (1.13), else the `grid`
# argument it was called with.
BandInference = Callable[
    [httpx.Client, str, GridSpec, schedules.BandName],
    tuple[np.ndarray, datetime, GridSpec],
]


def run_tick(band_name: schedules.BandName, infer: BandInference = model_band) -> dict:
    """One refresh of one band. Returns a small summary dict."""
    settings = get_settings()
    cache = ForecastCache(settings.cache_root)
    grid = cache.grid

    LOG.info("tick band=%s starting", band_name)
    with httpx.Client() as client:
        rates, issued_at, used_grid = infer(client, settings.kmi_base_url, grid, band_name)

    snap = _reuse_or_new_snapshot(cache, issued_at, band_name)

    cache.write_band(snap, band_name, rates, grid=used_grid)
    cache.write_overlays(snap, band_name, rates, grid=used_grid)

    # Fold the freshest data for *every* band into this snapshot's point index:
    # the band we just wrote, plus the most recent prior snapshot for the others
    # (bands refresh on different cadences). This lets the API serve the full
    # horizon — nowcast → 10-day long-range — from one published snapshot.
    all_bands: dict[schedules.BandName, np.ndarray] = {band_name: rates}
    for b in schedules.all_bands():
        if b.name == band_name:
            continue
        arr = cache.read_band_any(b.name)
        if arr is not None:
            all_bands[b.name] = arr

    # write_point_shards/write_sprite fold every band into one array/index
    # keyed by the cache's default grid — a band produced on its OWN, larger
    # grid (a v3/full-Benelux npz, before 1.9 aligns the cache's default to
    # match) can't be mixed in there without a shape mismatch. Serve it as
    # its own band + overlays (already written above, on `used_grid`) but
    # leave the uniform-grid point/sprite folding to bands that actually
    # share the cache's grid; full multi-grid folding is 1.9's job.
    shard_bands = {
        name: arr for name, arr in all_bands.items() if arr.shape[-2:] == cache.grid.shape
    }
    skipped = sorted(set(all_bands) - set(shard_bands))
    if skipped:
        LOG.warning(
            "tick band=%s: %s not on the cache grid %s — excluded from point shards/sprite",
            band_name,
            skipped,
            cache.grid.shape,
        )
    if shard_bands:
        cache.write_point_shards(snap, shard_bands)

    # One sprite-sheet PNG of every frame so the client animates the whole
    # horizon with a single download (no per-frame requests).
    sprite = cache.write_sprite(snap, shard_bands) if shard_bands else None

    # Fold per-band provenance (source tag + confidence, from the forecast cube)
    # into grid.json so the product can honestly label each horizon and widen
    # its uncertainty band with lead. None when a band is stub-served.
    provenance = {b: p for b in all_bands if (p := band_provenance(b)) is not None}
    extras: dict = {"sprite": sprite}
    if provenance:
        extras["provenance"] = provenance
    # grid.json carries ONE footprint, and the client places every overlay by
    # it, so record the nowcast band's — the band a published snapshot must
    # carry, and the one served from the npz's own grid. A later tick for
    # another band keeps whatever this snapshot already recorded rather than
    # stamping the cache's default over it (1.9 collapses the difference by
    # widening the cache grid to match).
    meta_grid = used_grid if band_name == "nowcast" else (cache.snapshot_grid(snap) or cache.grid)
    cache.write_grid_metadata(
        snap,
        model_version=settings.model_version,
        extras=extras,
        grid=meta_grid,
        band_grids={band_name: used_grid},
    )
    cache.mark_complete(snap, summary={"refreshed_band": band_name, "bands": sorted(all_bands)})

    # Only publish once the snapshot carries the nowcast band — the only
    # location-specific one. A longer-band-only snapshot would serve a single
    # uniform (stub) frame that reads identically everywhere; never expose that.
    published = "nowcast" in all_bands
    if published:
        cache.swap_latest(snap)
    else:
        LOG.warning(
            "tick band=%s: no nowcast band available yet — not publishing %s",
            band_name,
            snap.name,
        )

    removed = cache.prune(keep=24)
    summary = {
        "snapshot": snap.name,
        "band": band_name,
        "published": published,
        "bands": sorted(all_bands),
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
    if existing is not None:
        if existing.name.startswith(bucket_dt.strftime("%Y-%m-%dT%H-%M-")):
            return existing
        # Bands can carry different artifact issue times (the v2 nowcast npz
        # lags the full-horizon cube by its store-append latency). Never step
        # BACK to an older-named snapshot — that made `latest` ping-pong
        # between two dirs as bands alternated, churning sprite URLs. Join
        # the newer existing snapshot instead; only create a new dir when our
        # bucket genuinely moves time forward.
        try:
            existing_dt = datetime.strptime(existing.name, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)
            if existing_dt >= bucket_dt:
                return existing
        except ValueError:
            pass
    return cache.new_snapshot_dir(bucket_dt)


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
