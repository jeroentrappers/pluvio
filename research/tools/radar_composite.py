"""Multi-radar composite: merge single-site QPE grids into one field.

Sources are unified here because no single feed is enough. Each has a different reach
and a different packaging, and all three are needed to cover the domain:

    tools.knmi_volume   nlhrw, nldhl        archive back to 2019   (KNMI open data)
    tools.dwd_volume    deess, denhb        ~2-day rolling window  (opendata.dwd.de)
    tools.radar_single_site  BE/FR/rest     24-h rolling cache     (OPERA single-site)

The OPERA cache cannot be backfilled, so anything relying on it alone can only ever be
scored on days we happened to be capturing. The KNMI archive is what makes multi-day
verification possible at all.

⚠️ EVERY ONE of these sources hides the lowest sweep somewhere different, and getting it
wrong composites an upper-level or vertical scan as if it were surface rain:
FR stamps each elevation with its own time and put a 90 deg birdbath at the requested
minute; KNMI puts the birdbath in `scan1`; DWD numbers sweeps non-monotonically with the
lowest at `_05`, between 0.5 and 8.0 deg neighbours. Each reader resolves it explicitly.

Merge rule, measured on held-out days rather than assumed:

  1. **lowest beam wins** — where discs overlap take the radar whose beam centre is
     closest to the ground. `max` inflates rain in every overlap; `mean` blends a good
     near-range sample with a poor far-range one; `nearest radar` ignores elevation
     angle, and measurably lost.
  2. **consensus gate** — where more than one radar covers a cell, a single radar
     claiming rain against the others is not believed. This is what stops one bad radar
     leaking into the field: over the Netherlands the ungated composite scored BELOW its
     own best single radar (CSI 0.279 against 0.328) because Den Helder, a coastal site,
     contributed sea clutter and anaprop.
  3. **speckle removal** — a wet cell needs at least SPECKLE_MIN_NEIGHBOURS wet cells in
     its 3x3. Real rain is spatially coherent; isolated cells are noise. This is the
     single biggest gain: on held-out days it cut FAR from 0.729 to 0.599 at the trace
     threshold while POD fell only 0.835 -> 0.802.

Scored against rain gauges, never against OPERA, since OPERA is an estimate not truth.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.radar_composite")

MAX_RANGE_KM = 200.0
SPECKLE_MIN_NEIGHBOURS = 4     # chosen on TRAIN days; see gpu_results/composite_v2
WET_MM_H = 0.1

KNMI_RADARS = ("nlhrw", "nldhl")
from tools.dwd_volume import SITES as _DWD_SITES  # noqa: E402
DWD_RADARS = tuple(_DWD_SITES)


def beam_height_grid(site, bounds, shape, elangle_deg):
    """Beam-centre height (m AGL) over every grid cell, 4/3-earth refraction.

    The right merge criterion, and distance is not: two radars at equal range can sample
    very different heights if their lowest elevations differ (0.3 deg against 0.5 deg is
    common), and it is height above the ground that decides how close the measurement is
    to surface rain.
    """
    r = site_grid_distance(site, bounds, shape) * 1000.0
    ke = 4.0 / 3.0 * 6371000.0
    el = np.radians(elangle_deg)
    return np.sqrt(r ** 2 + ke ** 2 + 2 * r * ke * np.sin(el)) - ke


def site_grid_distance(site, bounds, shape):
    """Great-circle-ish distance (km) from a radar to every grid cell."""
    w, s, e, n = bounds
    h, wd = shape
    lon = np.linspace(w, e, wd)[None, :]
    lat = np.linspace(n, s, h)[:, None]
    dx = (lon - site[0]) * 111.32 * np.cos(np.radians(lat))
    dy = (lat - site[1]) * 111.32
    return np.sqrt(dx ** 2 + dy ** 2)


def wet_neighbours(grid, thr=WET_MM_H):
    """Count of wet cells among each cell's 8 neighbours."""
    w = (np.nan_to_num(grid, nan=0.0) > thr).astype("int8")
    p = np.pad(w, 1)
    total = sum(p[i:i + w.shape[0], j:j + w.shape[1]] for i in range(3) for j in range(3))
    return total - w


def read_radar(radar, stamp):
    """Raw lowest sweep from whichever feed carries this radar.

    Split out from radar_field so the blockage code can reuse the source dispatch
    without gridding a rain rate it does not need.
    """
    if radar in KNMI_RADARS:
        from tools import knmi_volume
        path = knmi_volume.fetch(radar, stamp)
        reader = knmi_volume.read_lowest_sweep
    elif radar in DWD_RADARS:
        from tools import dwd_volume
        path = dwd_volume.fetch(radar, stamp)
        reader = dwd_volume.read_sweep
    else:
        from tools.radar_single_site import find_volume, read_lowest_sweep
        path = find_volume(radar, stamp)
        reader = read_lowest_sweep
    if path is None:
        return None
    try:
        return reader(path)
    except Exception as exc:
        LOG.warning("  %-6s unreadable at %s (%s)", radar, stamp, exc)
        return None


def radar_field(radar, stamp, bounds, shape):
    """One radar's rain-rate grid, from whichever source carries it."""
    from tools.radar_single_site import dbz_to_rate, polar_to_grid

    got = read_radar(radar, stamp)
    if got is None:
        return None
    dbz, az, rng, site, el = got
    grid = polar_to_grid(dbz_to_rate(dbz), az, rng, site, shape, bounds,
                         elangle=el, max_beam_m=1e9)
    return grid, site, el


def build(stamp, radars, bounds, shape, max_range_km=MAX_RANGE_KM,
          consensus=True, speckle=SPECKLE_MIN_NEIGHBOURS):
    """Composite the named radars for one timestamp -> (field, provenance, used)."""
    grids, dists, beams, used = [], [], [], []
    for r in radars:
        got = radar_field(r, stamp, bounds, shape)
        if got is None:
            continue
        g, site, el = got
        d = site_grid_distance(site, bounds, shape)
        grids.append(np.where(d <= max_range_km, g, np.nan))
        dists.append(d)
        beams.append(np.where(d <= max_range_km, beam_height_grid(site, bounds, shape, el), np.inf))
        used.append(r)
        LOG.info("  %-6s el %.2f deg", r, el)
    if not used:
        return np.full(shape, np.nan, "float32"), np.full(shape, -1, "int8"), []

    stack = np.stack(grids)
    bstack = np.stack(beams)
    cov = np.isfinite(stack)
    n_cov = cov.sum(0)

    # 1. lowest beam wins
    pick = np.argmin(np.where(cov, bstack, np.inf), axis=0)
    out = np.take_along_axis(stack, pick[None], 0)[0]
    out = np.where(n_cov > 0, out, np.nan)
    prov = np.where(n_cov > 0, pick, -1).astype("int8")

    # 2. consensus: one radar cannot outvote the others where they also see the cell
    if consensus:
        votes = np.nansum(np.where(cov, stack > WET_MM_H, 0), axis=0)
        out = np.where((n_cov > 1) & (votes < n_cov), 0.0, out)

    # 3. speckle: isolated wet cells are noise, not rain
    if speckle:
        out = np.where(wet_neighbours(out) >= speckle, out,
                       np.where(np.isfinite(out), 0.0, np.nan))
    return out, prov, used


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", required=True)
    p.add_argument("--radars", default="nlhrw,nldhl")
    p.add_argument("--max-range-km", type=float, default=MAX_RANGE_KM)
    p.add_argument("--speckle", type=int, default=SPECKLE_MIN_NEIGHBOURS)
    p.add_argument("--no-consensus", action="store_true")
    p.add_argument("--grid-n", type=int, default=256)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))

    from model.geo import GRID, bbox
    bounds = bbox()
    radars = [r.strip() for r in args.radars.split(",") if r.strip()]
    comp, prov, used = build(args.time, radars, bounds, GRID, args.max_range_km,
                             consensus=not args.no_consensus, speckle=args.speckle)
    cov = np.isfinite(comp)
    LOG.info("composite: %.1f%% coverage, max %.2f mm/h, %d radars (%s)",
             100 * cov.mean(), np.nanmax(comp) if cov.any() else 0.0, len(used), ",".join(used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
