"""Multi-radar composite: merge single-site QPE grids into one field.

Builds on tools/radar_single_site (which proved the geometry and Z–R against gauges)
by compositing several radars onto the shared analysis grid.

Merge rule: **lowest beam wins**. Where discs overlap, the value is taken from the
radar whose beam centre is closest to the ground over that cell, rather than the
maximum, the mean, or the nearest site. The reasoning:

  * `max` biases high wherever discs overlap — it picks whichever radar happens to
    see the most, which systematically inflates rain in overlap regions and is the
    easiest way to manufacture a fake improvement in POD.
  * `mean` blends a good near-range sample with a poor far-range one, degrading the
    better measurement.
  * `nearest radar` looks equivalent but is not, and measurably hurt: at 20260830T0730
    it produced CSI 0.500 where nlhrw alone scored 0.625, because it handed cells to
    radars whose lowest sweep is 0.5 deg rather than 0.3 deg. Distance ignores
    elevation angle; beam height does not.

  A real operational chain would also weight by blockage and a quality index, which
  this does not yet do.

Scored the same way as the single-radar work: against KNMI rain gauges, never
against OPERA, since OPERA is an estimate rather than truth.

Usage:
    python -m tools.radar_composite --time 20260830T0730 --radars nlhrw,nldhl
    python -m tools.radar_composite --time 20260830T0730 --radars nlhrw,nldhl --compare
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.radar_composite")


def beam_height_grid(site, bounds, shape, elangle_deg):
    """Beam-centre height (m AGL) over every grid cell, 4/3-earth refraction.

    This is the right merge criterion, and distance is not. Measured 20260830T0730:
    a "nearest radar wins" composite scored CSI 0.500 while nlhrw ALONE scored 0.625
    — the merge handed cells to deess/denhb, whose lowest sweep is 0.5 deg against
    nlhrw's 0.3, so they sample a higher (worse) part of the storm and dragged the
    result below the best single radar. Beam height captures elevation angle AND
    range together, which is what actually determines sample quality.
    """
    r_km = site_grid_distance(site, bounds, shape)
    r = r_km * 1000.0
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


def build(stamp, radars, bounds, shape, max_range_km=200.0, max_beam_m=1e9):
    """Composite the named radars for one timestamp. Returns (field, provenance)."""
    from tools.radar_single_site import (find_volume, read_lowest_sweep, declutter,
                                         dbz_to_rate, polar_to_grid)

    best = np.full(shape, np.nan, "float32")
    best_q = np.full(shape, np.inf, "float32")   # lower beam = better sample
    prov = np.full(shape, -1, "int8")
    used = []
    for idx, r in enumerate(radars):
        vol = find_volume(r, stamp)
        if vol is None:
            LOG.warning("  %-6s no volume at %s", r, stamp)
            continue
        dbz, az, rng, site, el = read_lowest_sweep(vol)
        dbz, cfrac = declutter(dbz)
        g = polar_to_grid(dbz_to_rate(dbz), az, rng, site, shape, bounds,
                          elangle=el, max_beam_m=max_beam_m)
        d = site_grid_distance(site, bounds, shape)
        q = beam_height_grid(site, bounds, shape, el)
        # Accept only cells this radar can plausibly measure, and only where its beam
        # is LOWER than that of any radar already written there.
        take = np.isfinite(g) & (d <= max_range_km) & (q < best_q)
        best[take] = g[take]
        best_q[take] = q[take]
        prov[take] = idx
        used.append(r)
        LOG.info("  %-6s el %.2f deg, declutter %.1f%%, contributes %d cells",
                 r, el, 100 * cfrac, int(take.sum()))
    return best, prov, used


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", required=True)
    p.add_argument("--radars", default="nlhrw,nldhl")
    p.add_argument("--max-range-km", type=float, default=200.0)
    p.add_argument("--grid-n", type=int, default=256)
    p.add_argument("--compare", action="store_true",
                   help="score composite, each single radar, and OPERA against gauges")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))

    from model.geo import GRID, bbox
    bounds = bbox()
    radars = [r.strip() for r in args.radars.split(",") if r.strip()]

    LOG.info("compositing %s at %s", radars, args.time)
    comp, prov, used = build(args.time, radars, bounds, GRID, args.max_range_km)
    cov = np.isfinite(comp)
    LOG.info("composite: %.1f%% coverage, max %.2f mm/h, %d radars used",
             100 * cov.mean(), np.nanmax(comp) if cov.any() else 0.0, len(used))
    for i, r in enumerate(used):
        LOG.info("  provenance %-6s %.1f%% of covered cells",
                 r, 100 * (prov[cov] == i).mean() if cov.any() else 0)

    if not args.compare:
        return 0

    from model.nwp_regrid import reproject_to_analysis_grid
    from tools.gauge_validate import fetch_knmi_10min, read_gauges, sample, contingency
    from tools.radar_single_site import (find_volume, read_lowest_sweep, declutter,
                                         dbz_to_rate, polar_to_grid)

    gp = fetch_knmi_10min(args.time)
    if gp is None:
        LOG.error("no gauge file for %s", args.time)
        return 2
    gauges = read_gauges(gp)

    singles = {}
    for r in used:
        vol = find_volume(r, args.time)
        dbz, az, rng, site, el = read_lowest_sweep(vol)
        dbz, _ = declutter(dbz)
        singles[r] = polar_to_grid(dbz_to_rate(dbz), az, rng, site, GRID, bounds,
                                   elangle=el, max_beam_m=1e9)

    day = f"{args.time[0:4]}/{args.time[4:6]}/{args.time[6:8]}"
    oh = sorted(glob.glob(f"/mnt/storagebox/opera/RATE/{day}/{args.time}_RATE.tif*"))
    opera = (np.nan_to_num(reproject_to_analysis_grid(pathlib.Path(oh[0]),
                                                      nodata_as_zero=True), nan=0.0)
             if oh else None)

    fields = {"composite": comp, **singles}
    if opera is not None:
        fields["OPERA"] = opera

    obs, est = [], {k: [] for k in fields}
    for st, la, lo, o in gauges:
        v = {k: sample(f, la, lo, bounds, GRID, halo=1) for k, f in fields.items()}
        if not np.isfinite(v["composite"]):
            continue
        obs.append(o)
        for k in fields:
            est[k].append(v[k])
    obs = np.array(obs)
    if len(obs) < 10:
        LOG.error("only %d comparable gauges", len(obs))
        return 3
    LOG.info("=== %d gauges, %d wet ===", len(obs), int((obs > 0.1).sum()))
    for k in fields:
        v = np.nan_to_num(np.array(est[k]), nan=0.0)
        c = contingency(v, obs)
        LOG.info("%-10s POD %.3f FAR %.3f CSI %.3f | MAE %.3f corr %.3f",
                 k, c["pod"], c["far"], c["csi"], float(np.mean(np.abs(v - obs))),
                 float(np.corrcoef(v, obs)[0, 1]) if v.std() > 0 else float("nan"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
