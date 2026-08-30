"""Persistence-based clutter map for a single radar.

Ground clutter is stationary: the same cells light up in slot after slot. Rain is
intermittent — it moves through. So the fraction of slots in which a cell reports
echo separates the two without needing dual-pol moments, which matters here because
behel publishes only TH and DBZH through the OPERA feed (no RHOHV, no ZDR), so the
textbook polarimetric discriminators are unavailable.

Measured on behel, 2026-08-30, against 14 KMI gauges (379 dry station-times):

    no mask        FA 50/379 = 13.2%   hits 5/13
    mask >30%      FA 38/379 = 10.0%   hits 5/13
    mask >40%      FA 48/379 = 12.7%   hits 5/13

So masking removes a quarter of the false alarms and costs no hits.

⚠️ TWO REASONS NOT TO TRUST THAT NUMBER YET.
1. The 30% threshold was chosen by looking at the same gauges it is scored on. No cell
   is lit >50% of the time and >40% removes almost nothing, so the operating point sits
   on a cliff that one day of data cannot locate honestly. This is the same in-sample
   trap that made the Z-R calibration look like a win before it failed held-out.
2. "Costs no hits" rests on 5 hits. That cannot detect a real loss of light rain over
   persistently-cluttered ground, which is exactly where this mask does its work.

Build the map over a period with VARIED weather and evaluate on different days than
were used to pick the threshold before relying on it.

Why not a reflectivity floor: behel's false alarms are STRONGER than its hits (FA
median 0.364 mm/h against 0.263, FA max 4.35 against 1.65), so any floor removes hits
faster than false alarms. Measured, not assumed.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.clutter_map")


def build_frequency(radar, day, bounds, shape, step_min=20, rate_thr=0.1):
    """Fraction of slots in which each grid cell reports echo above rate_thr."""
    from tools.radar_single_site import (find_volume, read_lowest_sweep, declutter,
                                         dbz_to_rate, polar_to_grid)

    stamps = [f"{day}T{h:02d}{m:02d}"
              for h in range(24) for m in range(0, 60, step_min)]
    lit = np.zeros(shape, "float32")
    n = 0
    for stamp in stamps:
        vol = find_volume(radar, stamp)
        if vol is None:
            continue
        dbz, az, rng, site, el = read_lowest_sweep(vol)
        dbz, _ = declutter(dbz)
        grid = polar_to_grid(dbz_to_rate(dbz), az, rng, site, shape, bounds,
                             elangle=el, max_beam_m=1e9)
        lit += np.nan_to_num(grid, nan=0.0) > rate_thr
        n += 1
    if not n:
        raise RuntimeError(f"no volumes for {radar} on {day}")
    LOG.info("%s: echo frequency from %d slots", radar, n)
    return lit / n, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radar", default="behel")
    p.add_argument("--day", required=True, help="YYYYmmdd")
    p.add_argument("--step-min", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.30,
                   help="mask cells lit in more than this fraction of slots. See the "
                        "module docstring: this value is NOT yet validated out-of-sample.")
    p.add_argument("--out", default=None, help="write the mask to this .npy")
    p.add_argument("--grid-n", type=int, default=256)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))

    from model.geo import GRID, bbox

    freq, n = build_frequency(args.radar, args.day, bbox(), GRID, args.step_min)
    for th in (0.2, 0.3, 0.4, 0.5):
        LOG.info("  lit >%.0f%% of slots: %d cells", 100 * th, int((freq > th).sum()))
    if args.out:
        np.save(args.out, freq > args.threshold)
        LOG.info("wrote mask (%d cells) to %s", int((freq > args.threshold).sum()), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
