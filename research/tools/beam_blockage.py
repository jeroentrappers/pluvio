"""Beam blockage quality index from a DEM — the missing term in our compositing.

EUMETNET documents ODYSSEY as weighting each contributing radar by "a quality index, the
distance from centre of the pixel and an exponential index related to inverse of the beam
altitude". We had the geometric half (beam altitude and distance) but no quality index,
and the geometric half ALONE loses to our winner-takes-all rule (CSI 0.248 against 0.365
on held-out days). The quality index is the part that should matter where terrain blocks
the beam — which is precisely the Ardennes, the one region where we still lose to OPERA.

This computes the cumulative beam blockage fraction with `wradlib.qual`, following the
standard approach: sample terrain along each ray, compare it to the beam centre and beam
width, and accumulate the blocked fraction outward in range. Quality is then 1 - CBB.

⚠️ DEM RESOLUTION. The only terrain we hold is the analysis grid's `elevation_m` at 256x256
over the domain — roughly 3 km per cell, against 250 m radar bins. That undersamples ridge
lines badly: it will capture the broad Ardennes and Eifel massifs but not the narrow
ridges that actually clip a low beam. Treat the result as a lower bound on blockage. A
real implementation uses SRTM at 30-90 m (wradlib.io has no get_srtm in 2.9.5, so this
would need fetching separately).
"""

from __future__ import annotations

import functools
import logging
import pathlib

import numpy as np

LOG = logging.getLogger("pluvio.beam_blockage")

DEM_NPZ = pathlib.Path("/opt/pluvio/radarproc/dem_500m.npz")   # tools/build_dem.py


@functools.lru_cache(maxsize=1)
def _dem():
    """Terrain height (m) at ~500 m, with its own bounds — NOT the analysis grid.

    The analysis grid's elevation_m is ~3 km and finds no blockage at all, which is a
    resolution artefact: 3 km terrain cannot clip a beam. tools/build_dem.py mosaics the
    open Copernicus 30 m COGs to something fine enough to see ridges.
    """
    d = np.load(DEM_NPZ)
    return np.asarray(d["dem"], dtype="float32"), tuple(float(x) for x in d["bounds"])


def _sample_dem(lats, lons):
    """Nearest-neighbour DEM lookup for arbitrary lat/lon arrays."""
    dem, (w, s, e, n) = _dem()
    h, wd = dem.shape
    r = np.clip(((n - lats) / (n - s) * h).astype(int), 0, h - 1)
    c = np.clip(((lons - w) / (e - w) * wd).astype(int), 0, wd - 1)
    return dem[r, c]


def blockage_polar(site, az_deg, rng_m, elangle_deg, bounds, shape,
                   beamwidth_deg: float = 1.0):
    """Cumulative beam blockage fraction on the radar's own polar grid.

    Returns an array shaped (n_az, n_rng) in [0, 1]; 0 is clear, 1 fully blocked.
    """
    import wradlib.georef as georef
    import wradlib.qual as qual

    lon0, lat0 = site[0], site[1]
    alt0 = site[2] if len(site) > 2 else 0.0

    # Ray geometry: 4/3-earth beam centre height, and the lat/lon each bin falls over.
    xyz, crs = georef.spherical_to_xyz(rng_m, az_deg, elangle_deg, (lon0, lat0, alt0))
    ll = georef.reproject(xyz, src_crs=crs, trg_crs=georef.get_default_projection())
    lons, lats = ll[..., 0], ll[..., 1]
    beam_h = xyz[..., 2]                     # metres above sea level

    terrain = _sample_dem(lats, lons)
    rr = np.broadcast_to(rng_m[None, :], beam_h.shape)
    beam_radius = rr * np.radians(beamwidth_deg) / 2.0   # half-power beam radius (m)

    pbb = qual.beam_block_frac(terrain, beam_h, beam_radius)
    pbb = np.ma.filled(np.ma.masked_invalid(pbb), 0.0)
    # Cumulative blockage is the running maximum of the partial blockage along the ray
    # (Bech et al. 2003): once the beam is clipped it stays clipped further out.
    # wradlib's cum_beam_block_frac wants a different array layout than our (az, range)
    # grid, and doing it directly is both clearer and faster.
    cbb = np.maximum.accumulate(pbb, axis=-1)
    return np.clip(np.nan_to_num(cbb, nan=0.0), 0.0, 1.0)


def quality_grid(radar, stamp, bounds, shape, beamwidth_deg: float = 1.0):
    """Beam-blockage quality (1 = clear) for one radar, on the analysis grid."""
    from tools.radar_composite import read_radar
    from tools.radar_single_site import polar_to_grid

    got = read_radar(radar, stamp)
    if got is None:
        return None
    dbz, az, rng, site, el = got
    cbb = blockage_polar(site, az, rng, el, bounds, shape, beamwidth_deg)
    q = polar_to_grid(1.0 - cbb, az, rng, site, shape, bounds,
                      elangle=el, max_beam_m=1e9)
    return np.nan_to_num(q, nan=0.0)
