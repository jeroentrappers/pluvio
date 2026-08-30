"""Single-radar ODIM polar volume → Cartesian rain rate, validated against OPERA.

Step 1 of building our own Belgian composite. Deliberately the *simplest* useful
piece, chosen to de-risk everything after it:

  * ONE radar — bejab (Jabbeke), on the flat Flemish coast, so beam blockage is
    negligible. Blockage is the hard part of this pipeline and Wideumont in the
    Ardennes is the worst case; proving the method on the easy radar first means a
    later blockage failure is unambiguous rather than confounded.
  * LOWEST elevation only — closest to the surface, what QPE conventionally uses.
  * NO clutter correction, NO gauge adjustment, NO multi-radar merging yet.

The output is scored against the OPERA RATE composite at the same instant on the
same grid, so "is the method sound?" gets a number rather than an opinion. We are
NOT trying to beat OPERA here — OPERA is gauge-free too, but it is professionally
quality-controlled and composited from many radars. Reproducing its broad structure
from one raw volume is the pass criterion; large systematic disagreement means the
geometry or Z-R is wrong.

Usage:
    python -m tools.radar_single_site --time 20260830T1100 --radar bejab
    python -m tools.radar_single_site --time 20260830T1100 --compare-opera
"""

from __future__ import annotations

import argparse
import glob
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.radar_single_site")

VOLUMES = pathlib.Path("/mnt/storagebox/radar_volumes")
OPERA = pathlib.Path("/mnt/storagebox/opera/RATE")

# Marshall-Palmer: Z = a * R^b  ->  R = (Z/a)^(1/b), Z in mm^6/m^3, R in mm/h.
# a=200, b=1.6 is the conventional stratiform pair and what most operational
# gauge-free QPE starts from. It UNDER-estimates convective cores (where a~300,
# b~1.4 fits better), which matters for us because heavy rain is the product's
# whole point — revisit once single-radar geometry is proven.
ZR_A, ZR_B = 200.0, 1.6

# Above this the echo is almost certainly hail/bright-band contamination rather
# than rain; capping avoids a handful of pixels dominating any comparison.
DBZ_CAP = 55.0


def find_volume(radar: str, stamp: str, param: str = "DBZH") -> pathlib.Path | None:
    """Locate one radar/time/parameter volume in the date-partitioned archive.

    Globs only the specific day directory — the archive reaches millions of files
    on CIFS, so walking it is not an option (indexing 69k OPERA files already cost
    28 s before that lesson was learned).
    """
    day = f"{stamp[0:4]}/{stamp[4:6]}/{stamp[6:8]}"
    for cc in ("BE", "NL", "DE", "FR", "CH", "CZ"):
        # Packaging differs per country and the parameter is NOT always the filename
        # suffix: BE ends "@DBZH.h5" (one param per file), NL bundles everything into
        # "@CCORH_DBZH_PHIDP_..._ZDR.h5", FR combines "@DBZH_TH_VRADH.h5". Match the
        # parameter anywhere in the token rather than assuming it terminates the name.
        hits = sorted(glob.glob(str(VOLUMES / day / cc / radar / f"{radar}@{stamp}@*.h5")))
        for h in hits:
            params = pathlib.Path(h).stem.rsplit("@", 1)[-1].split("_")
            if param in params:
                return pathlib.Path(h)
    return None


def read_lowest_sweep(path: pathlib.Path):
    """Return (dbz, azimuths, ranges_m, site_lonlatalt, elangle) for the lowest sweep.

    Geometry comes from each file's own ODIM `where` attributes rather than any
    assumed elevation table: the radars genuinely differ — behel runs 12 elevations
    (0.3-25 deg), bewid 9 (0.5-25), NL bundles 14 including a 90 deg vertical scan,
    and DE splits every elevation into its own file.
    """
    import h5py

    with h5py.File(path, "r") as f:
        site = f["where"].attrs
        lon, lat, alt = float(site["lon"]), float(site["lat"]), float(site["height"])

        best, best_el = None, None
        for key in (k for k in f.keys() if k.startswith("dataset")):
            el = float(f[key]["where"].attrs["elangle"])
            if best_el is None or el < best_el:
                best, best_el = key, el

        d = f[best]
        wh = d["where"].attrs
        nbins, nrays = int(wh["nbins"]), int(wh["nrays"])
        rscale, rstart = float(wh["rscale"]), float(wh.get("rstart", 0.0)) * 1000.0

        raw = d["data1"]["data"][:]
        dw = d["data1"]["what"].attrs
        gain, offset = float(dw["gain"]), float(dw["offset"])
        nodata, undetect = float(dw["nodata"]), float(dw["undetect"])

        dbz = raw.astype("f4") * gain + offset
        # `undetect` means "measured, no echo" -> genuinely dry (-inf dBZ, R=0).
        # `nodata` means "not measured" -> must stay NaN, or we invent dry pixels.
        dbz[raw == undetect] = -np.inf
        dbz[raw == nodata] = np.nan

        ranges = rstart + (np.arange(nbins) + 0.5) * rscale
        azimuths = np.arange(nrays) * (360.0 / nrays)
        return dbz, azimuths, ranges, (lon, lat, alt), best_el


def declutter(dbz: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove non-meteorological echo with wradlib's Gabella filter.

    Without this the lowest sweep is dominated by ground and sea clutter: measured
    on bejab 20260830T1115, the raw sweep produced 2.25% wet area within 100 km at
    ~0.07 mm/h while the quality-controlled OPERA composite showed 0.00% — i.e.
    essentially all of our "rain" near the radar was clutter. Jabbeke is coastal and
    the sweep is 0.3 deg, so both ground and sea returns are in the beam.

    Gabella works on the reflectivity texture (clutter is spatially incoherent
    compared with precipitation). Flagged bins become NaN = not measured, NOT 0,
    since "clutter here" tells us nothing about whether it is raining.
    """
    # wradlib 2.x moved this from wradlib.clutter to wradlib.classify.
    from wradlib.classify import filter_gabella

    work = np.where(np.isfinite(dbz), dbz, -32.0)  # -inf/NaN -> low dBZ for texture
    mask = np.asarray(filter_gabella(work, wsize=5, thrsnorain=0.0,
                                     tr1=6.0, n_p=6, tr2=1.3))
    out = dbz.copy()
    out[mask] = np.nan
    return out, float(mask.mean())


def dbz_to_rate(dbz: np.ndarray) -> np.ndarray:
    """dBZ -> mm/h via Marshall-Palmer, preserving the dry/missing distinction."""
    out = np.full(dbz.shape, np.nan, dtype="f4")
    dry = np.isneginf(dbz)
    wet = np.isfinite(dbz)
    z = 10.0 ** (np.clip(dbz[wet], None, DBZ_CAP) / 10.0)
    out[wet] = (z / ZR_A) ** (1.0 / ZR_B)
    out[dry] = 0.0
    return out


def polar_to_grid(rate, azimuths, ranges, site, grid_shape, bounds, elangle=0.0):
    """Georeference polar bins and bin them onto a regular lat/lon grid.

    wradlib handles the spherical geometry (earth curvature + 4/3 refraction), which
    is exactly the part that is easy to get subtly and silently wrong by hand.

    ⚠️ `spherical_to_xyz` returns METRES in an azimuthal-equidistant projection
    centred on the radar, NOT degrees — feeding those straight into a lat/lon
    bounding box silently drops every bin (observed: 0.0% grid coverage, no error).
    Reproject to EPSG:4326 explicitly. The elevation angle must be the sweep's real
    one too: at 300 km range, 0.0 vs 0.3 deg is several km of height difference and
    a correspondingly wrong ground position.
    """
    import wradlib.georef as georef

    lon0, lat0, alt0 = site
    xyz, crs = georef.spherical_to_xyz(ranges, azimuths, elangle, (lon0, lat0, alt0))
    lonlat = georef.reproject(xyz, src_crs=crs, trg_crs=georef.get_default_projection())
    lons = lonlat[..., 0].ravel()
    lats = lonlat[..., 1].ravel()
    vals = np.asarray(rate).ravel()

    w, s, e, n = bounds
    h, wd = grid_shape
    col = ((lons - w) / (e - w) * wd).astype("int64")
    row = ((n - lats) / (n - s) * h).astype("int64")
    ok = np.isfinite(vals) & (col >= 0) & (col < wd) & (row >= 0) & (row < h)

    # Mean of contributing bins per cell. Near the radar many bins fall in one cell;
    # far out, cells may get none and stay NaN (genuinely unobserved, not dry).
    acc = np.zeros((h, wd), "f8")
    cnt = np.zeros((h, wd), "i8")
    np.add.at(acc, (row[ok], col[ok]), vals[ok])
    np.add.at(cnt, (row[ok], col[ok]), 1)
    out = np.full((h, wd), np.nan, "f4")
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", required=True, help="volume timestamp, e.g. 20260830T1100")
    p.add_argument("--radar", default="bejab")
    p.add_argument("--compare-opera", action="store_true",
                   help="score against the OPERA RATE composite at the same instant")
    p.add_argument("--no-declutter", action="store_true",
                   help="skip the Gabella clutter filter (to show what it removes)")
    p.add_argument("--grid-n", type=int, default=256)
    p.add_argument("--max-range-km", type=float, default=100.0,
                   help="restrict the OPERA comparison to cells within this range of the "
                        "radar. Beyond ~100 km the 0.3 deg beam centre is >2 km above ground "
                        "and no longer samples surface rain, so comparing the full 300 km "
                        "disc against a multi-radar composite is not a like-for-like test.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    vol = find_volume(args.radar, args.time)
    if vol is None:
        LOG.error("no %s volume for %s in %s", args.radar, args.time, VOLUMES)
        return 2
    LOG.info("volume: %s", vol.name)

    dbz, az, rng, site, el = read_lowest_sweep(vol)
    finite = np.isfinite(dbz)
    LOG.info("sweep: elangle=%.2f deg  %d rays x %d bins  range=%.0f km  site=%.3fN %.3fE %.0fm",
             el, dbz.shape[0], dbz.shape[1], rng[-1] / 1000, site[1], site[0], site[2])
    LOG.info("dBZ: %.1f%% echo, %.1f%% dry, %.1f%% no-data | max %.1f dBZ",
             100 * finite.mean(), 100 * np.isneginf(dbz).mean(),
             100 * np.isnan(dbz).mean(), np.nanmax(dbz[finite]) if finite.any() else float("nan"))

    if not args.no_declutter:
        dbz, frac = declutter(dbz)
        LOG.info("declutter: %.2f%% of bins flagged as non-meteorological", 100 * frac)
    rate = dbz_to_rate(dbz)
    LOG.info("rain rate: max %.2f mm/h, mean(wet) %.3f mm/h",
             np.nanmax(rate), np.nanmean(rate[np.isfinite(rate) & (rate > 0)]) if (rate > 0).any() else 0.0)

    from model.geo import GRID, bbox
    import os
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))
    grid = polar_to_grid(rate, az, rng, site, GRID, bbox(), elangle=el)
    cov = np.isfinite(grid)
    LOG.info("gridded %s: %.1f%% cells covered, max %.2f mm/h", GRID, 100 * cov.mean(), np.nanmax(grid))

    if args.compare_opera:
        from model.nwp_regrid import reproject_to_analysis_grid
        day = f"{args.time[0:4]}/{args.time[4:6]}/{args.time[6:8]}"
        hits = sorted(glob.glob(str(OPERA / day / f"{args.time}_RATE.tif*")))
        if not hits:
            LOG.error("no OPERA RATE frame for %s", args.time)
            return 3
        op_raw = reproject_to_analysis_grid(pathlib.Path(hits[0]))
        # ⚠️ OPERA RATE encodes NO RAIN as nodata: the reprojected field is ~0.5%
        # finite and every finite cell is wet. NaN therefore means DRY, not missing —
        # the training builder makes the same call (_opera_clean: "keep NaN->0 for the
        # target (dry) — radar covers the whole domain"). Requiring isfinite(op) here
        # silently restricted the comparison to OPERA's rainy cells only, which made a
        # correct field look like a total miss (ours mean 0.000 vs opera 0.63).
        op = np.nan_to_num(op_raw, nan=0.0)

        # Distance mask: only where this single radar can actually see the surface.
        w, s_, e, n = bbox()
        hh, ww = GRID
        glon = np.linspace(w, e, ww)[None, :]
        glat = np.linspace(n, s_, hh)[:, None]
        dy = (glat - site[1]) * 111.32
        dx = (glon - site[0]) * 111.32 * np.cos(np.radians(site[1]))
        dist_km = np.sqrt(dx**2 + dy**2)
        near = dist_km <= args.max_range_km

        both = cov & near & ((grid > 0) | (op > 0))
        LOG.info("comparison limited to <=%.0f km of %s (%.1f%% of grid)",
                 args.max_range_km, args.radar, 100 * near.mean())
        if both.sum() < 50:
            LOG.warning("only %d comparable wet cells — too few to judge", both.sum())
            return 0
        a, b = grid[both], op[both]
        corr = float(np.corrcoef(a, b)[0, 1])
        LOG.info("--- vs OPERA on %d overlapping wet cells ---", both.sum())
        LOG.info("  ours  mean %.3f  max %.2f mm/h", a.mean(), a.max())
        LOG.info("  opera mean %.3f  max %.2f mm/h", b.mean(), b.max())
        LOG.info("  correlation %.3f | mean bias %+.3f mm/h | MAE %.3f",
                 corr, float((a - b).mean()), float(np.abs(a - b).mean()))
        LOG.info("  wet-area agreement: ours %.2f%%, opera %.2f%% of grid",
                 100 * ((grid > 0.1) & cov & near).sum() / max(near.sum(), 1),
                 100 * ((op > 0.1) & near).sum() / max(near.sum(), 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
