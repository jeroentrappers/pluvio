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


def find_volume(radar: str, stamp: str, param: str = "DBZH",
                window_min: int = 8) -> pathlib.Path | None:
    """Locate one radar/time/parameter volume in the date-partitioned archive.

    Globs only the specific day directory — the archive reaches millions of files
    on CIFS, so walking it is not an option (indexing 69k OPERA files already cost
    28 s before that lesson was learned).

    Two packaging conventions exist and they need different handling:

    * **All elevations in one file** (BE, NL, DE). The elevation token is a list,
      e.g. ``behel@...@0.3_0.5_0.8_...@DBZH.h5``. read_lowest_sweep picks the lowest
      sweep inside the file, so an exact timestamp match is enough.
    * **One file per elevation** (FR). The token is a single angle and EACH ELEVATION
      CARRIES ITS OWN TIMESTAMP, one minute apart:
      ``frave@20260830T0000@90.0@...`` is the 90 deg birdbath while
      ``frave@20260830T0004@0.4@...`` is the lowest sweep of the same volume.

    An exact-stamp lookup against the second convention returns whichever elevation
    happens to share that minute — for frave at 20260830T0730 that was the **90 degree
    vertical scan**, which would have been composited as if it were surface rain. So
    when the token is a single angle, search a +/-window_min window and take the
    genuine lowest elevation.
    """
    day = f"{stamp[0:4]}/{stamp[4:6]}/{stamp[6:8]}"
    want = _stamp_minutes(stamp)

    for cc in ("BE", "NL", "DE", "FR", "CH", "CZ"):
        base = VOLUMES / day / cc / radar
        # The parameter is NOT always the filename suffix: BE ends "@DBZH.h5" (one
        # param per file), NL bundles everything into "@CCORH_DBZH_..._ZDR.h5", FR
        # combines "@DBZH_TH_VRADH.h5". Match the parameter anywhere in the token.
        exact = [h for h in sorted(glob.glob(str(base / f"{radar}@{stamp}@*.h5")))
                 if param in pathlib.Path(h).stem.rsplit("@", 1)[-1].split("_")]
        per_elev = any(_single_elevation(h) is not None for h in exact)

        if exact and not per_elev:
            return pathlib.Path(exact[0])

        # Per-elevation packaging, or no exact hit at all: widen and take the lowest.
        cands = []
        for h in sorted(glob.glob(str(base / f"{radar}@{stamp[:9]}*@*.h5"))):
            stem = pathlib.Path(h).stem
            if param not in stem.rsplit("@", 1)[-1].split("_"):
                continue
            el = _single_elevation(h)
            if el is None:
                continue
            try:
                mins = _stamp_minutes(stem.split("@")[1])
            except (IndexError, ValueError):
                continue
            if abs(mins - want) <= window_min:
                cands.append((el, mins, h))
        if cands:
            cands.sort(key=lambda c: (c[0], abs(c[1] - want)))
            return pathlib.Path(cands[0][2])
        if exact:
            return pathlib.Path(exact[0])
    return None


def _stamp_minutes(stamp: str) -> int:
    """YYYYmmddTHHMM -> minutes since midnight."""
    return int(stamp[9:11]) * 60 + int(stamp[11:13])


def _single_elevation(path: str) -> float | None:
    """Elevation angle if the file holds ONE sweep, else None.

    ``frave@...@0.4@DBZH_TH_VRADH.h5`` -> 0.4, while the multi-sweep
    ``behel@...@0.3_0.5_0.8_...@DBZH.h5`` -> None because its token is a list.
    """
    parts = pathlib.Path(path).stem.split("@")
    if len(parts) < 3:
        return None
    try:
        return float(parts[2])
    except ValueError:
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


_GEOM_CACHE: dict = {}


def _polar_geometry(azimuths, ranges, site, elangle, grid_shape, bounds):
    """Georeferencing of one sweep geometry, computed once and cached.

    The polar geometry of a sweep — bin positions, grid row/col, beam heights, and the
    hole-fill mapping — depends only on (site, elangle, ray/bin layout, grid), never on
    the data, yet it was being recomputed for every field of every timestep: ~21
    spherical_to_xyz + reproject calls per radar per slot. Cached, a timestep costs a
    fancy-index.

    The hole-fill mapping exists for fine grids: at 1 km cells, 1-degree rays are ~3 km
    apart at 180 km range, so scatter-binning leaves radial NaN stripes in the outer
    disc. Those cells ARE observed (the beam sweeps over them) — leaving them NaN would
    corrupt the measured-dry semantics this pipeline depends on. Each in-disc cell maps
    to its nearest polar bin, accepted only within that bin's own footprint (azimuthal
    half-width at its range plus the cell half-diagonal), so the fill never invents
    coverage beyond the scan.
    """
    import wradlib.georef as georef
    from scipy.spatial import cKDTree

    lon0, lat0 = site[0], site[1]
    alt0 = site[2] if len(site) > 2 else 0.0
    key = (round(lon0, 5), round(lat0, 5), round(alt0, 1), round(float(elangle), 3),
           len(azimuths), len(ranges), float(ranges[0]), float(ranges[-1]),
           tuple(grid_shape), tuple(bounds))
    got = _GEOM_CACHE.get(key)
    if got is not None:
        return got

    xyz, crs = georef.spherical_to_xyz(ranges, azimuths, elangle, (lon0, lat0, alt0))
    lonlat = georef.reproject(xyz, src_crs=crs, trg_crs=georef.get_default_projection())
    lons = lonlat[..., 0].ravel()
    lats = lonlat[..., 1].ravel()
    heights = xyz[..., 2].ravel() - alt0

    w, s, e, n = bounds
    h, wd = grid_shape
    col = ((lons - w) / (e - w) * wd).astype("int64")
    row = ((n - lats) / (n - s) * h).astype("int64")
    inb = (col >= 0) & (col < wd) & (row >= 0) & (row < h)

    # Hole-fill: nearest polar bin per grid cell, in km-scaled coordinates.
    coslat = np.cos(np.radians(lat0))
    tree = cKDTree(np.column_stack([(lons - lon0) * 111.32 * coslat,
                                    (lats - lat0) * 111.32]))
    lon_c = w + (np.arange(wd) + 0.5) * (e - w) / wd
    lat_c = n - (np.arange(h) + 0.5) * (n - s) / h
    cx = ((lon_c[None, :] - lon0) * 111.32 * coslat).ravel()
    cy = np.repeat((lat_c - lat0) * 111.32, wd)
    cxg = np.repeat(cx.reshape(1, wd), h, 0).ravel()
    dist, bin_idx = tree.query(np.column_stack([cxg, cy]), k=1)
    r_km = np.broadcast_to(np.asarray(ranges)[None, :] / 1000.0,
                           (len(azimuths), len(ranges))).ravel()[bin_idx]
    cell_km = max((e - w) * 111.32 * coslat / wd, (n - s) * 111.32 / h)
    az_halfwidth = r_km * np.pi / max(len(azimuths), 1)
    accept = dist <= (az_halfwidth + 0.75 * cell_km + 0.15)
    fill_cells = np.where(accept)[0].astype("int64")
    fill_bins = bin_idx[accept].astype("int64")

    got = dict(row=row, col=col, inb=inb, heights=heights,
               fill_cells=fill_cells, fill_bins=fill_bins)
    if len(_GEOM_CACHE) > 128:      # runaway guard; geometries are few in practice
        _GEOM_CACHE.clear()
    _GEOM_CACHE[key] = got
    return got


def polar_to_grid(rate, azimuths, ranges, site, grid_shape, bounds, elangle=0.0,
                  max_beam_m=2000.0):
    """Georeference polar bins and bin them onto a regular lat/lon grid.

    wradlib handles the spherical geometry (earth curvature + 4/3 refraction), which
    is exactly the part that is easy to get subtly and silently wrong by hand; the
    result is cached per sweep geometry (see _polar_geometry).

    ⚠️ `spherical_to_xyz` returns METRES in an azimuthal-equidistant projection
    centred on the radar, NOT degrees — feeding those straight into a lat/lon
    bounding box silently drops every bin (observed: 0.0% grid coverage, no error).
    Reproject to EPSG:4326 explicitly. The elevation angle must be the sweep's real
    one too: at 300 km range, 0.0 vs 0.3 deg is several km of height difference and
    a correspondingly wrong ground position.

    ⚠️ Beam-height mask (max_beam_m): a radar beam CLIMBS with range — echo at 4 km
    altitude is not surface rain, and converting it with Marshall-Palmer paints light
    rain over the whole outer disc (measured: 17.75% wet area against OPERA's 3.02%).

    Cells with contributing bins get the bin mean; in-disc cells that fall between
    rays on fine grids get their nearest bin's value (within that bin's footprint);
    cells outside the scan stay NaN — genuinely unobserved, not dry.
    """
    g = _polar_geometry(azimuths, ranges, site, elangle, grid_shape, bounds)
    h, wd = grid_shape
    vals = np.asarray(rate).ravel()
    ok = np.isfinite(vals) & (g["heights"] <= max_beam_m) & g["inb"]

    acc = np.zeros((h, wd), "f8")
    cnt = np.zeros((h, wd), "i8")
    np.add.at(acc, (g["row"][ok], g["col"][ok]), vals[ok])
    np.add.at(cnt, (g["row"][ok], g["col"][ok]), 1)
    out = np.full((h, wd), np.nan, "f4")
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]

    # fill in-disc holes from the nearest bin, respecting the same masks
    fc, fb = g["fill_cells"], g["fill_bins"]
    need = ~hit.ravel()[fc]
    good = np.isfinite(vals[fb]) & (g["heights"][fb] <= max_beam_m)
    sel = need & good
    out.ravel()[fc[sel]] = vals[fb[sel]]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", required=True, help="volume timestamp, e.g. 20260830T1100")
    p.add_argument("--radar", default="bejab")
    p.add_argument("--compare-opera", action="store_true",
                   help="score against the OPERA RATE composite at the same instant")
    p.add_argument("--max-beam-m", type=float, default=2000.0,
                   help="discard bins whose beam centre is above this height AGL. The "
                        "beam climbs with range, so far bins sample cloud rather than "
                        "surface rain.")
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
    grid = polar_to_grid(rate, az, rng, site, GRID, bbox(), elangle=el,
                         max_beam_m=args.max_beam_m)
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


def read_all_sweeps(path: pathlib.Path, max_elangle: float = 12.0):
    """Every sweep of an ODIM polar volume, in the rtcor_chain contract.

    The Belgian, French and other OPERA-feed radars ship ODIM files with datasets
    dataset1..datasetN, one per elevation (BE bundles all elevations in one file per
    parameter). This mirrors knmi_volume.read_all_sweeps so tools/rtcor_chain.py can
    process those radars unchanged. Only the file's own parameter is available —
    typically DBZH — so the polarimetric entries are None and the chain falls back
    accordingly (no fuzzy classification, no K_dp attenuation).

    `undetect` is a valid dry measurement and maps to the calibration floor; only
    `nodata` becomes NaN (same reasoning as in knmi_volume/dwd_volume: encoding dry as
    NaN removed every dry cell from evaluation and biased FAR).
    """
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        w = f["where"].attrs
        site = (float(w["lon"]), float(w["lat"]), float(w.get("height", 0.0)))
        for key in sorted(k for k in f if k.startswith("dataset")):
            g = f[key]
            try:
                dw = g["where"].attrs
                el = float(dw["elangle"])
            except Exception:
                continue
            if el > max_elangle or el >= 89.0:
                continue
            nbins, nrays = int(dw["nbins"]), int(dw["nrays"])
            rscale, rstart = float(dw["rscale"]), float(dw.get("rstart", 0.0))
            a1gate = int(dw.get("a1gate", 0))
            data = g.get("data1")
            if data is None:
                continue
            what = data["what"].attrs
            qty = what.get("quantity", b"")
            qty = qty.decode() if isinstance(qty, bytes) else str(qty)
            if qty not in ("DBZH", "TH"):
                continue
            raw = np.asarray(data["data"]).astype("float32")
            gain, offset = float(what.get("gain", 1.0)), float(what.get("offset", 0.0))
            dbz = offset + gain * raw
            dbz[raw == float(what.get("nodata", 255))] = np.nan
            dbz[raw == float(what.get("undetect", 0))] = offset  # dry, valid
            # ⚠️ Do NOT roll by a1gate. ODIM stores rays already north-aligned (row i =
            # azimuth i·360/nrays); `a1gate` only records where the antenna HAPPENED
            # to start acquiring. bejab's a1gate varies per scan (136 then 298), and
            # rolling rotated each scan by a different random angle: inter-scan
            # correlation 0.471 unrolled vs −0.101 rolled — the entire "93–98% of
            # bejab cells flip at every intensity" pathology was this line. The old
            # gauge-validated reader never rolled.
            _ = a1gate  # retained for provenance/debugging only
            out.append(dict(
                dbz=dbz, dbz_v=None, zdr=None, rhohv=None, kdp=None, phidp=None,
                cpa=None, elangle=el,
                az=(np.arange(nrays) * (360.0 / nrays)) % 360.0,
                rng=rstart + (np.arange(nbins) + 0.5) * rscale,
                site=site))
    best = {}
    for sw in out:
        k = round(sw["elangle"], 2)
        if k not in best or len(sw["rng"]) > len(best[k]["rng"]):
            best[k] = sw
    return [best[k] for k in sorted(best)]
