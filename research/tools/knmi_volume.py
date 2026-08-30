"""Read KNMI's own radar volume HDF5 (not ODIM) and convert to a rain-rate grid.

KNMI publishes full polar volumes for Herwijnen and Den Helder through the open-data
API with an archive reaching back to 2019 — far deeper than the OPERA 24-h rolling
cache, which is the only source for most other radars. Herwijnen covers northern
Belgium, so this is what makes a MULTI-DAY Belgian evaluation possible at all.

The format is KNMI HDF5 v3.6, not ODIM, so tools/radar_single_site cannot read it:

  * groups are `scan1`..`scan16`, attributes prefixed `scan_` — there is no `/where`
  * elevation is `scan_elevation`, and **scan1 is the 90 deg birdbath**, so the sweeps
    must be sorted by angle. Three scans sit at 0.30 deg; the one with the most range
    bins is taken. Reading scan1 by position would composite a vertical profile as if
    it were surface rain — the same trap the French per-elevation files set.
  * data are uint16 needing the per-scan linear calibration in
    `calibration/calibration_Z_formulas`, e.g. `GEO=0.00193793*PV+-31.5019`

Unlike the OPERA feed's Belgian radars (TH and DBZH only), these volumes carry the full
dual-pol set — RhoHV, PhiDP, KDP, ZDR — so non-meteorological echo can be removed with
the textbook discriminator rather than a persistence heuristic. Ground clutter, insects
and chaff decorrelate the two polarisations; rain does not.
"""

from __future__ import annotations

import logging
import pathlib
import re

import numpy as np

LOG = logging.getLogger("pluvio.knmi_volume")

RHOHV_MIN = 0.80   # below this the echo is not meteorological


def _calib(group, name):
    """Parse `GEO=a*PV+b` into (a, b) for one moment."""
    raw = group["calibration"].attrs.get(f"calibration_{name}_formulas")
    if raw is None:
        return None
    txt = raw.decode() if isinstance(raw, bytes) else str(raw)
    m = re.match(r"GEO=([-\d.eE+]+)\*PV\+([-\d.eE+]+)", txt.strip())
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def read_lowest_sweep(path: pathlib.Path, rhohv_min: float = RHOHV_MIN):
    """Lowest-elevation sweep as (dbz, azimuths_deg, ranges_m, (lon, lat, alt), elangle).

    dbz is masked to NaN where RhoHV falls below rhohv_min, which removes clutter and
    other non-meteorological returns. Cells with no RhoHV are kept rather than dropped:
    the point is to remove what is provably not rain, not to require proof of rain.
    """
    import h5py

    with h5py.File(path, "r") as f:
        scans = [(float(f[k].attrs["scan_elevation"][0]), int(f[k].attrs["scan_number_range"][0]), k)
                 for k in f if k.startswith("scan") and "scan_elevation" in f[k].attrs]
        if not scans:
            raise RuntimeError(f"no scans in {path}")
        # Lowest angle wins; among equal angles prefer the longest range.
        scans.sort(key=lambda s: (s[0], -s[1]))
        el, _, key = scans[0]
        g = f[key]

        cal = _calib(g, "Z")
        if cal is None:
            raise RuntimeError(f"no Z calibration in {path}:{key}")
        a, b = cal
        raw = np.asarray(g["scan_Z_data"]).astype("float32")
        dbz = a * raw + b
        # PV=0 is "no echo detected", which is a MEASUREMENT (the radar looked and saw
        # nothing), not missing data. Encoding it as NaN made dry-but-scanned cells
        # indistinguishable from never-scanned ones, so composites covered only echo
        # and every dry gauge silently dropped out of evaluation — biasing FAR. It maps
        # to the calibration floor (~-31.5 dBZ), i.e. zero rain, and stays valid.
        dbz[raw == 0] = a * 0 + b

        rho_cal = _calib(g, "RhoHV")
        if rho_cal is not None and "scan_RhoHV_data" in g:
            ra, rb = rho_cal
            rraw = np.asarray(g["scan_RhoHV_data"]).astype("float32")
            rho = ra * rraw + rb
            bad = (rraw != 0) & (rho < rhohv_min)
            dbz[bad] = np.nan

        naz = int(g.attrs["scan_number_azim"][0])
        nrng = int(g.attrs["scan_number_range"][0])
        abin = float(g.attrs["scan_azim_bin"][0])
        rbin_km = float(g.attrs["scan_range_bin"][0])
        loc = f["radar1"].attrs["radar_location"]
        lon, lat = float(loc[0]), float(loc[1])
        name = f["radar1"].attrs.get("radar_name", b"").decode().lower()
        alt = ANTENNA_HEIGHT_M.get(
            "nlhrw" if "herwijnen" in name else "nldhl" if "helder" in name else "", 0.0)

    # Ray i is centred on i*abin: the data are stored from due north, and
    # scan_start_azim records where the ANTENNA began, not where row 0 sits.
    az = (np.arange(naz) * abin) % 360.0
    rng = (np.arange(nrng) + 0.5) * rbin_km * 1000.0
    return dbz, az, rng, (lon, lat, alt), el


API = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
DATASETS = {                       # radar -> (dataset, version, filename prefix)
    "nlhrw": ("radar_volume_full_herwijnen", "1.0", "RAD_NL62_VOL_NA_"),
    "nldhl": ("radar_volume_denhelder", "2.0", "RAD_NL61_VOL_NA_"),
}
CACHE = pathlib.Path("/mnt/storagebox/knmi_vol")

# KNMI's HDF5 records only lon/lat — there is no antenna height anywhere in the file
# (checked radar1, geographic, overview and the scan groups). Beam blockage needs it:
# defaulting to 0 m puts the beam at sea level, where any terrain clips it, and produced
# an absurd 65% "blocked" for Herwijnen, which stands in the flattest part of the
# Netherlands. These values are read from the ODIM /where/height published for the SAME
# radars through the OPERA single-site feed, so they are the operators' own figures
# rather than something looked up.
ANTENNA_HEIGHT_M = {"nlhrw": 25.0, "nldhl": 55.0}


def _api_key() -> str:
    for line in pathlib.Path("/opt/pluvio/research/.env").read_text().splitlines():
        if line.startswith("KNMI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("KNMI_API_KEY not found")


def fetch(radar: str, stamp: str) -> pathlib.Path | None:
    """Download one archived volume (cached). stamp = YYYYmmddTHHMM.

    The archive reaches back to 2019, which is the whole point of this path: the OPERA
    single-site feed is a 24-h rolling cache that cannot be backfilled, so without this
    every non-Dutch radar can only be evaluated on days we happened to be capturing.
    """
    import json
    import urllib.request

    ds, ver, prefix = DATASETS[radar]
    fn = f"{prefix}{stamp[:8]}{stamp[9:13]}.h5"
    dest = CACHE / radar / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    key = _api_key()
    try:
        req = urllib.request.Request(f"{API}/{ds}/versions/{ver}/files/{fn}/url",
                                     headers={"Authorization": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            url = json.load(r)["temporaryDownloadUrl"]
        tmp = dest.with_suffix(".part")     # never leave a half file that looks fetched
        with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        tmp.rename(dest)
        return dest
    except Exception as exc:
        LOG.warning("no KNMI volume %s %s (%s)", radar, stamp, exc)
        return None


def read_all_sweeps(path: pathlib.Path, max_elangle: float = 12.0):
    """Every sweep up to max_elangle, with the polarimetric moments RTCOR's chain needs.

    RTCOR does not use one low sweep. It merges all elevations by a per-voxel quality
    index, corrects attenuation from K_dp, and fits a vertical profile of reflectivity
    to extrapolate to the ground — none of which is possible from a single sweep. KNMI
    volumes carry 16 scans and the full dual-pol set, so the raw material is here.

    Returns a list of dicts sorted by elevation: dbz, dbz_v, zdr (= Z - Zv), rhohv, kdp,
    phidp, cpa (each (n_az, n_rng) or None), az, rng, elangle, site. The 90 deg birdbath is excluded
    since it carries no horizontal information.
    """
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        loc = f["radar1"].attrs["radar_location"]
        name = f["radar1"].attrs.get("radar_name", b"").decode().lower()
        site = (float(loc[0]), float(loc[1]), ANTENNA_HEIGHT_M.get(
            "nlhrw" if "herwijnen" in name else "nldhl" if "helder" in name else "", 0.0))
        for key in f:
            if not key.startswith("scan") or "scan_elevation" not in f[key].attrs:
                continue
            g = f[key]
            el = float(g.attrs["scan_elevation"][0])
            if el > max_elangle or el >= 89.0:
                continue
            naz = int(g.attrs["scan_number_azim"][0])
            nrng = int(g.attrs["scan_number_range"][0])
            moments = {}
            # ZDR is not stored: KNMI writes Z (horizontal) and Zv separately, so
            # differential reflectivity is Z - Zv. CPA (clutter phase alignment) is one
            # of the five fuzzy-logic clutter variables RTCOR uses and is only available
            # from the Dutch radars.
            for name_out, name_in in (("dbz", "Z"), ("dbz_v", "Zv"), ("rhohv", "RhoHV"),
                                      ("kdp", "KDP"), ("phidp", "PhiDP"), ("cpa", "CPA")):
                ds = f"scan_{name_in}_data"
                cal = _calib(g, name_in)
                if ds not in g or cal is None:
                    moments[name_out] = None
                    continue
                raw = np.asarray(g[ds]).astype("float32")
                val = cal[0] * raw + cal[1]
                if name_in == "Z" or name_in == "Zv":
                    val[raw == 0] = cal[1]        # no echo = calibration floor, valid
                else:
                    val[raw == 0] = np.nan        # polarimetric moments undefined there
                moments[name_out] = val
            if moments["dbz"] is None:
                continue
            moments["zdr"] = (moments["dbz"] - moments["dbz_v"]
                              if moments["dbz_v"] is not None else None)
            out.append(dict(
                **moments, elangle=el,
                az=(np.arange(naz) * float(g.attrs["scan_azim_bin"][0])) % 360.0,
                rng=(np.arange(nrng) + 0.5) * float(g.attrs["scan_range_bin"][0]) * 1000.0,
                site=site))
    # Several scans share 0.3 deg (different PRFs / ranges); keep the longest per angle.
    best = {}
    for s in out:
        k = round(s["elangle"], 2)
        if k not in best or len(s["rng"]) > len(best[k]["rng"]):
            best[k] = s
    return [best[k] for k in sorted(best)]
