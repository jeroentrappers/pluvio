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
        dbz[raw == 0] = np.nan          # 0 is the no-data code in this profile

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
