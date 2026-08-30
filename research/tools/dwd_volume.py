"""Fetch and read DWD single-site sweeps (opendata.dwd.de).

DWD publishes standard ODIM HDF5 per site, per moment and per sweep, with a ~2-day
rolling window. That is shallower than KNMI's 2019 archive but deeper than the OPERA
single-site cache for the days it covers, and it brings Essen and Neuheilenbach — both
of which reach Belgium — with their full polarimetric moments.

⚠️ SWEEP NUMBERING IS NOT ELEVATION ORDER, and it is not monotonic either:

    _00 -> 5.50 deg    _03 -> 2.50    _06 ->  8.00
    _01 -> 4.50        _04 -> 1.50    _07 -> 12.00
    _02 -> 3.50        _05 -> 0.50    _08 -> 17.00, _09 -> 25.00

**Sweep 05 is the lowest**, not 00 and not 09. This is the third source in a row where
naive sweep indexing picks the wrong elevation — France puts a 90 deg birdbath at the
requested timestamp and KNMI puts one in `scan1`. Every one of them would have
composited a mid- or upper-level scan as if it were surface rain, so the lowest sweep is
resolved explicitly here rather than assumed from position.
"""

from __future__ import annotations

import logging
import pathlib
import re
import urllib.request

import numpy as np

LOG = logging.getLogger("pluvio.dwd_volume")

BASE = "https://opendata.dwd.de/weather/radar/sites"
CACHE = pathlib.Path("/mnt/storagebox/dwd_vol")
LOWEST_SWEEP = "05"                    # 0.50 deg — see the warning above
# The full DWD network. Names are prefixed "de" to match the OPERA single-site codes
# so a radar can be requested by one name regardless of which feed carries it.
SITES = {
    "deasb": ("asb", "10103"), "deboo": ("boo", "10132"), "dedrs": ("drs", "10488"),
    "deeis": ("eis", "10780"), "deess": ("ess", "10410"), "defbg": ("fbg", "10908"),
    "defld": ("fld", "10440"), "dehnr": ("hnr", "10339"), "deisn": ("isn", "10873"),
    "demem": ("mem", "10950"), "deneu": ("neu", "10557"), "denhb": ("nhb", "10605"),
    "deoft": ("oft", "10629"), "depro": ("pro", "10392"), "deros": ("ros", "10169"),
    "detur": ("tur", "10832"), "deumd": ("umd", "10356"),
}


_LISTING_CACHE: dict[str, str] = {}


def _listing(url: str) -> str:
    """Directory listing, cached per process.

    Each listing is ~1 MB and covers the whole 2-day window, so refetching it per file
    would move hundreds of MB to answer questions already in hand.
    """
    if url not in _LISTING_CACHE:
        with urllib.request.urlopen(url, timeout=90) as r:
            _LISTING_CACHE[url] = r.read().decode("utf-8", "replace")
    return _LISTING_CACHE[url]


def fetch(radar: str, stamp: str, moment: str = "dbzh",
          sweep: str = LOWEST_SWEEP, window_min: int = 10) -> pathlib.Path | None:
    """Download one sweep file nearest `stamp` (YYYYmmddTHHMM), cached.

    DWD filenames carry a second-resolution timestamp, and — as with the French
    per-elevation files — EACH SWEEP OF A VOLUME IS STAMPED SEPARATELY as the antenna
    reaches it. Sweep 05 of the scan that began at 16:05 is stamped 16:13. So there is
    no tidy 5-minute mark to reconstruct: the listing is parsed and the nearest file
    within `window_min` is chosen.
    """
    if radar not in SITES:
        return None
    site, wmo = SITES[radar]
    want = f"{stamp[:8]}{stamp[9:13]}"
    # DWD's layout differs per moment: filtered reflectivity sits under
    # hdf5/filter_polarimetric, while RhoHV and ZDR exist ONLY as unfiltered/ with a
    # "u" prefix on the moment name (urhohv, uzdr). filter_polarimetric 404s for them.
    if moment in ("rhohv", "zdr"):
        url = f"{BASE}/sweep_vol_{moment}/{site}/unfiltered/"
        moment = "u" + moment
    else:
        url = f"{BASE}/sweep_vol_z/{site}/hdf5/filter_polarimetric/"
    try:
        html = _listing(url)
    except Exception as exc:
        LOG.warning("DWD listing failed for %s (%s)", radar, exc)
        return None
    pat = re.compile(rf"ras[0-9A-Za-z_.\-]+_{moment}_{sweep}-(\d{{12}})\d*-{site}-{wmo}-hd5")
    cands = []
    for m in pat.finditer(html):
        ts = m.group(1)
        if ts[:8] != want[:8]:
            continue
        delta = abs((int(ts[8:10]) * 60 + int(ts[10:12]))
                    - (int(want[8:10]) * 60 + int(want[10:12])))
        if delta <= window_min:
            cands.append((delta, m.group(0)))
    if not cands:
        return None
    fn = min(cands)[1]
    dest = CACHE / radar / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    try:
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url + fn, timeout=180) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        if tmp.stat().st_size < 1000:      # an HTML error page, not a volume
            tmp.unlink()
            return None
        tmp.rename(dest)
        return dest
    except Exception as exc:
        LOG.warning("DWD fetch failed %s %s (%s)", radar, stamp, exc)
        return None


def read_sweep(path: pathlib.Path):
    """ODIM sweep -> (dbz, azimuths_deg, ranges_m, (lon, lat, alt), elangle)."""
    import h5py

    with h5py.File(path, "r") as f:
        w = f["where"].attrs
        lon, lat, alt = float(w["lon"]), float(w["lat"]), float(w["height"])
        dw = f["dataset1"]["where"].attrs
        el = float(dw["elangle"])
        nbins, nrays = int(dw["nbins"]), int(dw["nrays"])
        rscale, rstart = float(dw["rscale"]), float(dw["rstart"])
        a1gate = int(dw.get("a1gate", 0))
        d = f["dataset1"]["data1"]
        what = d["what"].attrs
        raw = np.asarray(d["data"]).astype("float32")
        dbz = float(what["offset"]) + float(what["gain"]) * raw
        dbz[raw == float(what.get("nodata", 65535))] = np.nan
        # undetect = scanned and saw nothing = dry, a valid measurement (see the same
        # fix in knmi_volume). Only for reflectivity; other moments stay NaN there.
        qty = what.get("quantity", b"DBZH")
        qty = qty.decode() if isinstance(qty, bytes) else str(qty)
        if qty.upper().endswith("DBZH") or qty.upper().endswith("TH"):
            dbz[raw == float(what.get("undetect", 0))] = -32.0
        else:
            dbz[raw == float(what.get("undetect", 0))] = np.nan

    # a1gate is the ray index that was sampled first; rays are stored in scan order,
    # so rotate back to make row 0 correspond to due north.
    if a1gate:
        dbz = np.roll(dbz, -a1gate, axis=0)
    az = (np.arange(nrays) * (360.0 / nrays)) % 360.0
    rng = rstart + (np.arange(nbins) + 0.5) * rscale
    return dbz, az, rng, (lon, lat, alt), el


def read_all_sweeps(radar: str, stamp: str, sweeps=("05", "04", "03", "02")):
    """Multi-sweep, multi-moment read for the chain — DWD's layout differs from KNMI's.

    DWD publishes one FILE per (moment, sweep): reflectivity under sweep_vol_z, RhoHV
    under sweep_vol_rhohv, ZDR under sweep_vol_zdr (all captured by fetch_dwd_sweeps.sh).
    This joins them into the same list-of-dicts contract knmi_volume.read_all_sweeps
    provides, so tools/rtcor_chain.py can process German radars unchanged. No K_dp and
    no CPA are published, so attenuation falls back to none (the modified-Kraemer method
    the paper uses for German radars is the TODO here) and the fuzzy classifier runs
    with the CPA weight zeroed.
    """
    out = []
    for sweep in sweeps:
        base = fetch(radar, stamp, moment="dbzh", sweep=sweep)
        if base is None:
            continue
        dbz, az, rng, site, el = read_sweep(base)
        entry = dict(dbz=dbz, dbz_v=None, zdr=None, rhohv=None, kdp=None,
                     phidp=None, cpa=None, az=az, rng=rng, elangle=el, site=site)
        for name, moment in (("rhohv", "rhohv"), ("zdr", "zdr")):
            p = fetch(radar, stamp, moment=moment, sweep=sweep)
            if p is None:
                continue
            try:
                m, maz, mrng, _, _ = read_sweep(p)
            except Exception:
                continue
            if m.shape == dbz.shape:
                entry[name] = m
        out.append(entry)
    out.sort(key=lambda s: s["elangle"])
    return out
