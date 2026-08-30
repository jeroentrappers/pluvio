"""KNMI RTCOR — the Dutch operational radar composite, as a best-in-class benchmark.

OPERA is the pan-European composite and is not the strongest product over any single
country: national services run their own, with their own QC and gauge adjustment, and
those are what a user in the Netherlands or Belgium actually compares against (RTCOR is
what sits behind Buienradar). Beating OPERA is therefore a weaker claim than it sounds,
and this exists so the harder comparison can be made.

`nl_rdr_data_rtcor_5m_tar` ships one tar per day holding 288 five-minute composites, and
the archive is deep. Each member is KNMI HDF5 v3.6 (not ODIM):

  * 765 x 700 grid, 1 km pixels, polar-stereographic
    `+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378137 +b=6356752 +units=km`
  * `image1/image_data` is uint16 with `GEO=0.010000*PV+0.000000` giving **mm per 5 min**,
    so rain rate is value * 12
  * 65534 marks missing and 65535 out-of-image — both must be dropped, and 65535 in
    particular is NOT zero rain, it is outside the product's footprint

⚠️ The daily tar runs 08:05 on its start date to 08:00 the next, not midnight to midnight,
so a calendar day spans two tars.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import logging
import pathlib
import tarfile
import urllib.request

import numpy as np

LOG = logging.getLogger("pluvio.knmi_rtcor")

API = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
DS, VER = "nl_rdr_data_rtcor_5m_tar", "1.0"
CACHE = pathlib.Path("/mnt/storagebox/knmi_rtcor")
PROJ4 = ("+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
         "+a=6378137 +b=6356752 +x_0=0 +y_0=0 +units=km")
NROW, NCOL = 765, 700
UPPER_LEFT_LONLAT = (0.0, 55.973602)
MISSING, OUT_OF_IMAGE = 65534, 65535


def _key() -> str:
    for line in pathlib.Path("/opt/pluvio/research/.env").read_text().splitlines():
        if line.startswith("KNMI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("KNMI_API_KEY not found")


def _tar_for(stamp: str) -> pathlib.Path | None:
    """Download the daily tar whose 08:05->08:00 window contains `stamp`."""
    t = dt.datetime.strptime(stamp[:13], "%Y%m%dT%H%M")
    # The tar windows run 08:05 -> 08:00 next day, so exactly 08:00 belongs to the
    # PREVIOUS day's tar while 08:05 starts the new one.
    start = t.date() if (t.hour, t.minute) > (8, 0) else t.date() - dt.timedelta(days=1)
    end = start + dt.timedelta(days=1)
    fn = (f"RAD25_OPER_R___TARRRT__L2__{start:%Y%m%d}T080500_"
          f"{end:%Y%m%d}T080000_0001.tar")
    dest = CACHE / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        req = urllib.request.Request(f"{API}/{DS}/versions/{VER}/files/{fn}/url",
                                     headers={"Authorization": _key()})
        with urllib.request.urlopen(req, timeout=60) as r:
            url = json.load(r)["temporaryDownloadUrl"]
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        tmp.rename(dest)
        return dest
    except Exception as exc:
        LOG.warning("no RTCOR tar for %s (%s)", stamp, exc)
        return None


@functools.lru_cache(maxsize=4)
def _index(tar_path: str) -> dict:
    with tarfile.open(tar_path) as t:
        return {m.name[-15:-3]: m.name for m in t.getmembers() if m.name.endswith(".h5")}


@functools.lru_cache(maxsize=1)
def _rowcol(bounds, shape):
    """Map every analysis-grid cell to an RTCOR (row, col)."""
    from pyproj import CRS, Transformer

    w, s, e, n = bounds
    h, wd = shape
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(PROJ4), always_xy=True)
    x0, y0 = tf.transform(*UPPER_LEFT_LONLAT)
    lon = np.linspace(w, e, wd)[None, :].repeat(h, 0)
    lat = np.linspace(n, s, h)[:, None].repeat(wd, 1)
    x, y = tf.transform(lon, lat)
    col = np.round(x - x0).astype(int)      # 1 km pixels, x increases east
    row = np.round(y0 - y).astype(int)      # y decreases south
    return row, col


def _calibrated(group):
    """Apply a KNMI `GEO=a*PV+b` calibration, blanking its missing/out-of-image codes."""
    import re

    c = group["calibration"].attrs
    form = c["calibration_formulas"]
    form = form.decode() if isinstance(form, bytes) else str(form)
    m = re.match(r"GEO=([-\d.eE+]+)\*PV\+([-\d.eE+]+)", form.strip())
    a, b = float(m.group(1)), float(m.group(2))
    raw = np.asarray(group["image_data"]).astype("float32")
    out = a * raw + b
    for key in ("calibration_missing_data", "calibration_out_of_image"):
        for code in np.atleast_1d(c.get(key, [])):
            out[raw == float(code)] = np.nan
    return out


def _regrid(field, bounds, shape):
    row, col = _rowcol(bounds, shape)
    ok = (row >= 0) & (row < NROW) & (col >= 0) & (col < NCOL)
    out = np.full(shape, np.nan, "float32")
    out[ok] = field[row[ok], col[ok]]
    return out


def fields(stamp: str, bounds, shape):
    """All three RTCOR layers on the analysis grid, or None.

    Returns dict(rate=mm/h, adjust_db=gauge adjustment applied in dB, quality=0..1).
    The product ships the adjustment it applied (`image3`, ADJUSTMENT_FACTOR_[DB]), so
    the UNADJUSTED radar-only field is recoverable as rate / 10**(adjust_db/10). That
    separation is what tells us whether RTCOR's edge over us is its radar chain or its
    gauge correction — two very different things to replicate.
    """
    import io
    import h5py

    tar = _tar_for(stamp)
    if tar is None:
        return None
    member = _index(str(tar)).get(f"{stamp[:8]}{stamp[9:13]}")
    if member is None:
        return None
    with tarfile.open(tar) as t:
        buf = t.extractfile(member).read()
    with h5py.File(io.BytesIO(buf), "r") as f:
        mm5 = _calibrated(f["image1"])
        adj = _calibrated(f["image3"]) if "image3" in f else np.zeros_like(mm5)
        qual = _calibrated(f["image2"]) if "image2" in f else np.ones_like(mm5)
    return dict(rate=_regrid(mm5, bounds, shape) * 12.0,      # mm per 5 min -> mm/h
                adjust_db=_regrid(adj, bounds, shape),
                quality=_regrid(qual, bounds, shape))


def rate(stamp: str, bounds, shape):
    """RTCOR rain rate (mm/h) on the analysis grid, or None."""
    got = fields(stamp, bounds, shape)
    return None if got is None else got["rate"]
