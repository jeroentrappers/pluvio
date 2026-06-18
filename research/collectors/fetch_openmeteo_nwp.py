"""High-resolution NWP precip from **Open-Meteo** as a downscaling training anchor.

Open-Meteo's Historical Forecast API archives convection-permitting models —
**ICON-D2 (~2.2 km, full BeNeLux+DE)** and **AROME-HD (~1.3 km, FR + S. BeNeLux)**
— over our OPERA window (verified 2024-08→now), which the *native* sources don't
keep. Unlike ERA5 (0.25°/~28 km), these are near convective scale, so they're a
far better anchor for the downscaling head AND the high-res baseline the paper
needs.

Mechanics that make a point API feasible: one call returns a full hourly *series*
per point, and many points batch into one call — so cost ≈ (#grid points) ×
(#month-chunks), not ×hours. We query a regular lat/lon grid over the domain,
month by month (per-call range cap), and write **one NetCDF per month** with one
variable per model — the **same (time,lat,lon) layout as the ERA5 collector**, so
`build_pretrain_zarr` / `nwp_regrid.reproject_era5_var` consume it unchanged.

    python collectors/fetch_openmeteo_nwp.py --start 2024-08 --end now \
        --res-deg 0.1 --models icon_d2,meteofrance_arome_france_hd --out data/openmeteo

CC-BY-4.0; cite Open-Meteo + the underlying models.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys
import time

import numpy as np

LOG = logging.getLogger("pluvio.fetch_openmeteo")
HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# bbox W,S,E,N — the analysis-grid envelope + margin (matches ERA5 area intent).
DEFAULT_BBOX = (-2.0, 47.0, 12.0, 56.0)
DEFAULT_MODELS = ("icon_d2", "meteofrance_arome_france_hd")


def _months(start: str, end: str):
    def parse(s):
        if s == "now":
            t = dt.datetime.now(dt.UTC); return t.year, t.month
        y, m = s.split("-"); return int(y), int(m)
    (y0, m0), (y1, m1) = parse(start), parse(end)
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _grid(bbox, res):
    w, s, e, n = bbox
    lats = np.arange(n, s - 1e-9, -res)   # north→south
    lons = np.arange(w, e + 1e-9, res)
    return lats.astype("float64"), lons.astype("float64")


def _fetch_batch(session, lats, lons, models, d0, d1, retries=12):
    """One Historical-Forecast call for a batch of points → list of per-point dicts.

    Free-tier friendly: the historical API rate-limits heavy (many-location ×
    many-model × month) calls hard, so on 429 we honor Retry-After and back off
    long (up to ~5 min), retrying many times rather than dropping the batch — the
    backfill self-paces to the limit instead of failing."""
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": "precipitation",
        "models": ",".join(models),
        "start_date": d0, "end_date": d1, "timezone": "UTC",
    }
    for k in range(retries):
        r = session.get(HIST_URL, params=params, timeout=180)
        if r.status_code == 200:
            j = r.json()
            return j if isinstance(j, list) else [j]
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait = int(ra) if (ra and ra.isdigit()) else min(300, 15 * (2 ** min(k, 4)))
            time.sleep(wait); continue
        if r.status_code in (500, 502, 503):
            time.sleep(min(120, 8 * (k + 1))); continue
        r.raise_for_status()
    raise RuntimeError(f"open-meteo batch failed after {retries} retries")


def build_month(session, year, month, lats, lons, models, batch, out_dir, pace=2.0):
    import xarray as xr

    target = out_dir / f"om_sl_{year:04d}{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        return False
    d0 = f"{year:04d}-{month:02d}-01"
    nxt = dt.date(year + month // 12, month % 12 + 1, 1)
    d1 = (nxt - dt.timedelta(days=1)).isoformat()

    # all grid points as flat (lat,lon) list, batched
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    pts = list(zip(LAT.ravel(), LON.ravel()))
    nlat, nlon = len(lats), len(lons)
    # model → (time, nlat*nlon) filled per point
    times = None
    cols = {m: None for m in models}
    var = {m: f"precipitation_{m}" for m in models}
    for i in range(0, len(pts), batch):
        chunk = pts[i:i + batch]
        res = _fetch_batch(session, [p[0] for p in chunk], [p[1] for p in chunk], models, d0, d1)
        for j, loc in enumerate(res):
            h = loc.get("hourly", {})
            if times is None:
                times = np.array(h["time"], dtype="datetime64[ns]")
                for m in models:
                    cols[m] = np.full((len(times), len(pts)), np.nan, dtype="float32")
            for m in models:
                v = h.get(var[m])
                if v is not None:
                    cols[m][:, i + j] = np.asarray([np.nan if x is None else x for x in v], dtype="float32")
        if (i // batch) % 10 == 0:
            LOG.info("  %s-%02d: %d/%d points", year, month, min(i + batch, len(pts)), len(pts))
        time.sleep(pace)  # inter-call pacing to stay under the free-tier rate limit

    if times is None:
        LOG.warning("%s-%02d: no data", year, month); return False
    ds = xr.Dataset(
        {f"om_{m}": (("time", "latitude", "longitude"),
                     cols[m].reshape(len(times), nlat, nlon)) for m in models},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    tmp = target.with_suffix(".nc.part")
    ds.to_netcdf(tmp, engine="netcdf4"); tmp.rename(target)
    LOG.info("wrote %s (%d steps, %d×%d grid, %d models)", target.name, len(times), nlat, nlon, len(models))
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2024-08")
    p.add_argument("--end", default="now")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)), help="W,S,E,N")
    p.add_argument("--res-deg", type=float, default=0.1, help="query grid spacing (deg); 0.1≈10km, 0.05≈5.5km")
    p.add_argument("--batch", type=int, default=150, help="grid points per API call")
    p.add_argument("--pace", type=float, default=2.0, help="seconds to sleep between API calls (free-tier pacing)")
    p.add_argument("--out", default="data/openmeteo")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import requests
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(float(x) for x in args.bbox.split(","))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lats, lons = _grid(bbox, args.res_deg)
    LOG.info("Open-Meteo NWP: grid %d×%d (%.2f°), models=%s, %s..%s → %s",
             len(lats), len(lons), args.res_deg, models, args.start, args.end, out)
    session = requests.Session()
    n = 0
    for y, m in _months(args.start, args.end):
        try:
            if build_month(session, y, m, lats, lons, models, args.batch, out, pace=args.pace):
                n += 1
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%04d-%02d failed (%s); skipping", y, m, exc)
    LOG.info("done: %d new months in %s", n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
