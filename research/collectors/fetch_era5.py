"""ERA5 reanalysis from **Google ARCO-ERA5** — the historical NWP field set used
to **pretrain the seamless model's outlook / downscaling head** before enough
live AIFS has accumulated.

Why ERA5: AIFS open-data only keeps a rolling ~4-day window, so it can't be
backfilled — the multi-day regime would otherwise wait months for forward
collection. ERA5 is a decades-deep reanalysis on the same ~0.25° global grid as
AIFS; its coarse precip + state fields are a good proxy for "what a good NWP
field looks like". Paired with the high-res OPERA composite as truth, it gives a
spatial-downscaling prior we can learn *today*; the head is then fine-tuned on
real AIFS as that fills in.

Why ARCO-ERA5 (not the CDS API): the analysis-ready ERA5 is mirrored as a public
Zarr on GCS (`gs://gcp-public-data-arco-era5`, anonymous, no account, no request
queue). We open it lazily and pull only our BeNeLux+context crop, one month at a
time → one NetCDF per month (`era5_sl_<YYYYMM>.nc`), resumable (skips months
already on disk). Differencing / regridding onto the analysis grid happens at
zarr-build time, not here (mirrors the AIFS collector).

`total_precipitation` is the downscaling target-proxy; the rest are convective /
moisture / near-surface-flow predictors.

    python collectors/fetch_era5.py --start 2018-01 --end now --out data/era5
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import shutil
import sys

LOG = logging.getLogger("pluvio.fetch_era5")

# Analysis-ready ERA5 (0.25°, hourly, 1940→present), public + anonymous on GCS.
DEFAULT_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Core single-level set: precip (the downscaling target-proxy) + convective /
# moisture / near-surface flow predictors. ARCO uses ECMWF long names.
DEFAULT_VARIABLES = [
    "total_precipitation",
    "convective_available_potential_energy",
    "total_column_water_vapour",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# N, W, S, E — BeNeLux + synoptic context margin (lon in -180..180; we wrap the
# dataset's 0..360 grid below).
DEFAULT_AREA = (58.0, -6.0, 46.0, 16.0)


def _months(start: str, end: str):
    """Inclusive month iterator over 'YYYY-MM' strings ('now' = current month)."""
    def parse(s: str) -> tuple[int, int]:
        if s == "now":
            today = dt.datetime.now(dt.UTC)
            return today.year, today.month
        y, m = s.split("-")
        return int(y), int(m)

    (y0, m0), (y1, m1) = parse(start), parse(end)
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    """First and last hour (inclusive) of the month, as ISO strings."""
    start = dt.datetime(year, month, 1)
    nxt = dt.datetime(year + (month // 12), (month % 12) + 1, 1)
    last = nxt - dt.timedelta(hours=1)
    return start.strftime("%Y-%m-%dT%H:%M"), last.strftime("%Y-%m-%dT%H:%M")


def _free_gb(path: pathlib.Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def open_arco(zarr_url: str, variables: list[str], area):
    """Open the ARCO-ERA5 Zarr, restrict to our variables, and crop to the box.
    Returns the lazily-cropped Dataset (no data pulled yet)."""
    import xarray as xr

    ds = xr.open_zarr(zarr_url, chunks={"time": 24},
                      storage_options={"token": "anon"})
    have = [v for v in variables if v in ds.data_vars]
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        LOG.warning("not in ARCO store, skipping: %s", missing)
    if not have:
        raise SystemExit("none of the requested variables exist in the store")
    ds = ds[have]
    # ARCO longitudes are 0..359.75; convert to -180..180 so the box doesn't wrap.
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    n, w, s, e = area
    # latitude is descending (90→-90), so slice high→low.
    ds = ds.sel(latitude=slice(n, s), longitude=slice(w, e))
    LOG.info("opened ARCO: %d vars, grid lat=%d lon=%d", len(have),
             ds.sizes.get("latitude", 0), ds.sizes.get("longitude", 0))
    return ds


def fetch_month(ds, year: int, month: int, out_dir: pathlib.Path) -> bool:
    """Pull one month of the cropped Dataset to a single NetCDF. True if written."""
    target = out_dir / f"era5_sl_{year:04d}{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        return False
    t0, t1 = _month_bounds(year, month)
    sub = ds.sel(time=slice(t0, t1))
    if sub.sizes.get("time", 0) == 0:
        LOG.info("%04d-%02d not yet in ERA5; stopping at the data edge", year, month)
        return False
    tmp = target.with_suffix(".nc.part")
    sub.load().to_netcdf(tmp, engine="netcdf4")
    tmp.rename(target)
    LOG.info("wrote %s (%.1f MB, %d steps)", target.name,
             target.stat().st_size / 1e6, sub.sizes["time"])
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr-url", default=DEFAULT_ZARR)
    p.add_argument("--variables", default=",".join(DEFAULT_VARIABLES))
    p.add_argument("--start", default="2018-01", help="first month YYYY-MM")
    p.add_argument("--end", default="now", help="last month YYYY-MM or 'now'")
    p.add_argument("--area", default=",".join(str(x) for x in DEFAULT_AREA),
                   help="N,W,S,E lat/lon box")
    p.add_argument("--out", default="data/era5")
    p.add_argument("--min-free-gb", type=float, default=50.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    area = tuple(float(x) for x in args.area.split(","))

    ds = open_arco(args.zarr_url, variables, area)
    months = list(_months(args.start, args.end))
    LOG.info("ERA5 backfill: %d months (%s … %s) → %s", len(months), args.start, args.end, out_dir)
    n_written = 0
    for year, month in months:
        if _free_gb(out_dir) < args.min_free_gb:
            LOG.error("free space < %.0f GB; stopping", args.min_free_gb)
            break
        try:
            if fetch_month(ds, year, month, out_dir):
                n_written += 1
        except Exception as exc:  # noqa: BLE001 — one bad month shouldn't kill the backfill
            LOG.warning("%04d-%02d failed (%s); skipping", year, month, exc)
    LOG.info("done: %d new months in %s", n_written, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
