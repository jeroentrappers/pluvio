"""Copernicus Marine sea-surface temperature (OSTIA L4, daily).

North Sea / Channel SST strongly modulates Belgian-Dutch summer convection, and
it's quasi-static at the nowcast timescale — a slowly-varying aux channel.

Source: OSTIA L4 NRT (dataset ``METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2``,
product SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001), gap-free `analysed_sst` at
0.05°, daily, archive since 2007 — so it backfills the training window.

Access is the Copernicus Marine Toolbox; auth comes from the cached credentials
file written by ``copernicusmarine login`` (no password needed here). We pull
the whole window in one subset, then write one north-up float32 GeoTIFF of SST
in °C per day to data/sst/, which build_zarr reprojects onto the analysis grid
exactly like the MSG/ALARO raster channels (land is NaN — the signal is sea).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

LOG = logging.getLogger("pluvio.fetch_sst")

DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
VARIABLE = "analysed_sst"
# Pixel-edge bbox of the 0.05° subset; must match SST_BBOX in tools/build_zarr.py.
BBOX = (-1.0, 48.0, 11.0, 57.0)  # minx, miny, maxx, maxy (EPSG:4326)


def _subset(start: datetime, end: datetime, out_nc: pathlib.Path) -> bool:
    minx, miny, maxx, maxy = BBOX
    cmd = [
        "copernicusmarine", "subset", "--dataset-id", DATASET,
        "--variable", VARIABLE,
        "--start-datetime", start.strftime("%Y-%m-%dT00:00:00"),
        "--end-datetime", end.strftime("%Y-%m-%dT00:00:00"),
        "--minimum-longitude", str(minx), "--maximum-longitude", str(maxx),
        "--minimum-latitude", str(miny), "--maximum-latitude", str(maxy),
        "--output-filename", out_nc.name, "--output-directory", str(out_nc.parent),
        "--netcdf-compression-level", "1", "--disable-progress-bar",
        "--overwrite", "--log-level", "ERROR",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        LOG.error("subset failed: %s", (r.stderr or r.stdout)[:400])
        return False
    return out_nc.exists()


def _write_geotiff(arr_north_up: np.ndarray, target: pathlib.Path) -> None:
    import rasterio
    from rasterio.transform import from_bounds
    minx, miny, maxx, maxy = BBOX
    h, w = arr_north_up.shape
    transform = from_bounds(minx, miny, maxx, maxy, w, h)
    with rasterio.open(
        target, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(arr_north_up.astype("float32"), 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC ISO date (inclusive).")
    parser.add_argument("--end", required=True, help="UTC ISO date (inclusive).")
    parser.add_argument("--out", default="data/sst", help="Output dir for per-day GeoTIFFs.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import pandas as pd
    import xarray as xr

    start, end = _iso(args.start), _iso(args.end)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        nc = pathlib.Path(td) / "sst.nc"
        if not _subset(start, end, nc):
            return 1
        ds = xr.open_dataset(nc)
        lat_ascending = float(ds.latitude[0]) < float(ds.latitude[-1])
        wrote = skipped = 0
        for t in ds.time.values:
            stamp = pd.Timestamp(t).strftime("%Y%m%dT000000Z")
            target = out_dir / f"sst_{stamp}.tif"
            if target.exists():
                skipped += 1
                continue
            celsius = ds[VARIABLE].sel(time=t).values - 273.15  # K → °C, land=NaN
            if lat_ascending:               # OSTIA lat runs S→N; GeoTIFF row0 = north
                celsius = np.flipud(celsius)
            _write_geotiff(celsius, target)
            wrote += 1
        ds.close()
    LOG.info("SST: wrote %d day(s), skipped %d existing → %s", wrote, skipped, out_dir)
    return 0


def _iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
