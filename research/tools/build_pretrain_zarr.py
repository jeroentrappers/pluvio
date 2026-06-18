"""Assemble the ERA5→OPERA **downscaling pretraining** store.

Stage A of the seamless plan: teach the net the coarse→fine mapping from ERA5
reanalysis fields to the high-res OPERA radar truth at the *same* valid time
(lead 0, no forecast error yet — that comes in the AIFS fine-tune). Pairs each
OPERA analysis with the ERA5 surface fields regridded onto the 100×100 analysis
grid.

    opera_rate  (n, H, W)   OPERA analysis (truth, mm/h)
    era5_tp     (n, H, W)   ERA5 total precip (mm/h)  ← the coarse field to sharpen
    era5_cape / era5_tcwv / era5_t2m / era5_msl / era5_u10 / era5_v10  (n, H, W)
    issue_time  (n,)        int64 epoch
    leads_min   (1,)        = [0]

ERA5 lives in monthly NetCDFs (collectors/fetch_era5.py) keyed era5_sl_<YYYYMM>.nc,
hourly contiguous from day-1 00:00 UTC, so band = hour-of-month + 1. Only builds
issue-times where BOTH an OPERA analysis and the ERA5 month exist (the overlap).

Torch-free → runs in the seamless-builder image on hetz1.

    python tools/build_pretrain_zarr.py --out /stage/pretrain.zarr --storage /stage \
        --era5 /mnt/storagebox/era5 --cadence-min 60
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.geo import GRID  # noqa: E402
from model.nwp_regrid import reproject_era5_var, reproject_to_analysis_grid  # noqa: E402

LOG = logging.getLogger("pluvio.build_pretrain")
TS_RE = re.compile(r"(\d{8}T\d{4})")

# ERA5 var → (zarr channel, unit scale). tp is hourly accumulation in m → mm/h.
ERA5_VARS = {
    "total_precipitation": ("era5_tp", 1000.0),
    "convective_available_potential_energy": ("era5_cape", 1.0),
    "total_column_water_vapour": ("era5_tcwv", 1.0),
    "2m_temperature": ("era5_t2m", 1.0),
    "mean_sea_level_pressure": ("era5_msl", 1.0),
    "10m_u_component_of_wind": ("era5_u10", 1.0),
    "10m_v_component_of_wind": ("era5_v10", 1.0),
}


def _index_tiffs(root: pathlib.Path):
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*.tiff"):
        m = TS_RE.search(p.name)
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=dt.UTC), p))
    out.sort()
    return out


def _era5_file_band(ts: dt.datetime, era5_dir: pathlib.Path):
    """(path, band) for ERA5 at hour ts, or None if that month isn't downloaded."""
    f = era5_dir / f"era5_sl_{ts.year:04d}{ts.month:02d}.nc"
    if not f.exists():
        return None
    month_start = dt.datetime(ts.year, ts.month, 1, tzinfo=dt.UTC)
    band = int((ts - month_start).total_seconds() // 3600) + 1  # 1-indexed
    return f, band


# Open-Meteo high-res NWP precip (collectors/fetch_openmeteo_nwp.py → om_sl_<YYYYMM>.nc,
# same (time,lat,lon) layout as ERA5). netcdf var → (zarr channel, scale). mm/h already.
OM_VARS = {
    "om_icon_d2": ("om_icon_d2", 1.0),
    "om_meteofrance_arome_france_hd": ("om_arome", 1.0),
    "om_ecmwf_ifs025": ("om_ifs", 1.0),
}


def _om_file_band(ts: dt.datetime, om_dir: pathlib.Path):
    """(path, band) for the Open-Meteo monthly NetCDF at hour ts, or None."""
    f = om_dir / f"om_sl_{ts.year:04d}{ts.month:02d}.nc"
    if not f.exists():
        return None
    month_start = dt.datetime(ts.year, ts.month, 1, tzinfo=dt.UTC)
    return f, int((ts - month_start).total_seconds() // 3600) + 1


def build(out_path, storage, era5_dir, cadence_min, limit, leads=(0,), om_dir=None):
    import zarr

    opera_idx = _index_tiffs(storage / "opera" / "RATE")
    if not opera_idx:
        LOG.error("no OPERA RATE crops under %s", storage / "opera/RATE")
        return
    LOG.info("OPERA truth: %d analyses (%s … %s)", len(opera_idx),
             opera_idx[0][0].date(), opera_idx[-1][0].date())

    # Issue-times: OPERA analyses on the requested cadence (round to the hour to
    # match ERA5) where the ERA5 month is on disk (the overlap).
    step = dt.timedelta(minutes=cadence_min)
    issues, last = [], None
    for t, p in opera_idx:
        if t.minute != 0:  # ERA5 is hourly — pair only on-the-hour analyses
            continue
        if last is not None and (t - last) < step:
            continue
        if _era5_file_band(t, era5_dir) is None:
            continue
        issues.append((t, p)); last = t
    if limit:
        issues = issues[:limit]
    n = len(issues)
    if n == 0:
        LOG.error("no overlap issue-times (ERA5 months on disk: %s)",
                  sorted(x.name for x in era5_dir.glob('*.nc'))[:3])
        return
    LOG.info("building %d downscaling pairs (cadence %dmin) → %s", n, cadence_min, out_path)

    H, W = GRID
    root = zarr.open_group(str(out_path), mode="w")
    root.create_array("issue_time", shape=(n,), dtype="int64")[:] = [int(t.timestamp()) for t, _ in issues]
    root.create_array("leads_min", shape=(len(leads),), dtype="int16")[:] = list(leads)
    opera_z = root.create_array("opera_rate", shape=(n, H, W), chunks=(1, H, W), dtype="float32")
    era5_z = {chan: root.create_array(chan, shape=(n, H, W), chunks=(1, H, W), dtype="float32")
              for _, (chan, _) in ERA5_VARS.items()}
    om_z = ({chan: root.create_array(chan, shape=(n, H, W), chunks=(1, H, W), dtype="float32")
             for _, (chan, _) in OM_VARS.items()} if om_dir else {})

    for i, (ts, opath) in enumerate(issues):
        opera_z[i] = np.nan_to_num(reproject_to_analysis_grid(opath), nan=0.0).astype("float32")
        f, band = _era5_file_band(ts, era5_dir)
        for var, (chan, scale) in ERA5_VARS.items():
            try:
                era5_z[chan][i] = reproject_era5_var(f, var, band, scale=scale)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("era5 %s @ %s failed (%s)", var, ts, exc)
                era5_z[chan][i] = np.full((H, W), np.nan, dtype="float32")
        if om_dir:
            omfb = _om_file_band(ts, om_dir)
            for var, (chan, scale) in OM_VARS.items():
                try:
                    om_z[chan][i] = (reproject_era5_var(omfb[0], var, omfb[1], scale=scale)
                                     if omfb else np.full((H, W), np.nan, dtype="float32"))
                except Exception:  # noqa: BLE001 — var missing for a model/month → NaN
                    om_z[chan][i] = np.full((H, W), np.nan, dtype="float32")
        if (i + 1) % 500 == 0:
            LOG.info("  %d/%d", i + 1, n)
    LOG.info("done: %s (%d pairs, %d ERA5 + %d Open-Meteo channels)", out_path, n, len(ERA5_VARS), len(om_z))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/stage/pretrain.zarr")
    p.add_argument("--storage", default="/stage", help="root holding opera/RATE crops")
    p.add_argument("--era5", default="/mnt/storagebox/era5", help="dir of era5_sl_<YYYYMM>.nc")
    p.add_argument("--openmeteo", default="", help="dir of om_sl_<YYYYMM>.nc (high-res NWP anchor); empty=skip")
    p.add_argument("--cadence-min", type=int, default=60, help="issue spacing (ERA5 is hourly)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--leads", default="0",
                   help="comma lead-mins to store in leads_min. '0' = Stage-A downscaling; "
                        "a 0..14400 set = Stage-B unified (anchor read at valid-time).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    leads = tuple(int(x) for x in args.leads.split(",") if x.strip())
    om_dir = pathlib.Path(args.openmeteo) if args.openmeteo else None
    build(pathlib.Path(args.out), pathlib.Path(args.storage), pathlib.Path(args.era5),
          args.cadence_min, args.limit, leads=leads, om_dir=om_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
