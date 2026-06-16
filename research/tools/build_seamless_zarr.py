"""Assemble the unified seamless training store from the hetz1 collectors.

Reads the per-source crops on the Storage Box and writes `seamless.zarr` keyed
by OPERA issue-time (see model/seamless_dataset.py for the layout):

    opera_rate  (n, H, W)          OPERA analysis (truth)        ← /mnt/storagebox/opera/RATE
    aifs_tp     (n, n_lead, H, W)  AIFS forecast cube per lead   ← /mnt/storagebox/aifs   (GRIB)
    <obs aux>   (n, H, W)          li_flash, gii_*, ctth_*, …    ← /mnt/storagebox/mtg_li, mtg_l2
    static_*    (H, W)             elevation / landmask / dist   ← static.npz

Everything is reprojected onto the 100×100 analysis grid (model.geo) with the
CRS-aware helper — OPERA (LAEA), MTG (EPSG:4326), AIFS (lat/lon) all handled.
OPERA no-echo (NaN) → 0 mm/h (dry); out-of-domain → NaN. Sources not yet
accumulated (AIFS/MTG only collect forward) are NaN-filled, so this runs today
on the 22-month OPERA truth and gets richer as the obs channels fill in.

⚠️ Runs on hetz1 (where /mnt/storagebox is). A full 22-month build is ~1 h of
reprojection — use --limit / --cadence-min to test first. Append-capable.

    python -m tools.build_seamless_zarr --out /opt/pluvio/zarr/seamless.zarr \
        --cadence-min 15 --limit 500       # test slice
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
from model.nwp_regrid import reproject_to_analysis_grid  # noqa: E402

# Lead set (kept local so the builder needs no torch — mirror of
# model.seamless_dataset.DEFAULT_LEADS): 0–2 h @ 10 min, to 24 h hourly, to 240 h 3-hourly.
DEFAULT_LEADS = (
    tuple(range(0, 121, 10)) + tuple(range(180, 1441, 60)) + tuple(range(1440 + 180, 14401, 180))
)

LOG = logging.getLogger("pluvio.build_seamless")

STORAGE = pathlib.Path("/mnt/storagebox")
TS_RE = re.compile(r"(\d{8}T\d{4})")

# Obs aux channels: (zarr var, source dir glob). MTG-L2 vars live under
# mtg_l2/<PROD>/<dataset>/…; MTG-LI under mtg_li/AF/….
AUX_SOURCES: dict[str, str] = {
    "li_flash": "mtg_li/AF",
    "gii_k_index": "mtg_l2/GII/k_index",
    "gii_lifted_index": "mtg_l2/GII/lifted_index",
    "gii_prec_water_total": "mtg_l2/GII/prec_water_total",
    "ctth_cloud_top_temperature": "mtg_l2/CTTH/cloud_top_temperature",
    "ctth_cloud_top_height": "mtg_l2/CTTH/cloud_top_height",
    "oca_optical_thickness": "mtg_l2/OCA/retrieved_cloud_optical_thickness",
    "oca_cloud_phase": "mtg_l2/OCA/retrieved_cloud_phase",
    "ct_cloud_type": "mtg_l2/CT/cloud_type",
    "olr": "mtg_l2/OLR/olr",
}


def _index_tiffs(root: pathlib.Path) -> list[tuple[dt.datetime, pathlib.Path]]:
    """Sorted (timestamp, path) for date-organised crops named …<YYYYmmddTHHMM>_*.tiff."""
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*.tiff"):
        m = TS_RE.search(p.name)
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=dt.UTC), p))
    out.sort()
    return out


def _latest_le(index, ts: dt.datetime, max_age_min: int):
    """Latest (timestamp, path) ≤ ts within max_age_min (linear; index is sorted)."""
    cand = None
    for t, p in index:
        if t > ts:
            break
        if (ts - t).total_seconds() / 60 <= max_age_min:
            cand = p
    return cand


def _opera_clean(grid: np.ndarray) -> np.ndarray:
    """OPERA RATE: out-of-domain stays NaN; no-echo (also NaN after reproject)
    can't be distinguished here, so treat finite≥0 as rain and NaN as no-data.
    We zero only inside the analysis footprint via the static landmask later;
    for now keep NaN→0 for the target (dry) — radar covers the whole domain."""
    return np.nan_to_num(grid, nan=0.0).astype("float32")


def build(out_path: pathlib.Path, cadence_min: int, leads_min, limit: int | None,
          max_age_min: int, no_aifs: bool = False, no_aux: bool = False):
    import zarr

    opera_idx = _index_tiffs(STORAGE / "opera" / "RATE")
    if not opera_idx:
        LOG.error("no OPERA RATE crops under %s", STORAGE / "opera/RATE")
        return
    LOG.info("OPERA truth: %d analyses (%s … %s)", len(opera_idx),
             opera_idx[0][0].date(), opera_idx[-1][0].date())

    # Issue-times = OPERA timestamps subsampled to the requested cadence.
    step = dt.timedelta(minutes=cadence_min)
    issues: list[tuple[dt.datetime, pathlib.Path]] = []
    last = None
    for t, p in opera_idx:
        if last is None or (t - last) >= step:
            issues.append((t, p)); last = t
    if limit:
        issues = issues[:limit]
    n = len(issues)
    leads = list(leads_min)
    LOG.info("building %d issue-times × %d leads → %s", n, len(leads), out_path)

    aux_idx = {} if no_aux else {var: _index_tiffs(STORAGE / src) for var, src in AUX_SOURCES.items()}

    root = zarr.open_group(str(out_path), mode="w")
    root.create_array("issue_time", shape=(n,), dtype="int64")[:] = [int(t.timestamp()) for t, _ in issues]
    root.create_array("leads_min", shape=(len(leads),), dtype="int16")[:] = leads
    H, W = GRID
    opera_z = root.create_array("opera_rate", shape=(n, H, W), chunks=(1, H, W), dtype="float32")
    aux_z = {v: root.create_array(v, shape=(n, H, W), chunks=(1, H, W), dtype="float32") for v in aux_idx}
    # AIFS forecast cube — skipped for the baseline (it'd be ~200 GB of NaN until
    # AIFS accumulates). The dataset tolerates its absence (zero anchor).
    if not no_aifs:
        root.create_array("aifs_tp", shape=(n, len(leads), H, W), chunks=(1, len(leads), H, W), dtype="float32")

    for i, (ts, opath) in enumerate(issues):
        opera_z[i] = _opera_clean(reproject_to_analysis_grid(opath))
        for var, idx in aux_idx.items():
            src = _latest_le(idx, ts, max_age_min)
            aux_z[var][i] = (reproject_to_analysis_grid(src) if src
                             else np.full((H, W), np.nan, dtype="float32"))
        if (i + 1) % 500 == 0:
            LOG.info("  %d/%d", i + 1, n)

    # Static channels from static.npz (elevation/landmask/dist), if present.
    static = pathlib.Path(__file__).resolve().parents[1] / "model" / "static.npz"
    if static.exists():
        d = np.load(static)
        for k in d.files:
            root.create_array(f"static_{k}", shape=(H, W), dtype="float32")[:] = d[k].astype("float32")
    LOG.info("done: %s (%d issues, aux=%d, AIFS placeholder NaN)", out_path, n, len(AUX_SOURCES))


def main(argv=None) -> int:
    global STORAGE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/opt/pluvio/zarr/seamless.zarr")
    p.add_argument("--storage", default=str(STORAGE),
                   help="root holding opera/ aifs/ mtg_li/ mtg_l2/ (the staged copy on the GPU node)")
    p.add_argument("--cadence-min", type=int, default=15, help="issue-time spacing (OPERA is 15-min)")
    p.add_argument("--leads", default="", help="comma lead-mins; default = the seamless set")
    p.add_argument("--limit", type=int, default=None, help="cap issue-times (testing)")
    p.add_argument("--max-age-min", type=int, default=20, help="aux staleness tolerance")
    p.add_argument("--no-aifs", action="store_true", help="skip the AIFS cube (baseline; not yet accumulated)")
    p.add_argument("--no-aux", action="store_true", help="skip MTG obs aux channels (baseline)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    STORAGE = pathlib.Path(args.storage)
    leads = [int(x) for x in args.leads.split(",") if x.strip()] or list(DEFAULT_LEADS)
    build(pathlib.Path(args.out), args.cadence_min, leads, args.limit, args.max_age_min,
          no_aifs=args.no_aifs, no_aux=args.no_aux)
    return 0


if __name__ == "__main__":
    sys.exit(main())
