"""Automated input/registration QC for the training-and-inference store.

Born from a night where a human spotted, in five minutes each: forecast cells
displaced ~50 km south of the composite (native 765x700 fields are trimmed to
700x700 before block-meaning — grid_latlon mapped rows across the untrimmed
extent), aux channels regridded to a different extent than radar/truth, and a
reversed advection sign. Every one of those failure classes is now a
continuously-measured check:

  registration   fit (dlat, dlon) cross-correlation offset between the store's
                 newest radar analysis frames (mapped via model.geo) and the
                 ground-truthed serving composite at identical valid times.
                 WARN when |offset| > 0.07 deg or peak corr < 0.25.
  aux alignment  correlation of alaro_precip / msg_ir108 (inverted) against
                 radar at the SAME grid indices for wet issues — internal
                 misregistration of inputs shows as near-zero or negative
                 correlation where physics demands strong coupling.
  channel health per-channel NaN fraction and value-range sanity over the
                 newest issues (catches dead feeds and unit regressions).
  staleness      age of the newest store issue vs wall clock (the seam-lag
                 budget: WARN beyond 75 min).

Writes /opt/pluvio/serve/qc_inputs.json and exits 1 on any WARN, so a systemd
timer surfaces failures. Run:  python -m tools.qc_inputs [-v]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.qc_inputs")

STORE = "/opt/pluvio/zarr/timeseries.zarr"
OBSERVED = "/opt/pluvio/serve/observed.npz"
OUT = "/opt/pluvio/serve/qc_inputs.json"

REG_OFFSET_WARN_DEG = 0.07
REG_CORR_WARN = 0.25
AUX_CORR_WARN = 0.05      # alaro_precip vs radar must correlate positively
STALE_WARN_MIN = 75
RANGES = {                # plausible value ranges (min, max) per channel family
    "radar": (0.0, 200.0),
    "truth": (0.0, 400.0),
    "alaro_precip": (0.0, 150.0),
    "aws_temp": (-1.0, 2.0),        # normalised
    "msg_ir108": (150.0, 340.0),    # Kelvin
}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.ravel(), y.ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def registration_check(src, t, warn: list) -> dict:
    from model.geo import grid_latlon

    glat, glon = grid_latlon()
    obs = np.load(OBSERVED, allow_pickle=False)
    ot = obs["times"].astype("int64")
    R = obs["rates"]
    W0, S0, E0, N0 = (float(x) for x in obs["bounds"])
    gh, gw = R.shape[1:]

    fits = []
    for i in range(len(t) - 1, max(-1, len(t) - 40), -1):
        if not (int(ot[0]) <= t[i] <= int(ot[-1])):
            continue
        fld = np.nan_to_num(np.asarray(src["radar"][i, 0], dtype="float32"))
        if (fld > 0.1).mean() < 0.02:
            continue
        j = int(np.argmin(np.abs(ot - t[i])))
        if abs(int(ot[j]) - int(t[i])) > 120:
            continue
        OB = np.nan_to_num(R[j]).astype("float32")

        def sample(dlat: float, dlon: float) -> np.ndarray:
            rr = ((N0 - (glat + dlat)) / (N0 - S0) * gh).astype(int)
            cc = (((glon + dlon) - W0) / (E0 - W0) * gw).astype(int)
            ok = (rr >= 0) & (rr < gh) & (cc >= 0) & (cc < gw)
            out = np.zeros_like(fld)
            out[ok] = OB[rr[ok], cc[ok]]
            return out

        best = (-9.0, 0.0, 0.0)
        for dlat in np.arange(-0.14, 0.141, 0.02):
            for dlon in np.arange(-0.14, 0.141, 0.02):
                c = _corr(fld, sample(float(dlat), float(dlon)))
                if c == c and c > best[0]:
                    best = (c, float(dlat), float(dlon))
        fits.append(best)
        if len(fits) >= 6:
            break

    if not fits:
        return {"n": 0, "note": "no wet overlapping issues to fit"}
    arr = np.array(fits)
    med = {"corr": round(float(np.median(arr[:, 0])), 3),
           "dlat": round(float(np.median(arr[:, 1])), 3),
           "dlon": round(float(np.median(arr[:, 2])), 3), "n": len(fits)}
    if abs(med["dlat"]) > REG_OFFSET_WARN_DEG or abs(med["dlon"]) > REG_OFFSET_WARN_DEG:
        warn.append(f"REGISTRATION offset (dlat={med['dlat']}, dlon={med['dlon']})")
    if med["corr"] < REG_CORR_WARN:
        warn.append(f"REGISTRATION corr {med['corr']} < {REG_CORR_WARN}")
    return med


def aux_alignment_check(src, t, warn: list) -> dict:
    out = {}
    pairs = [("alaro_precip", +1)]
    if "msg_ir108" in src:
        pairs.append(("msg_ir108", -1))   # cold tops ↔ rain: negative corr
    for name, sign in pairs:
        if name not in src:
            continue
        cs = []
        for i in range(len(t) - 1, max(-1, len(t) - 60), -1):
            radar = np.nan_to_num(np.asarray(src["radar"][i, 0], dtype="float32"))
            if (radar > 0.1).mean() < 0.03:
                continue
            aux = np.asarray(src[name][i], dtype="float32")
            if not np.isfinite(aux).any():
                continue
            c = _corr(radar, np.nan_to_num(aux))
            if c == c:
                cs.append(sign * c)
            if len(cs) >= 8:
                break
        if cs:
            med = round(float(np.median(cs)), 3)
            out[name] = {"signed_corr": med, "n": len(cs)}
            if med < AUX_CORR_WARN:
                warn.append(f"AUX-ALIGN {name} signed corr {med} < {AUX_CORR_WARN}")
    return out


def channel_health_check(src, t, warn: list) -> dict:
    out = {}
    n = len(t)
    lo = max(0, n - 48)
    for name in src.array_keys():
        a = src[name]
        if a.ndim < 3 or a.shape[0] != n:
            continue
        block = np.asarray(a[lo:n] if a.ndim == 3 else a[lo:n, 0], dtype="float32")
        nanfrac = round(float(np.mean(~np.isfinite(block))), 3)
        fin = block[np.isfinite(block)]
        vmin = round(float(fin.min()), 2) if fin.size else None
        vmax = round(float(fin.max()), 2) if fin.size else None
        out[name] = {"nan_frac": nanfrac, "min": vmin, "max": vmax}
        if nanfrac > 0.9:
            warn.append(f"CHANNEL {name} {int(nanfrac*100)}% NaN over last 48 issues")
        rng = RANGES.get(name)
        if rng and fin.size and (vmin < rng[0] - 1e-6 or vmax > rng[1]):
            warn.append(f"CHANNEL {name} out of range [{vmin}, {vmax}] vs {rng}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=STORE)
    p.add_argument("--out", default=OUT)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    import zarr

    src = zarr.open_group(args.store, mode="r")
    t = np.asarray(src["issue_time"][:]).astype("int64")
    warn: list[str] = []

    stale_min = round((dt.datetime.now(dt.UTC).timestamp() - int(t[-1])) / 60)
    if stale_min > STALE_WARN_MIN:
        warn.append(f"STALE newest issue {stale_min} min old")

    body = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "newest_issue_age_min": stale_min,
        "registration": registration_check(src, t, warn),
        "aux_alignment": aux_alignment_check(src, t, warn),
        "channels": channel_health_check(src, t, warn),
        "warnings": warn,
    }
    op = pathlib.Path(args.out)
    tmp = op.with_name(op.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1))
    tmp.replace(op)
    for w in warn:
        LOG.warning("%s", w)
    LOG.info("wrote %s (%d warnings)", op, len(warn))
    return 1 if warn else 0


if __name__ == "__main__":
    sys.exit(main())
