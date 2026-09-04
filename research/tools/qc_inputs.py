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
  aux alignment  correlation of alaro_precip / msg_ir108 (luminance) against
                 radar at the SAME grid indices for wet issues — internal
                 misregistration of inputs shows as near-zero or negative
                 correlation where physics demands strong coupling.
  channel health per-channel NaN fraction and value-range sanity over the
                 newest issues (catches dead feeds and unit regressions).
                 Ranges are calibrated to the store's OBSERVED conventions,
                 not idealised units — see tools/qc/thresholds.py.
  staleness      age of the newest store issue vs wall clock (the seam-lag
                 budget: WARN beyond 75 min).

The actual check math lives in tools/qc/ (a plain library, no I/O); this
file is the CLI: opens the store, drives the checks over it, and writes the
legacy-shaped JSON below so nothing downstream (systemd unit, dashboards)
has to change. The output also carries an additive "verdict" key — the one
{generated, checks, summary} shape from tools/qc/verdict.py — alongside the
untouched legacy top-level keys, for anything that wants the unified form.

Deploys must copy the whole tools/qc/ package, not just this file — both
qc_inputs.py and qc_watchdog.py import it.

Writes /opt/pluvio/serve/qc_inputs.json and exits 1 on any WARN, so a systemd
timer surfaces failures. Run:  python -m tools.qc_inputs [-v]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.qc import checks
from tools.qc.thresholds import load_thresholds
from tools.qc.verdict import build_verdict, write_atomic

LOG = logging.getLogger("pluvio.qc_inputs")

STORE = "/opt/pluvio/zarr/timeseries.zarr"
OBSERVED = "/opt/pluvio/serve/observed.npz"
OUT = "/opt/pluvio/serve/qc_inputs.json"


def registration_check(src, t, thresholds, warn: list, all_checks: list) -> dict:
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

        def sample(dlat: float, dlon: float, OB=OB) -> np.ndarray:
            rr = ((N0 - (glat + dlat)) / (N0 - S0) * gh).astype(int)
            cc = (((glon + dlon) - W0) / (E0 - W0) * gw).astype(int)
            ok = (rr >= 0) & (rr < gh) & (cc >= 0) & (cc < gw)
            out = np.zeros_like(fld)
            out[ok] = OB[rr[ok], cc[ok]]
            return out

        fits.append(checks.registration_offset(fld, sample))
        if len(fits) >= 6:
            break

    check = checks.aggregate_registration(fits, thresholds)
    all_checks.append(check)
    if check.status == "warn":
        # two separate warning strings (offset; corr), mirroring how
        # channel_health's detail splits on "; " — so a fit that trips both
        # thresholds contributes two lines to `warnings`/the warning count,
        # same as the original inline checks did.
        for detail in check.detail.split("; "):
            if detail:
                warn.append(f"REGISTRATION {detail}")
    return check.value


def aux_alignment_check(src, t, thresholds, warn: list, all_checks: list) -> dict:
    out = {}
    pairs = [("alaro_precip", +1)]
    if "msg_ir108" in src:
        # msg_ir108 in the store is band-1 LUMINANCE of the rendered IR image
        # (0-255; cold tops are bright), not a Kelvin brightness temperature —
        # so rain correlates POSITIVELY with it (+0.26 measured 2026-09-04).
        pairs.append(("msg_ir108", +1))
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
            c = checks.signed_corr(radar, aux, sign)
            if c == c:
                cs.append(c)
            if len(cs) >= 8:
                break
        check = checks.aggregate_aux_alignment(name, cs, thresholds)
        all_checks.append(check)
        if check.value is not None:
            out[name] = check.value
        if check.status == "warn":
            warn.append(f"AUX-ALIGN {check.detail}")
    return out


def channel_health_check(src, t, thresholds, warn: list, all_checks: list) -> dict:
    out = {}
    n = len(t)
    lo = max(0, n - 48)
    for name in src.array_keys():
        a = src[name]
        if a.ndim < 3 or a.shape[0] != n:
            continue
        block = np.asarray(a[lo:n] if a.ndim == 3 else a[lo:n, 0], dtype="float32")
        check = checks.channel_health(block, name, thresholds)
        all_checks.append(check)
        out[name] = check.value
        if check.status == "warn":
            for detail in check.detail.split("; "):
                if detail:
                    warn.append(f"CHANNEL {detail}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=STORE)
    p.add_argument("--out", default=OUT)
    p.add_argument("--thresholds", default=None,
                    help="path to a thresholds YAML/JSON file "
                         "(default: $PLUVIO_QC_THRESHOLDS or built-in defaults)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    import zarr
    from model.geo import log_resolved_geometry

    log_resolved_geometry()

    thresholds = load_thresholds(args.thresholds)
    src = zarr.open_group(args.store, mode="r")
    t = np.asarray(src["issue_time"][:]).astype("int64")
    warn: list[str] = []
    all_checks: list = []

    now = dt.datetime.now(dt.UTC).timestamp()
    stale = checks.staleness(int(t[-1]), now, thresholds.stale_warn_min)
    all_checks.append(stale)
    stale_min = stale.value
    if stale.status == "warn":
        warn.append(f"STALE {stale.detail}")

    generated = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    body = {
        "generated": generated,
        "newest_issue_age_min": stale_min,
        "registration": registration_check(src, t, thresholds, warn, all_checks),
        "aux_alignment": aux_alignment_check(src, t, thresholds, warn, all_checks),
        "channels": channel_health_check(src, t, thresholds, warn, all_checks),
        "warnings": warn,
        "verdict": build_verdict(all_checks, generated=generated),
    }
    op = write_atomic(args.out, body)
    for w in warn:
        LOG.warning("%s", w)
    LOG.info("wrote %s (%d warnings)", op, len(warn))
    return 1 if warn else 0


if __name__ == "__main__":
    sys.exit(main())
