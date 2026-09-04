"""Deployment gate for a nowcast checkpoint (TODO 3.5).

Nothing reaches serving on a feeling. A candidate checkpoint is promoted only
if, on the SAME frozen benchmark sample set as the incumbent:

  * the benchmark run is adequate (enough rain events), and
  * at 1 mm/h it is not significantly worse than the incumbent on CSI, FSS
    (3 px) or RMSE at any served lead — "significantly worse" meaning the
    candidate's 90 % CI lies entirely on the bad side of the incumbent's point
    estimate — and
  * it is significantly better (its CI clear of the incumbent's point) on at
    least one of those metrics at one lead,

and then survives a canary: N paired nowcast fields (incumbent vs candidate,
same issue) with finite values, plausible rates, a wet fraction within a
factor of `max_wet_ratio` of the incumbent's, and a field RMSE below
`max_rmse_mm_h`. Only then is the checkpoint file replaced, atomically, with
the previous one kept beside it.

    python -m tools.promote_checkpoint gate --incumbent bench/a.json --candidate bench/b.json \
        --incumbent-name v3_huber --candidate-name v3_fss --out gate.json
    python -m tools.promote_checkpoint canary --incumbent-dir canary/old --candidate-dir canary/new --out canary.json
    python -m tools.promote_checkpoint swap --candidate ckpt/new.pt --target /opt/pluvio/research/checkpoints/pluvio_unet.pt \
        --gate gate.json --canary canary.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import shutil
import sys

import numpy as np

LOG = logging.getLogger("pluvio.promote")

GATE_THRESHOLD = "1.0"
GATE_METRICS = ("csi", "fss_3", "rmse")   # higher-is-better, higher, lower
LOWER_IS_BETTER = {"rmse"}


# ───────────────────────────────────────────────────────────── benchmark gate


def _metric(row: dict, metric: str) -> tuple[float, float, float]:
    """(point, ci_lo, ci_hi) for one metric of a benchmark row."""
    if metric == "fss_3":
        point = float(row["fss"]["3"])
    else:
        point = float(row[metric])
    ci = (row.get("ci") or {}).get(metric) or {}
    lo = float(ci.get("ci_lo", point))
    hi = float(ci.get("ci_hi", point))
    return point, lo, hi


def compare_benchmarks(incumbent: dict, candidate: dict, incumbent_name: str,
                       candidate_name: str, leads=None, threshold: str = GATE_THRESHOLD) -> dict:
    """Gate verdict from two benchmark result JSONs (tools.benchmark output)."""
    reasons: list[str] = []
    mi, mc = incumbent.get("metadata", {}), candidate.get("metadata", {})
    if mi.get("sample_set_hash") != mc.get("sample_set_hash"):
        reasons.append(f"different sample sets: {mi.get('sample_set_hash')} vs {mc.get('sample_set_hash')}")
    if not mc.get("adequate", False):
        reasons.append("candidate benchmark not adequate (too few rain events)")
    ri = incumbent["results"].get(incumbent_name)
    rc = candidate["results"].get(candidate_name)
    if ri is None or rc is None:
        reasons.append(f"missing model rows: {incumbent_name!r} in incumbent={ri is not None}, "
                       f"{candidate_name!r} in candidate={rc is not None}")
        return {"promote": False, "reasons": reasons, "per_lead": {}}
    leads = [str(x) for x in (leads or sorted(ri.keys(), key=float))]
    per_lead: dict = {}
    n_better = 0
    for lead in leads:
        a = ri.get(lead, {}).get(threshold)
        b = rc.get(lead, {}).get(threshold)
        if not a or not b:
            reasons.append(f"lead {lead}: missing threshold {threshold} row")
            continue
        cell: dict = {}
        for metric in GATE_METRICS:
            pa, _, _ = _metric(a, metric)
            pb, lo_b, hi_b = _metric(b, metric)
            if metric in LOWER_IS_BETTER:
                worse = lo_b > pa           # whole candidate CI above incumbent point
                better = hi_b < pa
            else:
                worse = hi_b < pa
                better = lo_b > pa
            cell[metric] = {"incumbent": pa, "candidate": pb, "ci": [lo_b, hi_b],
                            "significantly_worse": bool(worse), "significantly_better": bool(better)}
            if worse:
                reasons.append(f"lead {lead} {metric}: candidate {pb:.3f} [{lo_b:.3f},{hi_b:.3f}] "
                               f"significantly worse than incumbent {pa:.3f}")
            if better:
                n_better += 1
        per_lead[lead] = cell
    if n_better == 0:
        reasons.append("candidate is not significantly better on any metric at any lead")
    return {"promote": not reasons, "reasons": reasons, "per_lead": per_lead,
            "n_significantly_better": n_better, "threshold_mm_h": threshold,
            "sample_set_hash": mc.get("sample_set_hash"),
            "incumbent": incumbent_name, "candidate": candidate_name,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


# ─────────────────────────────────────────────────────────────────── canary


def compare_canary_pair(old: dict, new: dict, *, max_rate_mm_h: float = 300.0,
                        max_rmse_mm_h: float = 2.0, max_wet_ratio: float = 3.0) -> dict:
    """One paired nowcast npz (dict-like with rates/leads/issue_epoch)."""
    reasons: list[str] = []
    ro, rn = np.asarray(old["rates"], dtype="float32"), np.asarray(new["rates"], dtype="float32")
    if ro.shape != rn.shape:
        return {"ok": False, "reasons": [f"shape {rn.shape} != incumbent {ro.shape}"]}
    if int(old["issue_epoch"]) != int(new["issue_epoch"]):
        reasons.append("issue_epoch differs — not the same issue")
    if not np.isfinite(rn).all():
        reasons.append("candidate has non-finite values")
    if float(np.nanmin(rn)) < 0.0:
        reasons.append("candidate has negative rates")
    if float(np.nanmax(rn)) > max_rate_mm_h:
        reasons.append(f"candidate max {float(np.nanmax(rn)):.1f} mm/h > {max_rate_mm_h}")
    wet_o = float((ro > 0.1).mean())
    wet_n = float((rn > 0.1).mean())
    if wet_o > 0.002 or wet_n > 0.002:
        ratio = (wet_n + 1e-6) / (wet_o + 1e-6)
        if ratio > max_wet_ratio or ratio < 1.0 / max_wet_ratio:
            reasons.append(f"wet fraction {wet_n:.4f} vs incumbent {wet_o:.4f} (ratio {ratio:.2f})")
    diff = np.nan_to_num(rn) - np.nan_to_num(ro)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    if rmse > max_rmse_mm_h:
        reasons.append(f"field RMSE vs incumbent {rmse:.3f} mm/h > {max_rmse_mm_h}")
    return {"ok": not reasons, "reasons": reasons, "rmse_mm_h": rmse,
            "wet_frac_incumbent": wet_o, "wet_frac_candidate": wet_n,
            "max_candidate": float(np.nanmax(rn))}


def compare_canary_dirs(old_dir: pathlib.Path, new_dir: pathlib.Path, min_pairs: int = 6, **kw) -> dict:
    """Pair npz files by name across two directories; all pairs must pass."""
    pairs = sorted(set(p.name for p in old_dir.glob("*.npz")) & set(p.name for p in new_dir.glob("*.npz")))
    results = {}
    for name in pairs:
        with np.load(old_dir / name, allow_pickle=True) as o, np.load(new_dir / name, allow_pickle=True) as n:
            results[name] = compare_canary_pair(o, n, **kw)
    reasons = [f"{k}: {'; '.join(v['reasons'])}" for k, v in results.items() if not v["ok"]]
    if len(pairs) < min_pairs:
        reasons.append(f"only {len(pairs)} paired canary ticks (< {min_pairs})")
    return {"ok": not reasons, "reasons": reasons, "n_pairs": len(pairs), "pairs": results,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


# ───────────────────────────────────────────────────────────────────── swap


def swap_checkpoint(candidate: pathlib.Path, target: pathlib.Path, gate: dict, canary: dict) -> pathlib.Path:
    """Replace ``target`` by ``candidate`` atomically; keep the previous file
    beside it as ``<target>.prev-<UTC stamp>``. Refuses without both verdicts."""
    if not gate.get("promote"):
        raise SystemExit("gate verdict is not promote: " + "; ".join(gate.get("reasons", [])))
    if not canary.get("ok"):
        raise SystemExit("canary verdict is not ok: " + "; ".join(canary.get("reasons", [])))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = target.with_name(target.name + f".prev-{stamp}")
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(candidate, tmp)
    if target.exists():
        shutil.copy2(target, backup)
    tmp.replace(target)
    record = target.with_name(target.name + ".promotion.json")
    record.write_text(json.dumps({"promoted_at": stamp, "candidate": str(candidate),
                                  "previous": str(backup) if backup.exists() else None,
                                  "gate": gate, "canary": {k: v for k, v in canary.items() if k != "pairs"}},
                                 indent=1))
    LOG.info("promoted %s -> %s (previous kept as %s)", candidate, target, backup.name)
    return backup


# ────────────────────────────────────────────────────────────────────── CLI


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--incumbent", required=True)
    g.add_argument("--candidate", required=True)
    g.add_argument("--incumbent-name", required=True)
    g.add_argument("--candidate-name", required=True)
    g.add_argument("--leads", default=None, help="comma-separated leads to gate on (default all)")
    g.add_argument("--out", required=True)
    c = sub.add_parser("canary")
    c.add_argument("--incumbent-dir", required=True)
    c.add_argument("--candidate-dir", required=True)
    c.add_argument("--min-pairs", type=int, default=6)
    c.add_argument("--max-rmse-mm-h", type=float, default=2.0)
    c.add_argument("--out", required=True)
    s = sub.add_parser("swap")
    s.add_argument("--candidate", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--gate", required=True)
    s.add_argument("--canary", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if args.cmd == "gate":
        leads = [x.strip() for x in args.leads.split(",")] if args.leads else None
        verdict = compare_benchmarks(json.load(open(args.incumbent)), json.load(open(args.candidate)),
                                     args.incumbent_name, args.candidate_name, leads=leads)
        pathlib.Path(args.out).write_text(json.dumps(verdict, indent=1))
        for r in verdict["reasons"]:
            LOG.warning("%s", r)
        LOG.info("gate: %s (%d significantly better)", "PROMOTE" if verdict["promote"] else "HOLD",
                 verdict.get("n_significantly_better", 0))
        return 0 if verdict["promote"] else 1
    if args.cmd == "canary":
        verdict = compare_canary_dirs(pathlib.Path(args.incumbent_dir), pathlib.Path(args.candidate_dir),
                                      min_pairs=args.min_pairs, max_rmse_mm_h=args.max_rmse_mm_h)
        pathlib.Path(args.out).write_text(json.dumps(verdict, indent=1))
        for r in verdict["reasons"]:
            LOG.warning("%s", r)
        LOG.info("canary: %s over %d pairs", "OK" if verdict["ok"] else "FAIL", verdict["n_pairs"])
        return 0 if verdict["ok"] else 1
    swap_checkpoint(pathlib.Path(args.candidate), pathlib.Path(args.target),
                    json.load(open(args.gate)), json.load(open(args.canary)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
