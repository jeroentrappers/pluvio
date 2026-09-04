"""Deployment gate (3.5): benchmark comparison, canary pairs, atomic swap."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import promote_checkpoint as pc


def _row(csi, fss3, rmse, hw=0.02):
    return {"csi": csi, "rmse": rmse, "fss": {"3": fss3},
            "ci": {"csi": {"ci_lo": csi - hw, "ci_hi": csi + hw},
                   "fss_3": {"ci_lo": fss3 - hw, "ci_hi": fss3 + hw},
                   "rmse": {"ci_lo": rmse - hw, "ci_hi": rmse + hw}}}


def _bench(name, rows, sample_hash="abc", adequate=True):
    return {"metadata": {"sample_set_hash": sample_hash, "adequate": adequate},
            "results": {name: {lead: {"1.0": r} for lead, r in rows.items()}}}


INC = _bench("old", {"30": _row(0.45, 0.73, 0.655), "60": _row(0.33, 0.59, 0.78)})


def test_gate_promotes_when_not_worse_anywhere_and_better_somewhere():
    cand = _bench("new", {"30": _row(0.46, 0.77, 0.63), "60": _row(0.34, 0.65, 0.77)})
    v = pc.compare_benchmarks(INC, cand, "old", "new")
    assert v["promote"], v["reasons"]
    assert v["n_significantly_better"] >= 2


def test_gate_holds_on_a_significant_regression_at_one_lead():
    cand = _bench("new", {"30": _row(0.46, 0.77, 0.63), "60": _row(0.28, 0.65, 0.77)})  # csi drop at 60
    v = pc.compare_benchmarks(INC, cand, "old", "new")
    assert not v["promote"]
    assert any("lead 60 csi" in r for r in v["reasons"])


def test_gate_holds_when_nothing_is_significantly_better():
    cand = _bench("new", {"30": _row(0.45, 0.735, 0.652), "60": _row(0.335, 0.595, 0.779)})
    v = pc.compare_benchmarks(INC, cand, "old", "new")
    assert not v["promote"]
    assert any("not significantly better" in r for r in v["reasons"])


def test_gate_requires_same_sample_set_and_adequacy():
    cand = _bench("new", {"30": _row(0.5, 0.8, 0.6), "60": _row(0.4, 0.7, 0.7)}, sample_hash="zzz")
    v = pc.compare_benchmarks(INC, cand, "old", "new")
    assert not v["promote"] and any("different sample sets" in r for r in v["reasons"])
    cand = _bench("new", {"30": _row(0.5, 0.8, 0.6), "60": _row(0.4, 0.7, 0.7)}, adequate=False)
    v = pc.compare_benchmarks(INC, cand, "old", "new")
    assert not v["promote"] and any("adequate" in r for r in v["reasons"])


def _npz(rates, issue=1_700_000_000):
    return {"rates": rates.astype("float32"), "leads": np.array([0, 30]), "issue_epoch": np.int64(issue)}


def test_canary_pair_accepts_similar_fields_and_rejects_bad_ones():
    rng = np.random.default_rng(0)
    old = rng.uniform(0, 3, (2, 20, 20)).astype("float32")
    new = old + rng.normal(0, 0.1, old.shape).astype("float32")
    assert pc.compare_canary_pair(_npz(old), _npz(np.clip(new, 0, None)))["ok"]
    bad = new.copy(); bad[0, 0, 0] = np.nan
    assert "non-finite" in "; ".join(pc.compare_canary_pair(_npz(old), _npz(bad))["reasons"])
    assert "issue_epoch" in "; ".join(pc.compare_canary_pair(_npz(old), _npz(new, issue=1))["reasons"])
    dry = np.zeros_like(old)
    r = pc.compare_canary_pair(_npz(old), _npz(dry))
    assert not r["ok"] and any("wet fraction" in x or "RMSE" in x for x in r["reasons"])


def test_canary_dirs_pairs_by_name_and_needs_enough_ticks(tmp_path):
    rng = np.random.default_rng(2)
    a, b = tmp_path / "old", tmp_path / "new"
    a.mkdir(); b.mkdir()
    for i in range(6):
        f = rng.uniform(0, 2, (2, 10, 10)).astype("float32")
        np.savez(a / f"t{i}.npz", **_npz(f, issue=i))
        np.savez(b / f"t{i}.npz", **_npz(f * 1.01, issue=i))
    v = pc.compare_canary_dirs(a, b, min_pairs=6)
    assert v["ok"] and v["n_pairs"] == 6
    assert not pc.compare_canary_dirs(a, b, min_pairs=8)["ok"]


def test_swap_is_atomic_keeps_previous_and_refuses_without_verdicts(tmp_path):
    cand = tmp_path / "new.pt"; cand.write_bytes(b"NEW")
    target = tmp_path / "live.pt"; target.write_bytes(b"OLD")
    with pytest.raises(SystemExit):
        pc.swap_checkpoint(cand, target, {"promote": False, "reasons": ["x"]}, {"ok": True})
    with pytest.raises(SystemExit):
        pc.swap_checkpoint(cand, target, {"promote": True}, {"ok": False, "reasons": ["y"]})
    assert target.read_bytes() == b"OLD"
    backup = pc.swap_checkpoint(cand, target, {"promote": True, "reasons": []}, {"ok": True, "reasons": []})
    assert target.read_bytes() == b"NEW" and backup.read_bytes() == b"OLD"
    rec = json.loads((tmp_path / "live.pt.promotion.json").read_text())
    assert rec["previous"].endswith(backup.name)
