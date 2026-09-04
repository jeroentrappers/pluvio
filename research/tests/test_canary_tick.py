"""The canary tick pairs candidate and incumbent outputs per issue, never
touches the served artifact, and is a no-op without a candidate."""

from __future__ import annotations

import numpy as np

from tools import canary_tick as ct


def _npz(path, issue, value):
    np.savez(path, rates=np.full((2, 4, 4), value, "float32"), leads=np.array([0, 30]),
             bounds=np.array([1.5, 48.9, 7.5, 52.5]), issue_epoch=np.int64(issue))


def test_noop_without_candidate(tmp_path):
    served = tmp_path / "served.npz"; _npz(served, 1000, 1.0)
    res = ct.tick(candidate=tmp_path / "missing.pt", served=served, canary_dir=tmp_path / "c",
                  zarr="z", python="py", cwd=tmp_path)
    assert res["ran"] is False and not (tmp_path / "c").exists()


def test_pairs_candidate_and_incumbent_by_issue_and_skips_repeats(tmp_path):
    served = tmp_path / "served.npz"; _npz(served, 1_700_000_000, 1.0)
    cand = tmp_path / "candidate.pt"; cand.write_bytes(b"x")
    calls = []

    def fake_runner(python, zarr, candidate, out, cwd):
        calls.append(out)
        _npz(out, 1_700_000_000, 2.0)

    res = ct.tick(candidate=cand, served=served, canary_dir=tmp_path / "c", zarr="z", python="py",
                  cwd=tmp_path, runner=fake_runner)
    assert res["ran"] and res["pairs"] == 1
    stamp = ct.issue_stamp(served)
    assert (tmp_path / "c" / "candidate" / f"{stamp}.npz").exists()
    assert (tmp_path / "c" / "incumbent" / f"{stamp}.npz").exists()
    assert np.load(served)["rates"].max() == 1.0            # served untouched
    res2 = ct.tick(candidate=cand, served=served, canary_dir=tmp_path / "c", zarr="z", python="py",
                   cwd=tmp_path, runner=fake_runner)
    assert res2["ran"] is False and "already paired" in res2["reason"] and len(calls) == 1


def test_mismatched_issue_is_discarded(tmp_path):
    served = tmp_path / "served.npz"; _npz(served, 1_700_000_000, 1.0)
    cand = tmp_path / "candidate.pt"; cand.write_bytes(b"x")

    def stale_runner(python, zarr, candidate, out, cwd):
        _npz(out, 1_700_000_000 - 1800, 2.0)                # candidate resolved the previous issue

    res = ct.tick(candidate=cand, served=served, canary_dir=tmp_path / "c", zarr="z", python="py",
                  cwd=tmp_path, runner=stale_runner)
    assert res["ran"] is False and "different issue" in res["reason"]
    assert not list((tmp_path / "c" / "candidate").glob("*.npz"))
