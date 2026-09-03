"""Unit tests for tools/benchmark.py on synthetic fields — no real store,
no checkpoints, no GPU."""

from __future__ import annotations

import json
import pathlib
from datetime import UTC

import numpy as np
import pytest
import yaml
import zarr
from model.metrics import categorical_scores, continuous_scores, fractions_skill_score
from tools import benchmark as bm
from tools._advection import advect_forecast, flow_for_pair, max_shift_px, warp
from tools._stats import SampleStats, block_bootstrap

GRID = (16, 16)


# ────────────────────────────────────────────────────────────── fixtures


def _rain_blob(shape=GRID, cy=None, cx=None, radius=3, rate=5.0) -> np.ndarray:
    h, w = shape
    if cy is None:
        cy = h // 2
    if cx is None:
        cx = w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros(shape, dtype="float32")
    field[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = rate
    return field


def _make_synthetic_store(path: pathlib.Path, *, n_issues=30, h=16, w=16,
                          leads_min=(0, 30, 60, 90, 120), cadence_min=30,
                          nan_corner: int = 0) -> pathlib.Path:
    """A tiny zarr store matching the training-store layout: radar
    (n_issues, n_lead, H, W) with index 0 = analysis, truth (n_issues, H, W),
    issue_time (30-min cadence), leads_min.

    ``nan_corner`` > 0 punches a fixed NaN square of that side length into
    the top-left corner of every radar/truth frame at every issue — the
    same "outside the radar domain, by construction" pattern the real store
    has (a static geometry mask, not per-frame noise).
    """
    n_lead = len(leads_min)
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("leads_min", shape=(n_lead,), dtype="int16", chunks=(n_lead,))
    root["leads_min"][:] = np.asarray(leads_min, dtype="int16")
    root.create_array("issue_time", shape=(n_issues,), dtype="int64", chunks=(n_issues,))
    base_epoch = 1_800_000_000
    epochs = base_epoch + np.arange(n_issues) * cadence_min * 60
    root["issue_time"][:] = epochs.astype("int64")

    z_radar = root.create_array("radar", shape=(n_issues, n_lead, h, w), dtype="float32",
                                chunks=(1, n_lead, h, w))
    z_truth = root.create_array("truth", shape=(n_issues, h, w), dtype="float32",
                                chunks=(1, h, w))

    rng = np.random.default_rng(0)
    # A blob that drifts one pixel to the right every issue, so analyses form
    # a genuinely advectable sequence, and the "truth" at issue+lead is just
    # the blob at the position it will have reached — giving the operational
    # nowcast (and persistence/advection) something non-trivial to be scored
    # against.
    for i in range(n_issues):
        cx = 4 + (i % 8)
        analysis = _rain_blob(cx=cx) + 0.01 * rng.standard_normal((h, w)).astype("float32")
        analysis = np.clip(analysis, 0.0, None)
        frames = np.zeros((n_lead, h, w), dtype="float32")
        frames[0] = analysis
        for li, lead in enumerate(leads_min):
            if lead == 0:
                continue
            steps_ahead = lead // cadence_min
            frames[li] = _rain_blob(cx=cx + steps_ahead)
        truth_frame = _rain_blob(cx=cx + 1)  # "observed" one step ahead of the analysis
        if nan_corner:
            frames[:, :nan_corner, :nan_corner] = np.nan
            truth_frame[:nan_corner, :nan_corner] = np.nan
        z_radar[i] = frames
        z_truth[i] = truth_frame
    return path


def _write_config(path: pathlib.Path, **overrides) -> pathlib.Path:
    cfg = {
        "name": "test-benchmark",
        "version": 1,
        "val_window": {"start": "1970-01-01", "end": "2100-01-01"},
        "val_frac_split": 0.2,
        "allow_train_overlap": True,  # tests use a window spanning the whole
                                      # tiny store, deliberately overriding
                                      # the train/val guard tested separately
        "case_days": [],
        "max_samples": 50,
        "seed": 42,
        "sample_cells": 0,
        "thresholds_mm_h": [0.1, 1.0],
        "leads_min": [30, 60, 90, 120],
        "fss_scales_px": [1, 3, 5],
    }
    cfg.update(overrides)
    path.write_text(yaml.safe_dump(cfg))
    return path


# ───────────────────────────────────────────────────────────────── metrics


def test_perfect_forecast_gives_csi1_rmse0():
    field = _rain_blob()
    cat = categorical_scores(field, field, threshold=1.0)
    cont = continuous_scores(field, field)
    assert cat["csi"] == pytest.approx(1.0)
    assert cat["pod"] == pytest.approx(1.0)
    assert cat["far"] == pytest.approx(0.0)
    assert cont["rmse"] == pytest.approx(0.0)
    assert cont["bias"] == pytest.approx(0.0)


def test_shifted_field_fss_rises_with_scale_and_scale1_is_dice():
    obs = _rain_blob(cx=8)
    pred = _rain_blob(cx=12)  # shifted 4 px — misses at the pixel level
    csi = categorical_scores(pred, obs, threshold=1.0)["csi"]
    fss1 = fractions_skill_score(pred, obs, threshold=1.0, scale_px=1)
    fss3 = fractions_skill_score(pred, obs, threshold=1.0, scale_px=3)
    fss5 = fractions_skill_score(pred, obs, threshold=1.0, scale_px=5)
    assert csi < fss5
    assert fss1 < fss3 < fss5
    # FSS at scale 1 (no neighbourhood pooling) is exactly the Dice
    # coefficient, which relates to CSI (IoU) by 2*CSI/(1+CSI).
    assert fss1 == pytest.approx(2 * csi / (1 + csi), abs=1e-6)


def test_flow_recovers_known_shift_within_half_pixel():
    """On a larger field (closer to a real analysis grid than the 16x16
    store used elsewhere), the NCC-based block matcher should recover a
    pure +3 px translation tightly, not just qualitatively."""
    shape = (128, 128)
    a = _rain_blob(shape=shape, cy=60, cx=60, radius=12, rate=5.0)
    b = _rain_blob(shape=shape, cy=60, cx=63, radius=12, rate=5.0)  # +3 px in x
    flow = flow_for_pair(a, b, max_shift=8)
    fy, fx = flow
    region = np.s_[40:80, 40:88]
    assert np.median(fx[region]) == pytest.approx(3.0, abs=0.5)
    assert np.median(fy[region]) == pytest.approx(0.0, abs=0.5)


def test_flow_and_warp_roundtrip_shift():
    """A pure translation should be recovered (approximately) by the
    block-matching flow and correctly undone by warp."""
    a = _rain_blob(cx=6)
    b = _rain_blob(cx=9)  # shifted +3 px in x
    flow = flow_for_pair(a, b)
    fy, fx = flow
    assert np.median(fx[4:12, 4:14]) > 0.5
    warped = warp(a, np.zeros_like(fy), -3 * np.ones_like(fx))
    assert warped.min() >= 0.0
    assert not np.allclose(warped, a)


def test_advect_forecast_extrapolates_motion():
    prev = _rain_blob(cx=6)
    curr = _rain_blob(cx=9)  # 3 px / step_min motion
    out = advect_forecast(prev, curr, lead_min=60.0, step_min=30.0)
    # Two more steps of the same motion -> blob should have moved further
    # right than `curr`; check mass shifted toward higher x.
    def _centroid_x(field):
        mass = field.sum()
        if mass <= 0:
            return None
        xs = np.arange(field.shape[1])
        return float((field.sum(axis=0) * xs).sum() / mass)

    assert _centroid_x(out) > _centroid_x(curr)


def test_max_shift_px_scales_with_grid_and_cadence():
    # Finer grid (smaller km/px) -> more pixels needed to cover the same
    # real-world motion in one step.
    coarse = max_shift_px(km_per_px=6.0, step_min=30.0)
    fine = max_shift_px(km_per_px=3.2, step_min=30.0)
    assert fine > coarse


# ─────────────────────────────────────────────────────────── end-to-end


def test_baselines_run_end_to_end_on_synthetic_store(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    config_path = _write_config(tmp_path / "benchmark.yaml")
    cfg = bm.load_config(config_path)

    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")

    assert set(report["results"].keys()) == {"persistence", "advection", "operational"}
    assert report["metadata"]["n_samples_selected"] > 0
    assert report["metadata"]["models"] == []
    assert report["metadata"]["sample_set_hash"]
    assert report["metadata"]["n_scores_total"] == report["metadata"]["n_samples_selected"]

    for by_lead in report["results"].values():
        assert set(by_lead.keys()) == {"30", "60", "90", "120"}
        for by_thr in by_lead.values():
            assert set(by_thr.keys()) == {"0.1", "1.0"}
            for row in by_thr.values():
                assert row["n_samples"] > 0
                assert np.isfinite(row["rmse"])
                assert np.isfinite(row["mean_error"])
                assert row["n_valid_cells"] == 16 * 16 * row["n_samples"]
                assert set(row["fss"].keys()) == {"1", "3", "5"}


def test_nan_domain_scored_on_identical_finite_support(tmp_path):
    """A store with a static NaN region (outside the radar domain) must not
    let one model's baseline see a different pixel count than another's, and
    must not turn into NaN RMSE/bias for the fields that ARE finite."""
    store = _make_synthetic_store(tmp_path / "store.zarr", nan_corner=4)
    config_path = _write_config(tmp_path / "benchmark.yaml")
    cfg = bm.load_config(config_path)

    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")

    full_cells = 16 * 16
    corner_cells = 4 * 4
    for by_lead in report["results"].values():
        for by_thr in by_lead.values():
            row = by_thr["0.1"]
            # Every sample lost exactly the NaN corner, for every model —
            # identical support.
            assert row["n_valid_cells"] == (full_cells - corner_cells) * row["n_samples"]
            assert np.isfinite(row["rmse"])
            assert np.isfinite(row["mean_error"])
            assert np.isfinite(row["csi"]) or row["csi"] != row["csi"]  # may be nan if dry


def test_cli_writes_valid_json(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    config_path = _write_config(tmp_path / "benchmark.yaml")
    out_path = tmp_path / "results.json"
    md_path = tmp_path / "results.md"

    rc = bm.main([
        "--zarr", str(store),
        "--config", str(config_path),
        "--out", str(out_path),
        "--markdown", str(md_path),
    ])
    assert rc == 0
    assert out_path.exists()

    with open(out_path) as fh:
        payload = json.load(fh)
    assert "metadata" in payload and "results" in payload
    assert payload["metadata"]["config_hash"]
    assert payload["metadata"]["sample_set_hash"]
    assert payload["metadata"]["thresholds_mm_h"] == [0.1, 1.0]
    assert md_path.exists()
    assert "persistence" in md_path.read_text()


def test_val_window_before_split_is_refused(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    # No allow_train_overlap override — 1970 is guaranteed to precede the
    # store's own (data-dependent) train/val split.
    config_path = _write_config(
        tmp_path / "benchmark.yaml",
        allow_train_overlap=False,
        val_window={"start": "1970-01-01", "end": "2100-01-01"},
    )
    cfg = bm.load_config(config_path)
    with pytest.raises(RuntimeError, match="train/val split"):
        bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")


def test_case_days_scored_in_full_and_reported_separately(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    root = zarr.open_group(str(store), mode="r")
    epochs = np.asarray(root["issue_time"][:])
    from datetime import datetime
    case_date = datetime.fromtimestamp(int(epochs[5]), tz=UTC).date().isoformat()

    config_path = _write_config(tmp_path / "benchmark.yaml", case_days=[case_date],
                                # A tiny max_samples that would normally starve
                                # everything, to prove case days bypass the cap.
                                max_samples=1)
    cfg = bm.load_config(config_path)
    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")

    assert report["metadata"]["n_case_day_samples"] > 0
    assert report["results_case_days"]
    for name in ("persistence", "advection", "operational"):
        assert name in report["results_case_days"]


# ────────────────────────────────────────────────────────────── 3.6: stats


def _bootstrap_cfg(**overrides):
    cfg = {"blocks_h": 6, "n": 300, "ci": 0.9, "seed": 7, "reference_model": "persistence"}
    cfg.update(overrides)
    return cfg


def test_bootstrap_ci_contains_the_point_estimate(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    config_path = _write_config(tmp_path / "benchmark.yaml", bootstrap=_bootstrap_cfg())
    cfg = bm.load_config(config_path)
    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")

    for lead in ("30", "60", "90", "120"):
        row = report["results"]["advection"][lead]["1.0"]
        for key in ("csi", "rmse", "mae", "mean_error"):
            ci = row["ci"][key]
            if ci["ci_lo"] is None:  # all-NaN replicate stratum, nothing to assert
                continue
            assert ci["ci_lo"] - 1e-9 <= row[key] <= ci["ci_hi"] + 1e-9


def test_bootstrap_ci_narrows_with_more_samples(tmp_path):
    cfg_overrides = {"bootstrap": _bootstrap_cfg(n=200), "leads_min": [30],
                     "thresholds_mm_h": [1.0], "max_samples": 100000}

    small_store = _make_synthetic_store(tmp_path / "small.zarr", n_issues=20)
    small_cfg = bm.load_config(_write_config(tmp_path / "small.yaml", **cfg_overrides))
    small_report = bm.run_benchmark(str(small_store), small_cfg, model_specs=[], device="cpu")

    big_store = _make_synthetic_store(tmp_path / "big.zarr", n_issues=200)
    big_cfg = bm.load_config(_write_config(tmp_path / "big.yaml", **cfg_overrides))
    big_report = bm.run_benchmark(str(big_store), big_cfg, model_specs=[], device="cpu")

    small_ci = small_report["results"]["advection"]["30"]["1.0"]["ci"]["rmse"]
    big_ci = big_report["results"]["advection"]["30"]["1.0"]["ci"]["rmse"]
    small_width = small_ci["ci_hi"] - small_ci["ci_lo"]
    big_width = big_ci["ci_hi"] - big_ci["ci_lo"]
    assert big_width < small_width


def test_bootstrap_same_seed_identical_different_seed_overlaps(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    cfg = bm.load_config(_write_config(tmp_path / "benchmark.yaml", bootstrap=_bootstrap_cfg(seed=7)))

    row_a = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")[
        "results"]["advection"]["30"]["1.0"]["ci"]
    row_b = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")[
        "results"]["advection"]["30"]["1.0"]["ci"]
    assert row_a == row_b  # same seed -> byte-identical

    cfg_other = bm.load_config(
        _write_config(tmp_path / "benchmark2.yaml", bootstrap=_bootstrap_cfg(seed=99)))
    row_c = bm.run_benchmark(str(store), cfg_other, model_specs=[], device="cpu")[
        "results"]["advection"]["30"]["1.0"]["ci"]

    assert row_c["csi"] != row_a["csi"]  # different seed -> different draw
    lo = max(row_a["csi"]["ci_lo"], row_c["csi"]["ci_lo"])
    hi = min(row_a["csi"]["ci_hi"], row_c["csi"]["ci_hi"])
    assert lo <= hi  # ...but still overlapping (same underlying data)


def _block_correlated_records(n=240, cadence_s=1800, block_h=6.0, seed=0):
    """Per-sample RMSE-style stats where the error is dominated by a
    per-block random offset (correlated within each ``block_h``-hour window)
    plus small iid noise — the pattern a real storm scored at several leads
    across a case day would leave in the accumulated statistics."""
    rng = np.random.default_rng(seed)
    span = max(1, int(block_h * 3600 / cadence_s))
    records = []
    epoch0 = 1_700_000_000
    for i in range(n):
        block_rng = np.random.default_rng(1000 + i // span)
        e = float(block_rng.normal(0.0, 2.0)) + float(rng.normal(0.0, 0.05))
        records.append({
            "issue_epoch": epoch0 + i * cadence_s, "n": 1, "sum_e": e,
            "sum_abs_e": abs(e), "sum_sq_e": e * e,
            "cat": {1.0: (1, 0, 0)}, "fss": {1.0: {1: (0.0, 1.0)}},
        })
    return records


def _rmse_ci_width(records, blocks_h, n_boot=400, seed=1):
    stats = SampleStats([1.0], [1])
    for r in records:
        stats.add(**r)
    boot = block_bootstrap({"model": stats}, blocks_h=blocks_h, n_boot=n_boot, ci=0.9, seed=seed)
    ci = boot["ci"]["model"]["1.0"]["rmse"]
    return ci["ci_hi"] - ci["ci_lo"]


def test_block_bootstrap_respects_correlated_blocks():
    """Resampling in 6 h blocks on data whose errors are correlated within
    each 6 h window must give a wider CI than resampling at the (here,
    effectively per-sample / iid) 30-min block granularity — a plain iid
    bootstrap over samples would understate the true uncertainty."""
    records = _block_correlated_records()
    wide = _rmse_ci_width(records, blocks_h=6.0)
    narrow = _rmse_ci_width(records, blocks_h=0.5)
    assert wide > narrow


def test_adequacy_flag_flips_at_threshold(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")  # blob rate is exactly 5.0 mm/h
    low_cfg = bm.load_config(_write_config(
        tmp_path / "adequate.yaml", adequacy={"threshold_mm_h": 4.0, "min_events": 1}))
    report = bm.run_benchmark(str(store), low_cfg, model_specs=[], device="cpu")
    n_events = report["metadata"]["adequacy"]["n_events"]
    assert n_events > 0
    assert report["metadata"]["adequate"] is True

    high_cfg = bm.load_config(_write_config(
        tmp_path / "inadequate.yaml", adequacy={"threshold_mm_h": 4.0, "min_events": n_events + 1}))
    report2 = bm.run_benchmark(str(store), high_cfg, model_specs=[], device="cpu")
    assert report2["metadata"]["adequacy"]["n_events"] == n_events
    assert report2["metadata"]["adequate"] is False


def test_stratified_sampling_equal_counts_per_lead(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr", n_issues=60)
    config_path = _write_config(tmp_path / "benchmark.yaml", max_samples=20, case_days=[])
    cfg = bm.load_config(config_path)
    from model.zarr_dataset import ZarrCorrectionDataset

    dataset = ZarrCorrectionDataset(str(store), leads_min=tuple(cfg["leads_min"]), build_index=True)
    selected, case_idx = bm._select_samples(dataset, cfg)
    assert not case_idx

    counts: dict[int, int] = {}
    for i in selected:
        lead = dataset.index[i].lead_min
        counts[lead] = counts.get(lead, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_manifest_sidecar_hash_matches_metadata_and_roundtrips(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    config_path = _write_config(tmp_path / "benchmark.yaml")
    out_path = tmp_path / "results.json"

    rc = bm.main(["--zarr", str(store), "--config", str(config_path), "--out", str(out_path)])
    assert rc == 0

    manifest_path = tmp_path / "results.json.samples.jsonl"
    assert manifest_path.exists()
    payload = json.loads(out_path.read_text())
    assert "manifest" not in payload  # sidecar file only, not duplicated in the JSON report

    records = bm.load_manifest(manifest_path)
    assert len(records) == payload["metadata"]["n_samples_selected"]
    assert bm.manifest_hash(records) == payload["metadata"]["sample_set_hash"]

    rec = records[0]
    assert set(rec) == {"issue_time", "lead_min", "target_time", "case_day", "n_valid_cells"}


def test_sample_set_hash_changes_with_different_sample_set(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr", n_issues=30)
    cfg_small = bm.load_config(_write_config(tmp_path / "small.yaml", max_samples=8))
    cfg_big = bm.load_config(_write_config(tmp_path / "big.yaml", max_samples=40))
    h_small = bm.run_benchmark(str(store), cfg_small, model_specs=[], device="cpu")["metadata"]["sample_set_hash"]
    h_big = bm.run_benchmark(str(store), cfg_big, model_specs=[], device="cpu")["metadata"]["sample_set_hash"]
    assert h_small != h_big


def test_paired_diff_ci_is_exact_for_a_constant_offset():
    """model B == model A with a constant offset added to every error. The
    paired-difference mean_error CI must collapse to exactly that offset
    (width ~0) for EVERY bootstrap draw, since B - A cancels the resampled
    noise term-by-term — this only holds if both models are aggregated from
    the same block draw each replicate (a genuinely paired comparison)."""
    thresholds, fss_scales = [1.0], [1]
    rng = np.random.default_rng(3)
    offset = 0.37
    common = {"cat": {1.0: (1, 0, 0)}, "fss": {1.0: {1: (0.0, 1.0)}}}
    stats_a = SampleStats(thresholds, fss_scales)
    stats_b = SampleStats(thresholds, fss_scales)
    epoch0 = 1_700_000_000
    for i in range(60):
        n = 10
        sum_e = float(rng.normal(0.0, 1.0) * n)
        stats_a.add(issue_epoch=epoch0 + i * 1800, n=n, sum_e=sum_e,
                    sum_abs_e=abs(sum_e), sum_sq_e=sum_e ** 2, **common)
        sum_e_b = sum_e + offset * n
        stats_b.add(issue_epoch=epoch0 + i * 1800, n=n, sum_e=sum_e_b,
                    sum_abs_e=abs(sum_e_b), sum_sq_e=sum_e_b ** 2, **common)

    boot = block_bootstrap({"A": stats_a, "B": stats_b}, blocks_h=6.0, n_boot=200, ci=0.9,
                          seed=5, ref_model="A")
    diff = boot["diff_vs_ref"]["B"]["1.0"]["mean_error"]
    assert (diff["ci_hi"] - diff["ci_lo"]) < 1e-9
    assert diff["ci_lo"] == pytest.approx(offset, abs=1e-9)
    assert diff["ci_hi"] == pytest.approx(offset, abs=1e-9)


def test_sample_stat_record_keys_by_issue_time_not_target_time():
    """Guards the exact line that ties a sample's sufficient statistics to
    its ISSUE time — if it were keyed by the lead-shifted target time
    instead, samples from one issue scored at different leads would land in
    different bootstrap blocks depending on lead, silently breaking the
    "resample by storm" assumption the whole feature rests on."""
    from types import SimpleNamespace

    s = SimpleNamespace(issue_epoch=1_700_000_000, lead_min=120)
    pred_sel = obs_sel = np.array([1.0, 2.0])
    pred_fss = obs_fss = np.zeros((4, 4))
    rec = bm._sample_stat_record(s, pred_sel, obs_sel, pred_fss, obs_fss, n_selected=2,
                                 thresholds=[1.0], fss_scales=[1])
    assert rec["issue_epoch"] == s.issue_epoch
    assert rec["issue_epoch"] != s.issue_epoch + s.lead_min * 60


def test_issue_block_groups_by_issue_time_even_when_targets_straddle_a_boundary():
    """Two issues that share a 6 h block by issue time, but whose targets
    (issue + a large, per-sample lead) fall in DIFFERENT blocks — proof that
    grouping by issue time (the correct key) is not incidentally equivalent
    to grouping by target time here."""
    from tools._stats import issue_block

    blocks_h = 6.0
    span_s = int(blocks_h * 3600)
    issue_a, lead_a_s = -100, 0            # target = -100
    issue_b, lead_b_s = -50, 24_000        # target = 23_950

    assert issue_block(issue_a, blocks_h) == issue_block(issue_b, blocks_h)  # same issue block
    target_a, target_b = issue_a + lead_a_s, issue_b + lead_b_s
    assert issue_block(target_a, blocks_h) != issue_block(target_b, blocks_h)  # different target block
    assert span_s > 0  # sanity: span computed as expected


def test_adequacy_counts_distinct_issue_times_not_samples(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr", n_issues=10,
                                  leads_min=(0, 30, 60, 90, 120))
    config_path = _write_config(tmp_path / "benchmark.yaml",
                                adequacy={"threshold_mm_h": 4.0, "min_events": 1},
                                leads_min=[30, 60, 90, 120], max_samples=100000, case_days=[])
    cfg = bm.load_config(config_path)
    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")

    distinct_issues = len({rec["issue_time"] for rec in report["manifest"]})
    n_events = report["metadata"]["adequacy"]["n_events"]
    assert n_events == distinct_issues
    assert n_events != len(report["manifest"])  # would be ~4x distinct_issues if mis-counted per-sample


def test_adequacy_anchored_on_scored_truth_not_t0_analysis(tmp_path):
    """A store where the t0 (issue-time) analysis is always dry but the
    scored truth at some lead is wet must still count as an adequacy event —
    adequacy has to reflect what the metrics were actually scored against."""
    store = _make_synthetic_store(tmp_path / "store.zarr", n_issues=10, leads_min=(0, 30))
    root = zarr.open_group(str(store), mode="a")
    root["radar"][:, 0, :, :] = 0.0  # every issue's t0 analysis: bone dry
    # truth (the scored target at the only non-zero lead) is untouched — the
    # synthetic fixture already writes a real blob there.

    config_path = _write_config(tmp_path / "benchmark.yaml", leads_min=[30],
                                adequacy={"threshold_mm_h": 4.0, "min_events": 1}, case_days=[])
    cfg = bm.load_config(config_path)
    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")
    assert report["metadata"]["adequacy"]["n_events"] > 0
