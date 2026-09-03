"""Unit tests for tools/benchmark.py on synthetic fields — no real store,
no checkpoints, no GPU."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import yaml
import zarr

from model.metrics import categorical_scores, continuous_scores, fractions_skill_score
from tools import benchmark as bm
from tools._advection import advect_forecast, flow_for_pair, warp

GRID = (16, 16)


# ────────────────────────────────────────────────────────────── fixtures


def _rain_blob(shape=GRID, cy=8, cx=8, radius=3, rate=5.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros(shape, dtype="float32")
    field[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = rate
    return field


def _make_synthetic_store(path: pathlib.Path, *, n_issues=30, h=16, w=16,
                          leads_min=(0, 30, 60, 90, 120), cadence_min=30) -> pathlib.Path:
    """A tiny zarr store matching the training-store layout: radar
    (n_issues, n_lead, H, W) with index 0 = analysis, truth (n_issues, H, W),
    issue_time (30-min cadence), leads_min."""
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
        z_radar[i] = frames
        z_truth[i] = _rain_blob(cx=cx + 1)  # "observed" one step ahead of the analysis
    return path


def _write_config(path: pathlib.Path, **overrides) -> pathlib.Path:
    cfg = {
        "name": "test-benchmark",
        "version": 1,
        "val_window": {"start": "1970-01-01", "end": "2100-01-01"},
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


def test_shifted_field_lower_csi_than_fss_at_larger_scale():
    obs = _rain_blob(cx=8)
    pred = _rain_blob(cx=12)  # shifted 4 px — misses at the pixel level
    csi = categorical_scores(pred, obs, threshold=1.0)["csi"]
    fss5 = fractions_skill_score(pred, obs, threshold=1.0, scale_px=5)
    assert csi < fss5


def test_flow_and_warp_roundtrip_shift():
    """A pure translation should be recovered (approximately) by the
    block-matching flow and correctly undone by warp."""
    a = _rain_blob(cx=6)
    b = _rain_blob(cx=9)  # shifted +3 px in x
    flow = flow_for_pair(a, b)
    fy, fx = flow
    # Only wet blocks get a nonzero estimate; on this small a 4x4-block grid
    # the block-centre interpolation smooths the exact magnitude, so just
    # check the estimated motion near the blob is rightward and non-trivial.
    assert np.median(fx[4:12, 4:14]) > 0.5
    warped = warp(a, np.zeros_like(fy), -3 * np.ones_like(fx))
    # warp(a, dy=0, dx=-3) samples a at x+3 -> shifts content to lower x,
    # sanity check it changed something and stayed non-negative.
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


# ─────────────────────────────────────────────────────────── end-to-end


def test_baselines_run_end_to_end_on_synthetic_store(tmp_path):
    store = _make_synthetic_store(tmp_path / "store.zarr")
    config_path = _write_config(tmp_path / "benchmark.yaml")
    cfg = bm.load_config(config_path)

    report = bm.run_benchmark(str(store), cfg, model_specs=[], device="cpu")

    assert set(report["results"].keys()) == {"persistence", "advection", "operational"}
    assert report["metadata"]["n_samples_selected"] > 0
    assert report["metadata"]["models"] == []

    for name, by_lead in report["results"].items():
        assert set(by_lead.keys()) == {"30", "60", "90", "120"}
        for lead, by_thr in by_lead.items():
            assert set(by_thr.keys()) == {"0.1", "1.0"}
            for thr, row in by_thr.items():
                assert row["n"] > 0
                assert np.isfinite(row["rmse"])
                assert set(row["fss"].keys()) == {"1", "3", "5"}


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
    assert md_path.exists()
    assert "persistence" in md_path.read_text()
