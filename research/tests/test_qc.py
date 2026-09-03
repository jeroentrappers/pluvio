"""Synthetic tests for the tools/qc library and its two thin CLIs."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from tools.qc import checks, verdict
from tools.qc.thresholds import Thresholds, load_thresholds


def _smooth_field(rng, shape):
    a = rng.random(shape).astype("float64")
    for _ in range(3):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
               + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a


# ---------------------------------------------------------------------------
# registration_offset
# ---------------------------------------------------------------------------

def test_registration_offset_recovers_planted_shift():
    rng = np.random.default_rng(0)
    h, w, margin = 40, 40, 10
    canvas = _smooth_field(rng, (h + 2 * margin, w + 2 * margin))
    cell = 0.02
    true_dlat, true_dlon = 0.04, -0.02  # exact multiples of `step_deg` below

    def sampler(dlat: float, dlon: float) -> np.ndarray:
        dr = int(round(dlat / cell))
        dc = int(round(dlon / cell))
        r0, c0 = margin - dr, margin + dc
        return canvas[r0:r0 + h, c0:c0 + w]

    field = sampler(true_dlat, true_dlon)
    corr, dlat, dlon = checks.registration_offset(field, sampler, search_deg=0.14, step_deg=cell)

    assert dlat == pytest.approx(true_dlat, abs=1e-9)
    assert dlon == pytest.approx(true_dlon, abs=1e-9)
    assert corr > 0.99


def test_aggregate_registration_warns_on_offset_and_low_corr():
    th = load_thresholds()
    # offset exceeds threshold
    fits = [(0.9, 0.10, 0.0), (0.9, 0.11, 0.01), (0.9, 0.09, 0.0)]
    c = checks.aggregate_registration(fits, th)
    assert c.status == "warn"
    assert "offset" in c.detail

    # in-band offset, healthy corr -> ok
    fits_ok = [(0.6, 0.01, -0.01), (0.62, 0.0, 0.01)]
    c_ok = checks.aggregate_registration(fits_ok, th)
    assert c_ok.status == "ok"

    # no fits at all
    c_empty = checks.aggregate_registration([], th)
    assert c_empty.status == "ok"
    assert c_empty.value == {"n": 0, "note": "no wet overlapping issues to fit"}


# ---------------------------------------------------------------------------
# aux alignment
# ---------------------------------------------------------------------------

def test_signed_corr_sign_logic():
    rng = np.random.default_rng(1)
    radar = rng.random((30, 30))
    aux_pos = radar * 2 + 0.01 * rng.random((30, 30))   # positively coupled
    aux_neg = -radar + 0.01 * rng.random((30, 30))      # negatively coupled

    assert checks.signed_corr(radar, aux_pos, +1) > 0.9
    assert checks.signed_corr(radar, aux_neg, -1) > 0.9   # sign flip recovers coupling
    assert checks.signed_corr(radar, aux_pos, -1) < -0.9  # wrong sign reads as anti-coupled


def test_aggregate_aux_alignment_warns_below_threshold():
    th = load_thresholds()
    warn = checks.aggregate_aux_alignment("alaro_precip", [0.01, -0.02, 0.0], th)
    assert warn.status == "warn"
    ok = checks.aggregate_aux_alignment("alaro_precip", [0.4, 0.5, 0.45], th)
    assert ok.status == "ok"
    empty = checks.aggregate_aux_alignment("alaro_precip", [], th)
    assert empty.status == "ok"
    assert empty.value is None


# ---------------------------------------------------------------------------
# channel_health
# ---------------------------------------------------------------------------

def test_channel_health_flags_all_nan_block():
    th = load_thresholds()
    block = np.full((48, 10, 10), np.nan, dtype="float32")
    c = checks.channel_health(block, "sst", th)
    assert c.status == "warn"
    assert c.value["nan_frac"] == 1.0
    assert "NaN" in c.detail


def test_channel_health_flags_out_of_range_block():
    th = load_thresholds()
    rng = np.random.default_rng(2)
    # alaro_precip default range is (0, 255) — push well past it
    block = rng.uniform(0, 1000, (48, 5, 5)).astype("float32")
    c = checks.channel_health(block, "alaro_precip", th)
    assert c.status == "warn"
    assert "out of range" in c.detail


def test_channel_health_passes_normalised_in_range_block():
    th = load_thresholds()
    rng = np.random.default_rng(3)
    # matches the OBSERVED aws_temp range from the first production run
    block = rng.uniform(-0.87, 2.51, (48, 5, 5)).astype("float32")
    c = checks.channel_health(block, "aws_temp", th)
    assert c.status == "ok"
    assert c.value["nan_frac"] == 0.0


def test_channel_health_msg_ir108_default_range_is_not_kelvin():
    # regression guard for the false alarm: observed range [4.26, 239], not Kelvin
    th = load_thresholds()
    block = np.full((48, 4, 4), 50.0, dtype="float32")
    c = checks.channel_health(block, "msg_ir108", th)
    assert c.status == "ok"


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------

def test_staleness_threshold():
    th = load_thresholds()
    now = 1_000_000.0
    ok = checks.staleness(now - 60 * 10, now, th.stale_warn_min)
    assert ok.status == "ok"
    warn = checks.staleness(now - 60 * 100, now, th.stale_warn_min)
    assert warn.status == "warn"
    assert "min old" in warn.detail


# ---------------------------------------------------------------------------
# region_metrics / evaluate_region (qc_watchdog)
# ---------------------------------------------------------------------------

def test_region_metrics_detects_parity_pulsation():
    h = w = 6
    n = 20
    times = np.arange(0, 300 * n, 300)
    rates = np.zeros((n, h, w), dtype="float32")
    rates[0::2] = 1.0  # fully wet
    rates[1::2] = 0.0  # fully dry -> area alternates 100%/0% each scan
    bounds = (0.0, 0.0, 10.0, 10.0)
    box = (0.0, 0.0, 10.0, 10.0)

    m = checks.region_metrics(rates, times, bounds, box)
    assert m is not None
    assert m["parity_lag1"] < -0.9

    th = load_thresholds()
    verdicts = checks.evaluate_region(m, None, th)
    assert "PARITY-PULSE" in verdicts


def test_region_metrics_slowly_evolving_field_is_quiet():
    # a healthy field: mostly-wet mask that drifts a little each scan
    # (a handful of cells flip), never frozen solid and never alternating.
    rng = np.random.default_rng(5)
    h, w = 12, 12
    n = 20
    times = np.arange(0, 300 * n, 300)
    wet = rng.random((h, w)) > 0.5
    frames = [wet.copy()]
    for _ in range(n - 1):
        flip = rng.random((h, w)) < 0.05
        wet = np.logical_xor(wet, flip)
        frames.append(wet.copy())
    rates = np.where(np.stack(frames), 1.0, 0.0).astype("float32")
    bounds = (0.0, 0.0, 10.0, 10.0)
    box = (0.0, 0.0, 10.0, 10.0)

    m = checks.region_metrics(rates, times, bounds, box)
    th = load_thresholds()
    verdicts = checks.evaluate_region(m, None, th)
    assert verdicts == []


# ---------------------------------------------------------------------------
# verdict schema
# ---------------------------------------------------------------------------

def test_verdict_json_serialises_without_numpy_types():
    c1 = verdict.Check("channel:sst", "warn", np.float32(1.0), np.float64(0.9), "100% NaN")
    c2 = verdict.Check("staleness", "ok", np.int64(12), 75.0)
    v = verdict.build_verdict([c1, c2], generated="2026-09-03T00:00:00+00:00")

    text = verdict.to_json(v)
    parsed = json.loads(text)  # raises if numpy types leaked through

    assert parsed["summary"] == "warn"
    assert parsed["generated"] == "2026-09-03T00:00:00+00:00"
    assert len(parsed["checks"]) == 2
    assert isinstance(parsed["checks"][0]["value"], float)
    assert isinstance(parsed["checks"][1]["value"], int)
    assert verdict.exit_code(v) == 1
    assert verdict.exit_code(verdict.build_verdict([c2])) == 0


def test_check_rejects_bad_status():
    with pytest.raises(ValueError):
        verdict.Check("x", "bad-status")


def test_thresholds_env_override(tmp_path, monkeypatch):
    override = tmp_path / "thresholds.yaml"
    override.write_text("nan_limit: 0.5\nranges:\n  alaro_precip: [0, 10]\n")
    monkeypatch.setenv("PLUVIO_QC_THRESHOLDS", str(override))
    th = load_thresholds()
    assert th.nan_limit == 0.5
    assert th.range_for("alaro_precip") == (0.0, 10.0)
    # untouched defaults still present
    assert th.range_for("aws_temp") == (-3.0, 3.0)


# ---------------------------------------------------------------------------
# CLI exit-code semantics
# ---------------------------------------------------------------------------

def test_qc_watchdog_cli_exits_1_on_stale(tmp_path):
    import tools.qc_watchdog as qc_watchdog

    now = dt.datetime.now(dt.UTC).timestamp()
    n, h, w = 10, 8, 8
    times = (now - 3600 * 5 + np.arange(n) * 300).astype("int64")  # 5h stale
    rates = np.random.default_rng(4).uniform(0, 1, (n, h, w)).astype("float32")
    bounds = np.array([2.0, 49.0, 8.0, 53.0], dtype="float64")

    npz = tmp_path / "observed.npz"
    np.savez(npz, times=times, rates=rates, bounds=bounds)
    out = tmp_path / "qc_status.json"

    rc = qc_watchdog.main(["--npz", str(npz), "--gauge-dir", str(tmp_path / "no-gauges"),
                           "--out", str(out)])
    assert rc == 1
    body = json.loads(out.read_text())
    assert body["staleness_s"] > qc_watchdog.load_thresholds().stale_warn_s


def test_qc_watchdog_cli_exits_0_when_fresh_and_quiet(tmp_path):
    import tools.qc_watchdog as qc_watchdog

    now = dt.datetime.now(dt.UTC).timestamp()
    n, h, w = 10, 8, 8
    times = (now - n * 300 + np.arange(n) * 300).astype("int64")
    rates = np.zeros((n, h, w), dtype="float32")  # bone dry everywhere -> not assessable
    bounds = np.array([2.0, 49.0, 8.0, 53.0], dtype="float64")

    npz = tmp_path / "observed.npz"
    np.savez(npz, times=times, rates=rates, bounds=bounds)
    out = tmp_path / "qc_status.json"

    rc = qc_watchdog.main(["--npz", str(npz), "--gauge-dir", str(tmp_path / "no-gauges"),
                           "--out", str(out)])
    assert rc == 0


def test_qc_inputs_cli_exits_1_on_stale_issue(tmp_path, monkeypatch):
    zarr = pytest.importorskip("zarr")
    import tools.qc_inputs as qc_inputs

    n, h, w = 5, 6, 6
    old_epoch = int(dt.datetime.now(dt.UTC).timestamp() - 3600 * 10)  # way past 75 min
    issue_time = old_epoch + np.arange(n) * 1800

    store_path = tmp_path / "store.zarr"
    grp = zarr.open_group(str(store_path), mode="w")
    grp.create_array("issue_time", shape=(n,), dtype="int64")[:] = issue_time
    grp.create_array("radar", shape=(n, 1, h, w), dtype="float32")[:] = 0.0

    # registration/aux need model.geo + the observed npz — bypass both here,
    # the exit-1 semantics under test come from staleness + channel_health.
    monkeypatch.setattr(qc_inputs, "registration_check",
                         lambda src, t, thresholds, warn: {"n": 0, "note": "skipped in test"})
    monkeypatch.setattr(qc_inputs, "aux_alignment_check",
                         lambda src, t, thresholds, warn: {})

    out = tmp_path / "qc_inputs.json"
    rc = qc_inputs.main(["--store", str(store_path), "--out", str(out)])

    assert rc == 1
    body = json.loads(out.read_text())
    assert any("STALE" in w for w in body["warnings"])
