"""Synthetic tests for the tools/qc library and its two thin CLIs."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
import pytest

from tools.qc import checks, verdict
from tools.qc.thresholds import Thresholds, load_thresholds

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


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
        dr = round(dlat / cell)
        dc = round(dlon / cell)
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


def test_aggregate_registration_emits_two_distinct_detail_lines_when_both_trip():
    # offset AND corr both cross threshold -> the CLI splits this on "; " into
    # two separate "REGISTRATION ..." warnings, mirroring channel_health.
    th = load_thresholds()
    fits = [(0.1, 0.12, 0.0), (0.1, 0.13, 0.0)]
    c = checks.aggregate_registration(fits, th)
    assert c.status == "warn"
    parts = [p for p in c.detail.split("; ") if p]
    assert len(parts) == 2
    assert any("offset" in p for p in parts)
    assert any("corr" in p for p in parts)


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


def test_channel_health_msg_ir108_default_range_is_luminance_not_kelvin():
    # regression guard for the false alarm: msg_ir108 is band-1 luminance
    # [0, 255] of the rendered GeoTIFF, not a Kelvin brightness temperature.
    th = load_thresholds()
    block = np.full((48, 4, 4), 50.0, dtype="float32")
    c = checks.channel_health(block, "msg_ir108", th)
    assert c.status == "ok"


@pytest.mark.parametrize("name,centre,scale,inside_phys,outside_phys", [
    ("aws_pressure", 1013.0, 20.0, 1000.0, 900.0),   # band (-4.0, 2.5)
    ("aws_temp", 10.0, 10.0, 15.0, -40.0),           # band (-4.5, 3.5)
    ("aws_wind", 4.0, 4.0, 10.0, 50.0),              # band (-1.0, 8.0)
    ("aws_humidity", 70.0, 30.0, 60.0, -20.0),       # band (-2.4, 1.0)
])
def test_channel_health_per_channel_aws_bands(name, centre, scale, inside_phys, outside_phys):
    # each AWS channel is normalised independently, (value - centre) / scale
    # per build_aux.AWS_CHANNELS, so a value fine for one channel's band is
    # not necessarily fine for another — this guards the per-channel bands
    # stay distinct rather than sharing one +-3 band across all of them.
    th = load_thresholds()
    inside = (inside_phys - centre) / scale
    outside = (outside_phys - centre) / scale
    ok = checks.channel_health(np.full((48, 4, 4), inside, dtype="float32"), name, th)
    assert ok.status == "ok"
    warn = checks.channel_health(np.full((48, 4, 4), outside, dtype="float32"), name, th)
    assert warn.status == "warn"


def test_channel_health_range_check_uses_percentile_not_hard_minmax():
    # a handful of outlier cells (a bad IDW sample, one noisy report) over a
    # 48-issue x 50x50 window must not page anyone — the bulk of the
    # distribution has to move for a warning to fire.
    th = load_thresholds()
    rng = np.random.default_rng(6)
    block = rng.uniform(-4.0, 3.0, (48, 50, 50)).astype("float32")
    flat = block.reshape(-1)
    flat[:5] = 500.0  # 5 of 120000 cells (~0.004%) are wildly out of range
    c = checks.channel_health(block, "aws_temp", th)
    assert c.value["max"] == pytest.approx(500.0, rel=1e-3)
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


def test_staleness_boundary_matches_old_integer_minute_semantics():
    # the original code computed round(age_min) ONCE and compared that
    # integer to warn_min; comparing unrounded minutes moved the boundary
    # ~30s earlier, so staleness() rounds first, then compares.
    warn_min = 75.0
    now = 2_000_000.0
    just_under = checks.staleness(now - 75.4 * 60, now, warn_min)  # rounds to 75
    assert just_under.status == "ok"
    assert just_under.value == 75
    just_over = checks.staleness(now - 75.6 * 60, now, warn_min)  # rounds to 76
    assert just_over.status == "warn"
    assert just_over.value == 76


# ---------------------------------------------------------------------------
# region_metrics / evaluate_region / gauge_bias (qc_watchdog)
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


def test_evaluate_region_churn_and_interp():
    th = load_thresholds()
    m = {"assessable": True, "churn_scan_pct": 90.0, "churn_interp_ratio": 2.0,
         "parity_lag1": 0.0, "freeze_frac": 0.0}
    verdicts = checks.evaluate_region(m, None, th)
    assert "CHURN" in verdicts
    assert "INTERP" in verdicts


def test_evaluate_region_freeze():
    th = load_thresholds()
    m = {"assessable": True, "churn_scan_pct": 0.0, "churn_interp_ratio": 0.0,
         "parity_lag1": 0.0, "freeze_frac": 0.9}
    assert checks.evaluate_region(m, None, th) == ["FREEZE"]


def test_evaluate_region_gauge_bias():
    th = load_thresholds()
    m = {"assessable": False, "churn_scan_pct": None, "churn_interp_ratio": None,
         "parity_lag1": 0.0, "freeze_frac": 0.0}
    assert checks.evaluate_region(m, 10.0, th) == ["GAUGE-BIAS"]
    assert checks.evaluate_region(m, 1.0, th) == []


def test_gauge_bias_computes_mean_diff_over_stations(tmp_path):
    h = w = 20
    n = 13
    h0 = dt.datetime(2026, 9, 3, 0, 0, tzinfo=dt.UTC).timestamp()
    times = np.array([h0] + [h0 + 300 * i for i in range(1, 13)], dtype="int64")
    bounds = (2.0, 49.0, 8.0, 53.0)
    rates = np.zeros((n, h, w), dtype="float32")

    stations = [(51.0 + 0.1 * i, 5.0 + 0.1 * i) for i in range(5)]
    W, S, E, N = bounds
    for la, lo in stations:
        c = int((lo - W) / (E - W) * w)
        r = int((N - la) / (N - S) * h)
        rates[:, r, c] = 7.0  # served rate constant across the hour

    gauge_dir = tmp_path / "gauges"
    gauge_dir.mkdir()
    rows = [[la, lo, 5.0, "test"] for la, lo in stations]  # gauge saw 5mm
    (gauge_dir / "2026090300.json").write_text(json.dumps(rows))

    gb = checks.gauge_bias(rates, times, bounds, bounds, str(gauge_dir))
    # served: 12 scans x 7.0/12 mm summed = 7.0mm; gauge: 5.0mm -> bias +2.0
    assert gb == pytest.approx(2.0, abs=1e-6)


def test_gauge_bias_none_when_too_few_stations(tmp_path):
    h = w = 20
    n = 13
    h0 = dt.datetime(2026, 9, 3, 0, 0, tzinfo=dt.UTC).timestamp()
    times = np.array([h0] + [h0 + 300 * i for i in range(1, 13)], dtype="int64")
    bounds = (2.0, 49.0, 8.0, 53.0)
    rates = np.zeros((n, h, w), dtype="float32")
    gauge_dir = tmp_path / "gauges"
    gauge_dir.mkdir()
    (gauge_dir / "2026090300.json").write_text(json.dumps([[51.0, 5.0, 5.0, "test"]]))
    assert checks.gauge_bias(rates, times, bounds, bounds, str(gauge_dir)) is None


def test_gauge_bias_none_without_gauge_files(tmp_path):
    gb = checks.gauge_bias(np.zeros((1, 2, 2)), np.array([0]), (0.0, 0.0, 1.0, 1.0),
                            (0.0, 0.0, 1.0, 1.0), str(tmp_path))
    assert gb is None


# ---------------------------------------------------------------------------
# threshold prefix matching
# ---------------------------------------------------------------------------

def test_range_for_prefix_requires_trailing_underscore():
    th = Thresholds(ranges={"radar": (0.0, 400.0), "aws_": (-3.0, 3.0)})
    assert th.range_for("radar") == (0.0, 400.0)
    # a bare "radar" entry must NOT act as a prefix for a future channel
    assert th.range_for("radar_dbz") is None
    assert th.range_for("radar_quality") is None
    # "aws_" was declared WITH the trailing underscore, so it IS a prefix
    assert th.range_for("aws_temp") == (-3.0, 3.0)
    assert th.range_for("aws_anything_else") == (-3.0, 3.0)


def test_default_ranges_have_no_accidental_prefixes():
    th = load_thresholds()
    assert th.range_for("radar_dbz") is None
    assert th.range_for("truth_quality") is None
    assert th.range_for("alaro_precip_raw") is None


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


# ---------------------------------------------------------------------------
# thresholds loading
# ---------------------------------------------------------------------------

def test_thresholds_env_override(tmp_path, monkeypatch):
    override = tmp_path / "thresholds.yaml"
    override.write_text("nan_limit: 0.5\nranges:\n  alaro_precip: [0, 10]\n")
    monkeypatch.setenv("PLUVIO_QC_THRESHOLDS", str(override))
    th = load_thresholds()
    assert th.nan_limit == 0.5
    assert th.range_for("alaro_precip") == (0.0, 10.0)
    # untouched defaults still present
    assert th.range_for("aws_temp") == (-4.5, 3.5)


def test_thresholds_fixture_file_loads_and_matches_defaults():
    th = load_thresholds(str(FIXTURES / "qc_thresholds.yaml"))
    defaults = load_thresholds()
    assert th.ranges == defaults.ranges
    assert th.nan_limit == defaults.nan_limit
    assert th.range_percentile == defaults.range_percentile
    assert th.stale_warn_s == defaults.stale_warn_s


def test_load_thresholds_rejects_non_mapping_top_level(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        load_thresholds(str(p))


def test_load_thresholds_rejects_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("nan_limi: 0.5\n")  # typo: should be nan_limit
    with pytest.raises(ValueError, match="nan_limi"):
        load_thresholds(str(p))


# ---------------------------------------------------------------------------
# CLI exit-code semantics
# ---------------------------------------------------------------------------

def test_qc_watchdog_cli_exits_1_on_stale(tmp_path):
    from tools import qc_watchdog

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
    assert body["verdict"]["summary"] in ("warn", "crit")
    assert any(c["name"] == "staleness" and c["status"] != "ok" for c in body["verdict"]["checks"])


def test_qc_watchdog_cli_exits_0_when_fresh_and_quiet(tmp_path):
    from tools import qc_watchdog

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
    body = json.loads(out.read_text())
    assert body["verdict"]["summary"] == "ok"


def test_qc_inputs_cli_exits_1_on_stale_issue(tmp_path, monkeypatch):
    zarr = pytest.importorskip("zarr")
    from tools import qc_inputs

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
                         lambda src, t, thresholds, warn, all_checks: {"n": 0, "note": "skipped in test"})
    monkeypatch.setattr(qc_inputs, "aux_alignment_check",
                         lambda src, t, thresholds, warn, all_checks: {})

    out = tmp_path / "qc_inputs.json"
    rc = qc_inputs.main(["--store", str(store_path), "--out", str(out)])

    assert rc == 1
    body = json.loads(out.read_text())
    assert any("STALE" in w for w in body["warnings"])
    assert body["verdict"]["summary"] in ("warn", "crit")
    assert any(c["name"] == "staleness" and c["status"] != "ok" for c in body["verdict"]["checks"])


def test_msg_ir108_alignment_sign_is_positive_for_luminance():
    # The stored channel is rendered-image luminance (cold tops BRIGHT), so a
    # correctly registered IR field correlates positively with rain; the check
    # must use sign=+1 (sign=-1 was a false alarm on every wet hour).
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "tools" / "qc_inputs.py"
    assert '("msg_ir108", +1)' in src.read_text()
    rng = np.random.default_rng(0)
    radar = rng.uniform(0, 5, (24, 24)).astype("float32")
    lum = (40.0 * radar + rng.normal(0, 5, radar.shape)).astype("float32")  # bright where wet
    assert checks.signed_corr(radar, lum, +1) > 0.9
    assert checks.signed_corr(radar, lum, -1) < 0


def test_issue_time_order_warns_only_for_the_live_tail():
    t = np.arange(0, 3000 * 1800, 1800, dtype="int64")
    ok = checks.issue_time_order(t)
    assert ok.status == "ok" and ok.value["non_increasing_steps"] == 0
    hist = t.copy(); hist[100], hist[101] = hist[101], hist[100]      # historic swap: reported, not a warn
    r = checks.issue_time_order(hist)
    assert r.status == "ok" and r.value["non_increasing_steps"] == 1 and "historic" in r.detail
    live = t.copy(); live[-2], live[-1] = live[-1], live[-2]           # disorder in the newest issues: warn
    assert checks.issue_time_order(live).status == "warn"
