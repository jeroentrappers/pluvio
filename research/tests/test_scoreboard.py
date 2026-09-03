"""Tests for tools.scoreboard against a tiny synthetic fixture (forecast
archive npz + QPE day-zarr + Buienradar JSONL) — no servers, no network.

Hand-computed reference (documented inline) exercises the same code path
production runs: SampleStats / block_bootstrap from tools/_stats.py, and
external_baselines.score_against_truth for the point comparison.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
import pytest
from tools import scoreboard as sb

DAY = dt.date(2026, 9, 2)
ISSUE_DT = dt.datetime(2026, 9, 2, 0, 0, tzinfo=dt.UTC)
ISSUE_EPOCH = int(ISSUE_DT.timestamp())
LEAD_MIN = 30
VALID_EPOCH = ISSUE_EPOCH + LEAD_MIN * 60  # slot 6 of the day (1800s / 300s)

QPE_BOUNDS = (0.0, 0.0, 4.0, 4.0)  # simple square domain, degrees
FC_BOUNDS = (0.0, 0.0, 4.0, 4.0)   # forecast covers the same full domain

# Truth composite (4x4), row 0 = north (matches production convention).
TRUTH_FULL = np.array([
    [0.0, 0.0, 2.0, 2.0],
    [0.0, 0.0, 2.0, 2.0],
    [3.0, 3.0, 0.5, 5.0],
    [3.0, 3.0, 0.5, 5.0],
], dtype="float32")
# block-mean downsample to 2x2 (quadrant means): [[0,2],[3, (0.5+5+0.5+5)/4]]
TRUTH_2X2 = np.array([[0.0, 2.0], [3.0, 2.75]], dtype="float32")

# Forecast prediction (2x2) for lead 30.
PRED_2X2 = np.array([[0.0, 0.5], [4.0, 5.0]], dtype="float32")


def _write_forecast_npz(root: pathlib.Path, kind: str, issue_epoch: int,
                        leads_min, rates: np.ndarray, bounds=FC_BOUNDS) -> pathlib.Path:
    ts = dt.datetime.fromtimestamp(issue_epoch, dt.UTC)
    day_dir = root / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{kind}_{ts:%H%M}.npz"
    np.savez_compressed(path, leads=np.asarray(leads_min, dtype="int32"),
                        rates=rates.astype("float16"),
                        bounds=np.asarray(bounds, dtype="float64"),
                        issue_epoch=np.int64(issue_epoch))
    return path


def _write_qpe_day(root: pathlib.Path, day: dt.date, truth_full: np.ndarray,
                   fill_slot: int) -> pathlib.Path:
    import zarr

    path = root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    rate = np.full((288, *truth_full.shape), np.nan, dtype="float16")
    rate[fill_slot] = truth_full.astype("float16")
    g.create_array("rate", data=rate, chunks=(1, *truth_full.shape))
    return path


def _write_buienradar_jsonl(root: pathlib.Path, day: dt.date, rows: list[dict]) -> pathlib.Path:
    path = root / "buienradar" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


@pytest.fixture()
def fixture_root(tmp_path):
    forecast_archive = tmp_path / "forecast_archive"
    qpe_root = tmp_path / "qpe"
    external_archive = tmp_path / "external_baselines"

    _write_forecast_npz(forecast_archive, "forecast", ISSUE_EPOCH, [LEAD_MIN],
                        PRED_2X2[None, :, :])
    slot = int(round((VALID_EPOCH % 86400) / 300))
    assert slot == 6
    _write_qpe_day(qpe_root, DAY, TRUTH_FULL, fill_slot=slot)

    buien_rows = [
        {"source": "buienradar", "station": "A", "lat": 2.0, "lon": 2.0,
         "issue_epoch": ISSUE_EPOCH, "fetch_epoch": ISSUE_EPOCH, "valid_epoch": VALID_EPOCH,
         "lead_min": LEAD_MIN, "mm_per_h": 4.5},
        {"source": "buienradar", "station": "B", "lat": 0.5, "lon": 0.5,
         "issue_epoch": ISSUE_EPOCH, "fetch_epoch": ISSUE_EPOCH, "valid_epoch": VALID_EPOCH,
         "lead_min": LEAD_MIN, "mm_per_h": 2.0},
    ]
    _write_buienradar_jsonl(external_archive, DAY, buien_rows)

    return {"forecast_archive": forecast_archive, "qpe_root": qpe_root,
           "external_archive": external_archive, "out_root": tmp_path / "scoreboard_out"}


# ---------------------------------------------------------------------------
# grid scoring: hand-computed reference
# ---------------------------------------------------------------------------

def test_score_grid_day_matches_hand_computation(fixture_root):
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast",), thresholds=(0.1, 1.0), fss_scales=(1,),
                            bootstrap_cfg=None)
    row_thr = out["results"]["forecast"][str(LEAD_MIN)]

    # pred_sel = [0, 0.5, 4, 5] (row-major PRED_2X2), obs_sel = TRUTH_2X2 flattened
    # = [0, 2, 3, 2.75] (both resampled onto the 2x2 forecast grid).
    pred_sel = PRED_2X2.flatten()
    obs_sel = TRUTH_2X2.flatten()
    e = pred_sel - obs_sel
    assert row_thr["0.1"]["mean_error"] == pytest.approx(float(e.mean()))
    assert row_thr["0.1"]["rmse"] == pytest.approx(float(np.sqrt((e ** 2).mean())))
    assert row_thr["0.1"]["mae"] == pytest.approx(float(np.abs(e).mean()))

    # thr=0.1: wet everywhere except pred[0]=0 / obs[0]=0 -> 1 dry cell agrees,
    # 3 wet cells agree -> hits=3, misses=0, fa=0 -> csi=1.
    assert row_thr["0.1"]["hits"] == 3
    assert row_thr["0.1"]["misses"] == 0
    assert row_thr["0.1"]["false_alarms"] == 0
    assert row_thr["0.1"]["csi"] == pytest.approx(1.0)

    # thr=1.0: pred_sel=[0,0.5,4,5] -> wet=[F,F,T,T]; obs_sel (resampled to
    # the 2x2 forecast grid) = [0,2,3,2.75] -> wet=[F,T,T,T]. idx0 dry/dry
    # (neither), idx1 miss, idx2 hit, idx3 hit -> hits=2, misses=1, fa=0.
    assert row_thr["1.0"]["hits"] == 2
    assert row_thr["1.0"]["misses"] == 1
    assert row_thr["1.0"]["false_alarms"] == 0
    assert row_thr["1.0"]["csi"] == pytest.approx(2 / 3)
    assert row_thr["1.0"]["pod"] == pytest.approx(2 / 3)
    assert row_thr["1.0"]["far"] == pytest.approx(0.0)

    assert out["n_issues"] == {"forecast": 1}


def test_score_grid_day_fss_scale_1_equals_pointwise(fixture_root):
    # At scale_px=1 the FSS "fraction field" IS the binary wet mask, so FSS
    # numerator/denominator reduce to a pointwise comparison of the full
    # (fill-consistent) prediction/observation fields (not just the selected
    # cells — there's no invalid cell in this fixture, so the two coincide).
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast",), thresholds=(1.0,), fss_scales=(1,),
                            bootstrap_cfg=None)
    row = out["results"]["forecast"][str(LEAD_MIN)]["1.0"]
    pf = (PRED_2X2 >= 1.0).astype("float64")
    po = (TRUTH_2X2 >= 1.0).astype("float64")
    expected_fss = 1.0 - float(((pf - po) ** 2).mean()) / (float((pf ** 2).mean()) + float((po ** 2).mean()))
    assert row["fss"]["1"] == pytest.approx(expected_fss)


def test_score_grid_day_bootstrap_ci_present(fixture_root):
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    boot_cfg = {"n": 50, "ci": 0.9, "blocks_h": 6.0, "seed": 1, "reference_model": None}
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast",), thresholds=(0.1,), fss_scales=(1,),
                            bootstrap_cfg=boot_cfg)
    row = out["results"]["forecast"][str(LEAD_MIN)]["0.1"]
    assert row["ci"] is not None
    assert "ci_lo" in row["ci"]["csi"]


# ---------------------------------------------------------------------------
# point scoring: identical truth samples for buienradar and "ours"
# ---------------------------------------------------------------------------

def test_score_points_day_identical_truth_samples(fixture_root):
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    out = sb.score_points_day(DAY, fixture_root["external_archive"],
                              fixture_root["forecast_archive"], truth,
                              kind="forecast", thresholds=(0.1, 1.0))
    assert out["n_matched"] == 2
    assert out["stations"] == ["A", "B"]

    # station A: lat=2, lon=2 -> full-res truth cell TRUTH_FULL[2,2]=0.5.
    # station B: lat=0.5, lon=0.5 -> TRUTH_FULL[3,0]=3.0.
    # bias/rmse for lead 30 computed from exactly those two (pred, truth) pairs.
    buien = out["buienradar"][str(LEAD_MIN)]
    ours = out["ours"][str(LEAD_MIN)]
    assert buien["n"] == ours["n"] == 2

    truths = [0.5, 3.0]
    buien_preds = [4.5, 2.0]
    buien_bias = sum(p - t for p, t in zip(buien_preds, truths)) / 2
    assert buien["bias"] == pytest.approx(buien_bias)

    # "ours" prediction sampled from the same 2x2 forecast grid at the same
    # points: station A -> PRED_2X2[1,1]=5.0, station B -> PRED_2X2[1,0]=4.0.
    ours_preds = [5.0, 4.0]
    ours_bias = sum(p - t for p, t in zip(ours_preds, truths)) / 2
    assert ours["bias"] == pytest.approx(ours_bias)


def test_score_points_day_drops_rows_with_no_truth(fixture_root):
    # A station outside the QPE domain must be dropped from BOTH series, not
    # scored with a fabricated truth value.
    extra = {"source": "buienradar", "station": "OUTSIDE", "lat": 99.0, "lon": 99.0,
            "issue_epoch": ISSUE_EPOCH, "fetch_epoch": ISSUE_EPOCH, "valid_epoch": VALID_EPOCH,
            "lead_min": LEAD_MIN, "mm_per_h": 1.0}
    path = fixture_root["external_archive"] / "buienradar" / f"{DAY:%Y}" / f"{DAY:%m}" / f"{DAY:%d}.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps(extra, sort_keys=True) + "\n")

    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    out = sb.score_points_day(DAY, fixture_root["external_archive"],
                              fixture_root["forecast_archive"], truth, kind="forecast")
    assert out["n_matched"] == 2
    assert "OUTSIDE" not in out["stations"]


# ---------------------------------------------------------------------------
# archive round-trip
# ---------------------------------------------------------------------------

def test_record_round_trips(fixture_root):
    record = sb.run(DAY, forecast_archive=fixture_root["forecast_archive"],
                    qpe_root=fixture_root["qpe_root"],
                    external_archive=fixture_root["external_archive"],
                    out_root=fixture_root["out_root"], qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
                    thresholds=(0.1, 1.0), fss_scales=(1,),
                    bootstrap_cfg={"n": 20, "ci": 0.9, "blocks_h": 6.0, "seed": 1,
                                   "reference_model": None})
    path = sb.write_record(record, fixture_root["out_root"])
    assert path == sb.archive_path(fixture_root["out_root"], DAY)
    assert path.exists()

    loaded = sb.load_record(fixture_root["out_root"], DAY)
    assert loaded["day"] == record["day"]
    assert loaded["adequacy"] == record["adequacy"]
    assert loaded["grid"]["n_issues"] == record["grid"]["n_issues"]
    assert (loaded["grid"]["results"]["forecast"][str(LEAD_MIN)]["0.1"]["csi"]
           == pytest.approx(record["grid"]["results"]["forecast"][str(LEAD_MIN)]["0.1"]["csi"]))
    assert loaded["points"]["n_matched"] == record["points"]["n_matched"]


# ---------------------------------------------------------------------------
# adequacy: this fixture's one issue has no truth cell > 5.0 mm/h -> inadequate
# ---------------------------------------------------------------------------

def test_inadequate_day_is_flagged(fixture_root):
    record = sb.run(DAY, forecast_archive=fixture_root["forecast_archive"],
                    qpe_root=fixture_root["qpe_root"],
                    external_archive=fixture_root["external_archive"],
                    out_root=fixture_root["out_root"], qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
                    thresholds=(0.1,), fss_scales=(1,), bootstrap_cfg=None,
                    adequacy_min_events=5)
    assert record["adequacy"]["n_events"] == 0
    assert record["adequacy"]["adequate"] is False


def test_adequate_day_when_events_meet_minimum(fixture_root):
    record = sb.run(DAY, forecast_archive=fixture_root["forecast_archive"],
                    qpe_root=fixture_root["qpe_root"],
                    external_archive=fixture_root["external_archive"],
                    out_root=fixture_root["out_root"], qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
                    thresholds=(0.1,), fss_scales=(1,), bootstrap_cfg=None,
                    adequacy_min_events=1, adequacy_threshold_mm_h=2.5)
    # Adequacy is counted on the SAME (resampled-to-forecast-grid) target
    # field every metric above is scored against, per tools/benchmark.py's
    # convention — obs_sel = [0, 2, 3, 2.75], domain max 3.0 > 2.5 -> 1 event.
    assert record["adequacy"]["n_events"] == 1
    assert record["adequacy"]["adequate"] is True


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def test_render_html_has_expected_rows(fixture_root):
    record = sb.run(DAY, forecast_archive=fixture_root["forecast_archive"],
                    qpe_root=fixture_root["qpe_root"],
                    external_archive=fixture_root["external_archive"],
                    out_root=fixture_root["out_root"], qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
                    thresholds=(0.1, 1.0), fss_scales=(1,), bootstrap_cfg=None,
                    adequacy_min_events=5)
    sb.write_record(record, fixture_root["out_root"])
    trend = sb.load_trend(fixture_root["out_root"], DAY, n_days=30)
    page = sb.render_html(record, trend)

    assert "<!doctype html>" in page.lower()
    assert "Pluvio scoreboard" in page
    assert "forecast" in page
    assert "lead 30 min" in page
    assert "NOT adequate" in page  # this fixture's single issue is inadequate
    assert "buienradar" in page
    assert "ours (same stations)" in page
    # 30-day trend must include today's own record (just written).
    assert DAY.isoformat() in page


def test_render_html_trend_table_reflects_archive(fixture_root):
    record = sb.run(DAY, forecast_archive=fixture_root["forecast_archive"],
                    qpe_root=fixture_root["qpe_root"],
                    external_archive=fixture_root["external_archive"],
                    out_root=fixture_root["out_root"], qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
                    thresholds=(0.1,), fss_scales=(1,), bootstrap_cfg=None)
    sb.write_record(record, fixture_root["out_root"])

    prior_day = DAY - dt.timedelta(days=1)
    prior_record = dict(record)
    prior_record["day"] = prior_day.isoformat()
    sb.write_record(prior_record, fixture_root["out_root"])

    trend = sb.load_trend(fixture_root["out_root"], DAY, n_days=30)
    assert len(trend) == 2
    assert [r["day"] for r in trend] == [prior_day.isoformat(), DAY.isoformat()]


# ---------------------------------------------------------------------------
# forecast archive iteration
# ---------------------------------------------------------------------------

def test_iter_forecast_issues_finds_only_matching_kind(fixture_root):
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast", ISSUE_EPOCH,
                        [LEAD_MIN], PRED_2X2[None, :, :])
    forecasts = list(sb.iter_forecast_issues(fixture_root["forecast_archive"], DAY, "forecast"))
    nowcasts = list(sb.iter_forecast_issues(fixture_root["forecast_archive"], DAY, "nowcast"))
    assert len(forecasts) == 1
    assert len(nowcasts) == 1
    assert forecasts[0][0] == ISSUE_EPOCH


def test_iter_forecast_issues_empty_dir(tmp_path):
    assert list(sb.iter_forecast_issues(tmp_path / "nope", DAY, "forecast")) == []
