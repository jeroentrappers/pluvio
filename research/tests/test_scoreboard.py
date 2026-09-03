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
                   fill_slot, bounds=QPE_BOUNDS) -> pathlib.Path:
    """A synthetic day-zarr. `bounds` is written as the store's `bounds` attr
    (mandatory in production, so a fixture without one is not a realistic
    store); pass None only to build the attr-less store that must be refused."""
    import zarr

    path = root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    rate = np.full((288, *truth_full.shape), np.nan, dtype="float16")
    for slot in ([fill_slot] if isinstance(fill_slot, int) else fill_slot):
        rate[slot] = truth_full.astype("float16")
    g.create_array("rate", data=rate, chunks=(1, *truth_full.shape))
    if bounds is not None:
        g.attrs["bounds"] = [float(x) for x in bounds]
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
    slot = round((VALID_EPOCH % 86400) / 300)
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
    buien_bias = sum(p - t for p, t in zip(buien_preds, truths, strict=True)) / 2
    assert buien["bias"] == pytest.approx(buien_bias)

    # "ours" prediction sampled from the same 2x2 forecast grid at the same
    # points: station A -> PRED_2X2[1,1]=5.0, station B -> PRED_2X2[1,0]=4.0.
    ours_preds = [5.0, 4.0]
    ours_bias = sum(p - t for p, t in zip(ours_preds, truths, strict=True)) / 2
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
                    qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
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
                    qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
                    thresholds=(0.1,), fss_scales=(1,), bootstrap_cfg=None,
                    adequacy_min_events=5)
    assert record["adequacy"]["n_events"] == 0
    assert record["adequacy"]["adequate"] is False


def test_adequate_day_when_events_meet_minimum(fixture_root):
    record = sb.run(DAY, forecast_archive=fixture_root["forecast_archive"],
                    qpe_root=fixture_root["qpe_root"],
                    external_archive=fixture_root["external_archive"],
                    qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
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
                    qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
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
                    qpe_bounds=QPE_BOUNDS, kinds=("forecast",),
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


# ---------------------------------------------------------------------------
# truth geometry: the QPE day-zarr is on the research analysis grid, NOT the
# 100x100 Belgium serving box. Regression for the reviewed defect where
# DEFAULT_QPE_BOUNDS was produce_forecast.BE_BOUNDS: Brussels truth was then
# read from cell (351, 365) instead of (602, 302) — 237 km away — and the
# whole 768^2 composite was squashed onto the 100^2 serving box.
# ---------------------------------------------------------------------------

QPE_N = 768                                # PLUVIO_GRID_N the archiver runs at
BRUSSELS = (50.85, 4.35)                   # lat, lon
BE_BOUNDS = (1.5, 48.9, 7.5, 52.5)         # produce_forecast.BE_BOUNDS, W S E N
BE_N = 100
# Measured on hetz1: the archive is written by the /opt/pluvio/radarproc
# checkout, whose model/geo.py predates both the 700/765 trim and the
# registration bias, so the day-stores are binned onto THIS box — not the
# (0.07, 49.4387, 10.9265, 55.9736) a derivation from this checkout returns.
# 60 km apart at the south edge, hence the mandatory attr.
PROD_QPE_BOUNDS = (0.0, 48.895301818847656, 10.856452941894531, 55.973602294921875)
DERIVED_FROM_THIS_CHECKOUT = (0.07, 49.4386863708, 10.9264535904, 55.9736022949)


def _sparse_qpe_day(root: pathlib.Path, day: dt.date, shape, fill_slot: int,
                    frame: np.ndarray, bounds=PROD_QPE_BOUNDS) -> pathlib.Path:
    """A day-zarr of the real (288, N, N) shape with only one slot chunk
    materialised, so a 768-grid fixture costs ~1 MB rather than 340 MB."""
    import zarr

    path = root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    arr = g.create_array("rate", shape=(288, *shape), dtype="float16",
                         chunks=(1, *shape), fill_value=np.nan)
    arr[fill_slot] = frame.astype("float16")
    if bounds is not None:
        g.attrs["bounds"] = [float(x) for x in bounds]
    return path


@pytest.fixture()
def real_grid_root(tmp_path):
    """768-grid composite on the PRODUCTION bounds: 0 mm/h everywhere
    (measured dry) with a 9x9 patch of 8 mm/h centred on the cell that holds
    Brussels."""
    w, s, e, n = PROD_QPE_BOUNDS
    lat, lon = BRUSSELS
    row = int((n - lat) / (n - s) * QPE_N)
    col = int((lon - w) / (e - w) * QPE_N)
    # The georeference this fixture asserts against, measured on hetz1.
    assert (row, col) == (555, 307)

    frame = np.zeros((QPE_N, QPE_N), dtype="float32")
    frame[row - 4:row + 5, col - 4:col + 5] = 8.0
    # No radar coverage south of ~49.5 deg: NaN, not dry. That region is
    # inside the serving box, so the target grid's validity mask has to catch
    # it — this is the half of the fix `nan_to_num` used to defeat.
    frame[int((n - 49.5) / (n - s) * QPE_N):, :] = np.nan

    qpe_root = tmp_path / "qpe"
    slot = round((VALID_EPOCH % 86400) / 300)
    _sparse_qpe_day(qpe_root, DAY, (QPE_N, QPE_N), slot, frame)
    return {"qpe_root": qpe_root, "bounds": PROD_QPE_BOUNDS, "cell": (row, col),
            "tmp_path": tmp_path}


def test_truth_bounds_read_from_the_store_attr(real_grid_root):
    truth = sb.QpeTruth(real_grid_root["qpe_root"])
    got = truth.frame(VALID_EPOCH)
    assert got is not None
    rate, bounds = got
    assert rate.shape == (QPE_N, QPE_N)
    assert bounds == pytest.approx(PROD_QPE_BOUNDS)
    # neither the serving box (the original defect) nor a derivation from this
    # checkout's model.geo, which is 60 km out at the south edge
    assert bounds != pytest.approx(BE_BOUNDS)
    assert bounds != pytest.approx(DERIVED_FROM_THIS_CHECKOUT)


def test_truth_point_hits_the_station_cell(real_grid_root):
    truth = sb.QpeTruth(real_grid_root["qpe_root"])
    lat, lon = BRUSSELS
    assert truth.point(lat, lon, VALID_EPOCH) == pytest.approx(8.0)
    # a derivation from this checkout would read row 602 instead of 555 —
    # 47 rows, ~48 km north of the patch, where this fixture is dry
    _w, s, _e, n = DERIVED_FROM_THIS_CHECKOUT
    wrong_row = int((n - lat) / (n - s) * QPE_N)
    assert wrong_row == 602
    assert abs(wrong_row - real_grid_root["cell"][0]) > 8


def test_store_without_bounds_attr_is_refused(tmp_path):
    """A day-store that does not state its own georeference must stop the run,
    not fall back to a derivation: the archiver's model/geo.py is not this
    checkout's, so any derivation here is 60 km out at the south edge."""
    frame = np.zeros((16, 16), dtype="float32")
    slot = round((VALID_EPOCH % 86400) / 300)
    path = _sparse_qpe_day(tmp_path / "qpe", DAY, (16, 16), slot, frame, bounds=None)
    truth = sb.QpeTruth(tmp_path / "qpe")
    with pytest.raises(sb.QpeGeometryError) as exc:
        truth.frame(VALID_EPOCH)
    assert str(path) in str(exc.value)
    assert "bounds" in str(exc.value)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [5.0, 1.0, 1.0, 2.0], "nope"])
def test_unusable_bounds_attr_is_refused(tmp_path, bad):
    frame = np.zeros((16, 16), dtype="float32")
    slot = round((VALID_EPOCH % 86400) / 300)
    _sparse_qpe_day(tmp_path / "qpe", DAY, (16, 16), slot, frame, bounds=None)
    import zarr
    g = zarr.open_group(str(tmp_path / "qpe" / f"{DAY:%Y}" / f"{DAY:%m}" / f"{DAY:%d}.zarr"),
                        mode="a")
    g.attrs["bounds"] = bad
    with pytest.raises(sb.QpeGeometryError):
        sb.QpeTruth(tmp_path / "qpe").frame(VALID_EPOCH)


def test_explicit_override_beats_the_store_attr(tmp_path):
    attr_bounds = (0.0, 45.0, 10.0, 55.0)
    frame = np.zeros((16, 16), dtype="float32")
    frame[3, 4] = 7.0
    slot = round((VALID_EPOCH % 86400) / 300)
    _sparse_qpe_day(tmp_path / "qpe", DAY, (16, 16), slot, frame, bounds=attr_bounds)

    truth = sb.QpeTruth(tmp_path / "qpe")
    _rate, bounds = truth.frame(VALID_EPOCH)
    assert bounds == pytest.approx(attr_bounds)
    # cell (3, 4) of a 16x16 grid on those bounds
    lat = 55.0 - (3 + 0.5) * (55.0 - 45.0) / 16
    lon = 0.0 + (4 + 0.5) * 10.0 / 16
    assert truth.point(lat, lon, VALID_EPOCH) == pytest.approx(7.0)

    override = (0.0, 40.0, 20.0, 60.0)
    forced = sb.QpeTruth(tmp_path / "qpe", bounds=override)
    _rate, got = forced.frame(VALID_EPOCH)
    assert got == pytest.approx(override)


def test_grid_path_scores_on_the_serving_box_without_squashing(real_grid_root):
    """The serving box is a different, much smaller grid than the truth grid:
    the composite must be area-averaged in place over that window, not
    stretched to fit it, and the composite's own uncovered region must arrive
    as NaN rather than as measured-dry."""
    truth = sb.QpeTruth(real_grid_root["qpe_root"])
    obs = truth.field_on(VALID_EPOCH, (BE_N, BE_N), BE_BOUNDS)
    assert obs is not None

    bw, bs, be_, bn = BE_BOUNDS
    lat, lon = BRUSSELS
    frow = int((bn - lat) / (bn - bs) * BE_N)
    fcol = int((lon - bw) / (be_ - bw) * BE_N)
    assert (frow, fcol) == (45, 47)
    # the 8 mm/h patch is wide enough to fill this target cell's whole
    # footprint, so an honest area mean is exactly 8.0
    assert obs[frow, fcol] == pytest.approx(8.0)
    # away from the patch the composite measured dry, so 0.0 — not NaN
    assert obs[10, 10] == pytest.approx(0.0)
    # rows wholly inside the composite's uncovered region stay unobserved
    # (lat 49.5 is target row ~83.3, so row 85 down is entirely below it)
    assert np.isnan(obs[85:, :]).all()
    assert np.isfinite(obs[80, :]).all()
    # ... and the validity mask is therefore not vacuous
    assert np.isfinite(obs).sum() < BE_N * BE_N

    fc_root = real_grid_root["tmp_path"] / "forecast_archive"
    pred = np.zeros((1, BE_N, BE_N), dtype="float32")
    pred[0, frow, fcol] = 8.0
    _write_forecast_npz(fc_root, "forecast", ISSUE_EPOCH, [LEAD_MIN], pred,
                        bounds=BE_BOUNDS)
    out = sb.score_grid_day(DAY, fc_root, truth, kinds=("forecast",),
                            thresholds=(1.0,), fss_scales=(1,), bootstrap_cfg=None)
    row = out["results"]["forecast"][str(LEAD_MIN)]["1.0"]
    assert row["hits"] == 1
    assert row["n_valid_cells"] == int(np.isfinite(obs).sum())
    assert row["n_valid_cells"] < BE_N * BE_N


# ---------------------------------------------------------------------------
# NaN propagation: an uncovered composite cell must not score as observed-dry
# ---------------------------------------------------------------------------

def test_nan_region_stays_invalid_and_is_not_scored(tmp_path):
    truth_full = TRUTH_FULL.copy()
    truth_full[0:2, 2:4] = np.nan          # NE quadrant unobserved
    qpe_root = tmp_path / "qpe"
    slot = round((VALID_EPOCH % 86400) / 300)
    _write_qpe_day(qpe_root, DAY, truth_full, fill_slot=slot)
    fc_root = tmp_path / "forecast_archive"
    _write_forecast_npz(fc_root, "forecast", ISSUE_EPOCH, [LEAD_MIN], PRED_2X2[None, :, :])

    truth = sb.QpeTruth(qpe_root, bounds=QPE_BOUNDS)
    obs = truth.field_on(VALID_EPOCH, (2, 2), FC_BOUNDS)
    assert np.isnan(obs[0, 1])             # NOT 0.0 via nan_to_num
    assert np.isfinite(obs[0, 0]) and np.isfinite(obs[1, 0]) and np.isfinite(obs[1, 1])

    out = sb.score_grid_day(DAY, fc_root, truth, kinds=("forecast",),
                            thresholds=(1.0,), fss_scales=(1,), bootstrap_cfg=None)
    row = out["results"]["forecast"][str(LEAD_MIN)]["1.0"]
    assert row["n_valid_cells"] == 3        # the NaN cell is excluded, not dry
    # pred [0, 0.5, 4, 5] vs obs [0, NaN, 3, 2.75]: the dropped cell was a
    # miss (pred 0.5 < 1 <= obs 2.0) — scoring it as dry would have hidden it
    assert row["hits"] == 2
    assert row["misses"] == 0
    assert row["false_alarms"] == 0


def test_partial_block_coverage_threshold(tmp_path):
    truth_full = TRUTH_FULL.copy()
    truth_full[0, 2] = np.nan              # 1 of 4 cells in the NE block
    truth_full[1, 2] = np.nan              # 2 of 4 -> coverage exactly 0.5
    qpe_root = tmp_path / "qpe"
    slot = round((VALID_EPOCH % 86400) / 300)
    _write_qpe_day(qpe_root, DAY, truth_full, fill_slot=slot)

    keep = sb.QpeTruth(qpe_root, bounds=QPE_BOUNDS, min_block_coverage=0.5)
    assert keep.field_on(VALID_EPOCH, (2, 2), FC_BOUNDS)[0, 1] == pytest.approx(2.0)
    strict = sb.QpeTruth(qpe_root, bounds=QPE_BOUNDS, min_block_coverage=0.75)
    assert np.isnan(strict.field_on(VALID_EPOCH, (2, 2), FC_BOUNDS)[0, 1])


# ---------------------------------------------------------------------------
# slot rounding wraps into the next day instead of falling off the end
# ---------------------------------------------------------------------------

def test_slot_rounding_wraps_to_next_day(tmp_path):
    """A valid time in the last 150 s of a day belongs to the NEXT day's slot
    0. Rounding within the day gave slot 288, off the end of the array, and
    those samples were silently dropped."""
    next_day = DAY + dt.timedelta(days=1)
    qpe_root = tmp_path / "qpe"
    # distinguishable values: only the next day's store has 6.0
    _write_qpe_day(qpe_root, DAY, np.full((4, 4), 1.0, dtype="float32"),
                   fill_slot=[0, 287])
    _write_qpe_day(qpe_root, next_day, np.full((4, 4), 6.0, dtype="float32"), fill_slot=0)

    truth = sb.QpeTruth(qpe_root, bounds=QPE_BOUNDS)
    late = int(dt.datetime(2026, 9, 2, 23, 59, tzinfo=dt.UTC).timestamp())
    assert truth._slot_of(late) == (next_day, 0)
    # the value proves which store/slot was actually read
    assert truth.point(2.0, 2.0, late) == pytest.approx(6.0)
    # ... and a time that does not need to wrap still resolves within the day
    early = int(dt.datetime(2026, 9, 2, 23, 55, tzinfo=dt.UTC).timestamp())
    assert truth._slot_of(early) == (DAY, 287)
    assert truth.point(2.0, 2.0, early) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap with two kinds on different issue sequences (separate producers)
# ---------------------------------------------------------------------------

def test_bootstrap_handles_kinds_with_different_issue_sets(fixture_root):
    # nowcast runs on its own cadence: an extra issue at 00:30 the forecast
    # producer never made. A single paired block_bootstrap call over both
    # kinds raises ValueError on the mismatch, so each kind gets its own draw.
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast", ISSUE_EPOCH,
                        [LEAD_MIN], PRED_2X2[None, :, :])
    later = ISSUE_EPOCH + 1800
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast", later,
                        [LEAD_MIN], PRED_2X2[None, :, :])
    # truth for both 00:30 and 01:00 so the extra nowcast issue really scores
    _write_qpe_day(fixture_root["qpe_root"], DAY, TRUTH_FULL, fill_slot=[6, 12])

    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    boot_cfg = {"n": 20, "ci": 0.9, "blocks_h": 6.0, "seed": 1,
                "reference_model": "forecast"}
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast", "nowcast"), thresholds=(0.1,),
                            fss_scales=(1,), bootstrap_cfg=boot_cfg)
    rows = {k: out["results"][k][str(LEAD_MIN)]["0.1"] for k in ("forecast", "nowcast")}
    # the sequences really do differ — the paired-draw precondition is violated
    assert rows["forecast"]["n_samples"] == 1
    assert rows["nowcast"]["n_samples"] == 2
    for row in rows.values():
        assert row["ci"] is not None
        assert "ci_lo" in row["ci"]["csi"]


def test_bootstrap_pairs_kinds_that_share_an_issue_sequence(fixture_root):
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast", ISSUE_EPOCH,
                        [LEAD_MIN], (PRED_2X2 * 0.5)[None, :, :])
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    boot_cfg = {"n": 20, "ci": 0.9, "blocks_h": 6.0, "seed": 1,
                "reference_model": "forecast"}
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast", "nowcast"), thresholds=(0.1,),
                            fss_scales=(1,), bootstrap_cfg=boot_cfg)
    assert out["results"]["nowcast"][str(LEAD_MIN)]["0.1"]["ci"] is not None
    assert out["results"]["forecast"][str(LEAD_MIN)]["0.1"]["ci"] is not None


# ---------------------------------------------------------------------------
# point join: the right (issue, lead) pair, and one shared truth sample
# ---------------------------------------------------------------------------

def test_point_join_picks_the_matching_issue_not_the_nearest_valid_time(fixture_root):
    """Two runs can both be valid at 00:30: the 00:00 run's lead 30 and the
    00:30 run's lead 0. A row tagged lead_min=30 must be compared against the
    00:00 run's lead-30 field, or a 30-min forecast is silently scored as an
    analysis."""
    fc_root = fixture_root["forecast_archive"]
    later = ISSUE_EPOCH + 1800
    # 00:00 run: leads 0 and 30, distinguishable fields
    lead0 = np.full((2, 2), 1.0, dtype="float32")
    lead30 = np.full((2, 2), 2.0, dtype="float32")
    _write_forecast_npz(fc_root, "forecast", ISSUE_EPOCH, [0, LEAD_MIN],
                        np.stack([lead0, lead30]))
    # 00:30 run: lead 0 valid at exactly the same time, different value
    _write_forecast_npz(fc_root, "forecast", later, [0, LEAD_MIN],
                        np.stack([np.full((2, 2), 9.0, dtype="float32"),
                                  np.full((2, 2), 9.0, dtype="float32")]))

    index = [(issue, sb.load_forecast_npz(p))
             for issue, p in sb.iter_forecast_issues(fc_root, DAY, "forecast")]
    assert len(index) == 2
    got = sb._nearest_forecast_point(index, 2.0, 2.0, LEAD_MIN, VALID_EPOCH)
    assert got == pytest.approx(2.0)       # 00:00 run's lead 30, not 9.0
    # and the lead-0 row valid at 00:30 picks the 00:30 run
    assert sb._nearest_forecast_point(index, 2.0, 2.0, 0, VALID_EPOCH) == pytest.approx(9.0)


def test_point_join_loads_previous_day_issues(tmp_path):
    """Buienradar's archive is keyed by VALID day: a row valid 00:05 with a
    30-min lead was issued at 23:35 the previous day, so its own run lives in
    the previous day's forecast directory."""
    day = dt.date(2026, 9, 3)
    valid = int(dt.datetime(2026, 9, 3, 0, 5, tzinfo=dt.UTC).timestamp())
    issue = valid - 30 * 60                              # 2026-09-02 23:35
    fc_root = tmp_path / "forecast_archive"
    _write_forecast_npz(fc_root, "forecast", issue, [LEAD_MIN],
                        np.full((1, 2, 2), 3.0, dtype="float32"))
    qpe_root = tmp_path / "qpe"
    slot = round((valid % 86400) / 300)
    _write_qpe_day(qpe_root, day, TRUTH_FULL, fill_slot=slot)
    ext_root = tmp_path / "external_baselines"
    _write_buienradar_jsonl(ext_root, day, [
        {"source": "buienradar", "station": "A", "lat": 2.0, "lon": 2.0,
         "issue_epoch": issue, "fetch_epoch": issue, "valid_epoch": valid,
         "lead_min": LEAD_MIN, "mm_per_h": 1.0}])

    truth = sb.QpeTruth(qpe_root, bounds=QPE_BOUNDS)
    out = sb.score_points_day(day, ext_root, fc_root, truth, kind="forecast")
    assert out["n_matched"] == 1
    assert out["ours"][str(LEAD_MIN)]["n"] == 1


def test_both_point_series_share_one_truth_sample(fixture_root, monkeypatch):
    """Guard on the shared-truth invariant: if either series re-looked-up the
    truth instead of reading the sample taken while `matched` was built, this
    counter would hand it a different value and the hand-computed biases
    below would not hold."""
    calls = {"n": 0}

    def _counting_point(self, lat, lon, valid_epoch):
        calls["n"] += 1
        return float(calls["n"])

    monkeypatch.setattr(sb.QpeTruth, "point", _counting_point)
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    out = sb.score_points_day(DAY, fixture_root["external_archive"],
                              fixture_root["forecast_archive"], truth,
                              kind="forecast", thresholds=(0.1,))
    assert calls["n"] == 2                 # exactly one lookup per matched row
    truths = [1.0, 2.0]                    # station A, then station B
    buien = out["buienradar"][str(LEAD_MIN)]
    ours = out["ours"][str(LEAD_MIN)]
    assert buien["bias"] == pytest.approx(sum(p - t for p, t in zip([4.5, 2.0], truths, strict=True)) / 2)
    assert ours["bias"] == pytest.approx(sum(p - t for p, t in zip([5.0, 4.0], truths, strict=True)) / 2)


# ---------------------------------------------------------------------------
# archive layout
# ---------------------------------------------------------------------------

def test_archive_path_has_no_duplicate_scoreboard_level(tmp_path):
    out_root = tmp_path / "scoreboard"
    assert sb.archive_path(out_root, DAY) == out_root / "2026" / "09" / "02.json"


# ---------------------------------------------------------------------------
# _regrid_block_mean: the two boxes need not be nested
# ---------------------------------------------------------------------------

def test_regrid_handles_a_target_box_hanging_off_the_source():
    """The truth and serving boxes happen to be nested today, but nothing
    guarantees it (they were not under the reviewed defect's bounds, and a
    wider serving box is a config change away). A target cell with no source
    footprint must be NaN, never an edge value stretched to fill it."""
    src = np.ones((4, 4), dtype="float32")
    # target box shifted 4 degrees east: only its western half overlaps
    out = sb._regrid_block_mean(src, (0.0, 0.0, 4.0, 4.0), (2.0, 0.0, 6.0, 4.0), (2, 2))
    assert out[:, 0] == pytest.approx([1.0, 1.0])
    assert np.isnan(out[:, 1]).all()
    # entirely disjoint -> all NaN
    away = sb._regrid_block_mean(src, (0.0, 0.0, 4.0, 4.0), (10.0, 0.0, 14.0, 4.0), (2, 2))
    assert np.isnan(away).all()


def test_regrid_accumulates_in_float64():
    """The integral image is a running total over the whole 768^2 field. In
    float32 it reaches ~1e6 where the representable spacing is ~0.06 mm/h, so
    a block mean taken near the far corner drifts into the third decimal the
    report prints. Measured below: ~6e-3 mm/h.
    """
    rng = np.random.default_rng(0)
    src = (rng.random((768, 768)).astype("float32") * 3.0)
    src[300:500, 300:500] = 60.0                     # a heavy core to load the sum
    src = src.astype("float16").astype("float32")     # as the store holds it
    box = (0.0, 0.0, 8.0, 8.0)

    # 192x192 target over the same box -> each target cell is exactly 4x4
    # source cells, so the expected value is a plain mean of a known window.
    out = sb._regrid_block_mean(src, box, box, (192, 192))
    expected = float(src[760:764, 760:764].astype("float64").mean())
    assert out[190, 190] == pytest.approx(expected, abs=1e-9)

    # the same accumulation in float32, to show the guard is not vacuous
    f32 = np.zeros((769, 769), "float32")
    f32[1:, 1:] = src.cumsum(0).cumsum(1)
    naive = float(f32[764, 764] - f32[760, 764] - f32[764, 760] + f32[760, 760]) / 16
    assert abs(naive - expected) > 1e-3


# ---------------------------------------------------------------------------
# paired difference vs the reference model
# ---------------------------------------------------------------------------

def test_paired_difference_surfaced_for_kinds_drawn_together(fixture_root):
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast", ISSUE_EPOCH,
                        [LEAD_MIN], (PRED_2X2 * 0.5)[None, :, :])
    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    boot_cfg = {"n": 20, "ci": 0.9, "blocks_h": 6.0, "seed": 1,
                "reference_model": "forecast"}
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast", "nowcast"), thresholds=(0.1,),
                            fss_scales=(1,), bootstrap_cfg=boot_cfg)
    now_row = out["results"]["nowcast"][str(LEAD_MIN)]["0.1"]
    ref_row = out["results"]["forecast"][str(LEAD_MIN)]["0.1"]
    assert now_row["ci_vs_reference"] is not None
    assert now_row["ci_vs_reference"]["reference_model"] == "forecast"
    assert "ci_lo" in now_row["ci_vs_reference"]["csi"]
    # the reference model has no difference against itself
    assert ref_row["ci_vs_reference"] is None


def test_no_paired_difference_across_issue_sequence_groups(fixture_root):
    """An unpaired difference interval would not mean what the column says, so
    a kind bootstrapped in its own group reports None rather than a number."""
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast", ISSUE_EPOCH,
                        [LEAD_MIN], PRED_2X2[None, :, :])
    _write_forecast_npz(fixture_root["forecast_archive"], "nowcast",
                        ISSUE_EPOCH + 1800, [LEAD_MIN], PRED_2X2[None, :, :])
    _write_qpe_day(fixture_root["qpe_root"], DAY, TRUTH_FULL, fill_slot=[6, 12])

    truth = sb.QpeTruth(fixture_root["qpe_root"], bounds=QPE_BOUNDS)
    boot_cfg = {"n": 20, "ci": 0.9, "blocks_h": 6.0, "seed": 1,
                "reference_model": "forecast"}
    out = sb.score_grid_day(DAY, fixture_root["forecast_archive"], truth,
                            kinds=("forecast", "nowcast"), thresholds=(0.1,),
                            fss_scales=(1,), bootstrap_cfg=boot_cfg)
    for kind in ("forecast", "nowcast"):
        row = out["results"][kind][str(LEAD_MIN)]["0.1"]
        assert row["ci"] is not None
        assert row["ci_vs_reference"] is None
