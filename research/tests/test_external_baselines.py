"""Tests for tools.external_baselines. No network access: HTTP is mocked and
all payloads come from tests/fixtures/."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.error

import pytest

from tools import external_baselines as eb

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# value_to_mm_per_h
# ---------------------------------------------------------------------------

def test_value_to_mm_per_h_reference_points():
    # 109 is the feed's defined "1.0 mm/h" anchor.
    assert eb.value_to_mm_per_h(109) == pytest.approx(1.0)
    # 0 is not exactly zero rain on this log scale -- just very small.
    assert eb.value_to_mm_per_h(0) == pytest.approx(0.0003924, rel=1e-3)
    # 255 is the byte's formal ceiling; the log scale sends it to an
    # enormous, never-observed rate -- that's the encoding, not a bug.
    assert eb.value_to_mm_per_h(255) == pytest.approx(36517.4, rel=1e-3)


def test_value_to_mm_per_h_monotonic():
    values = [0, 50, 109, 150, 200, 255]
    rates = [eb.value_to_mm_per_h(v) for v in values]
    assert rates == sorted(rates)


# ---------------------------------------------------------------------------
# parse_raintext: real payload (live-recorded, all-zero-rain)
# ---------------------------------------------------------------------------

def test_parse_raintext_live_fixture_no_rain():
    text = read_fixture("buienradar_brussels_live_20260903.txt")
    issue = dt.datetime(2026, 9, 3, 8, 27, 44, tzinfo=dt.timezone.utc)
    rows = eb.parse_raintext(text, issue)

    non_blank_lines = [l for l in text.splitlines() if l.strip()]
    assert len(rows) == len(non_blank_lines) == 24

    # first line is 10:30 local (CEST, UTC+2) on issue day.
    first_epoch, first_rate = rows[0]
    first_dt = dt.datetime.fromtimestamp(first_epoch, dt.timezone.utc)
    assert first_dt == dt.datetime(2026, 9, 3, 8, 30, tzinfo=dt.timezone.utc)
    assert first_rate == pytest.approx(eb.value_to_mm_per_h(0))

    # steps are 5 minutes apart, strictly increasing.
    epochs = [e for e, _ in rows]
    assert epochs == sorted(epochs)
    assert all(b - a == 300 for a, b in zip(epochs, epochs[1:]))


# ---------------------------------------------------------------------------
# parse_raintext: local-midnight rollover
# ---------------------------------------------------------------------------

def test_parse_raintext_day_rollover():
    text = read_fixture("buienradar_rollover_20260630.txt")
    # issued a few minutes before local midnight, CEST (UTC+2).
    issue = dt.datetime(2026, 6, 30, 21, 52, 0, tzinfo=dt.timezone.utc)
    rows = eb.parse_raintext(text, issue)
    assert len(rows) == 7

    by_local = {}
    for epoch, rate in rows:
        local = dt.datetime.fromtimestamp(epoch, eb.AMSTERDAM)
        by_local[local.strftime("%Y-%m-%d %H:%M")] = rate

    # pre-midnight lines stay on 30 June local.
    assert "2026-06-30 23:40" in by_local
    assert "2026-06-30 23:55" in by_local
    # post-midnight lines roll onto 1 July local.
    assert "2026-07-01 00:00" in by_local
    assert "2026-07-01 00:10" in by_local

    # values round-trip through the same formula.
    assert by_local["2026-06-30 23:55"] == pytest.approx(eb.value_to_mm_per_h(109))
    assert by_local["2026-07-01 00:00"] == pytest.approx(eb.value_to_mm_per_h(141))

    # epochs strictly increase across the rollover.
    epochs = [e for e, _ in rows]
    assert epochs == sorted(epochs)
    assert len(set(epochs)) == len(epochs)


# ---------------------------------------------------------------------------
# parse_raintext: malformed-input tolerance
# ---------------------------------------------------------------------------

def test_parse_raintext_malformed_lines_are_skipped():
    text = read_fixture("buienradar_malformed.txt")
    issue = dt.datetime(2026, 9, 3, 10, 25, 0, tzinfo=dt.timezone.utc)
    rows = eb.parse_raintext(text, issue)
    # only "000|10:30", "109|10:40", "999|10:45", "109|10:50" are valid;
    # blank line, no-pipe line, non-numeric value, and bad time are skipped.
    assert len(rows) == 4
    minutes = sorted(dt.datetime.fromtimestamp(e, eb.AMSTERDAM).minute for e, _ in rows)
    assert minutes == [30, 40, 45, 50]


def test_parse_raintext_empty_string():
    issue = dt.datetime(2026, 9, 3, 10, 25, 0, tzinfo=dt.timezone.utc)
    assert eb.parse_raintext("", issue) == []


# ---------------------------------------------------------------------------
# BuienradarSource / fetch: HTTP mocked, never touches the network
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text: str):
        self._text = text

    def read(self):
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_buienradar_source_fetch_point_success(monkeypatch):
    text = read_fixture("buienradar_brussels_live_20260903.txt")

    def fake_urlopen(url, timeout=None):
        assert "lat=50.85" in url and "lon=4.35" in url
        return _FakeResponse(text)

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    source = eb.BuienradarSource()
    issue = dt.datetime(2026, 9, 3, 8, 27, 44, tzinfo=dt.timezone.utc)
    rows = source.fetch_point(50.85, 4.35, issue)
    assert rows is not None
    assert len(rows) == 24


def test_buienradar_source_fetch_point_http_error_returns_none(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    source = eb.BuienradarSource()
    issue = dt.datetime.now(dt.timezone.utc)
    assert source.fetch_point(50.85, 4.35, issue) is None


# ---------------------------------------------------------------------------
# sample_all with an injected fake Source (no HTTP at all)
# ---------------------------------------------------------------------------

class _FakeSource:
    name = "buienradar"

    def __init__(self, points_by_station):
        self._points_by_station = points_by_station
        self.calls = []

    def fetch_point(self, lat, lon, issue_time_utc):
        self.calls.append((lat, lon))
        return self._points_by_station.get((lat, lon))


def test_sample_all_builds_rows_and_lead_min():
    issue = dt.datetime(2026, 9, 3, 10, 0, 0, tzinfo=dt.timezone.utc)
    stations = [("A", 1.0, 2.0), ("B", 3.0, 4.0)]
    points = {
        (1.0, 2.0): [(int(issue.timestamp()) + 300, 0.5), (int(issue.timestamp()) + 600, 1.5)],
        (3.0, 4.0): None,  # simulates a failed station
    }
    source = _FakeSource(points)
    rows = eb.sample_all(issue_time=issue, stations=stations, source=source, delay_s=0)

    assert len(rows) == 2
    assert {r["station"] for r in rows} == {"A"}
    row0 = rows[0]
    assert row0["source"] == "buienradar"
    assert row0["lat"] == 1.0 and row0["lon"] == 2.0
    assert row0["issue_epoch"] == int(issue.timestamp())
    assert row0["lead_min"] == 5
    assert rows[1]["lead_min"] == 10


# ---------------------------------------------------------------------------
# Archive: append + idempotency + load
# ---------------------------------------------------------------------------

def _mk_row(station, issue_epoch, valid_epoch, mm_per_h=1.0):
    return {
        "source": "buienradar",
        "station": station,
        "lat": 50.0,
        "lon": 4.0,
        "issue_epoch": issue_epoch,
        "valid_epoch": valid_epoch,
        "lead_min": round((valid_epoch - issue_epoch) / 60),
        "mm_per_h": mm_per_h,
    }


def test_append_archive_writes_and_is_idempotent(tmp_path):
    day = dt.date(2026, 9, 3)
    issue_epoch = int(dt.datetime(2026, 9, 3, 10, 0, tzinfo=dt.timezone.utc).timestamp())
    valid_epoch = issue_epoch + 600
    rows = [_mk_row("Brussels", issue_epoch, valid_epoch)]

    written1 = eb.append_archive(rows, tmp_path)
    assert written1 == 1

    # exact same row again: no-op.
    written2 = eb.append_archive(rows, tmp_path)
    assert written2 == 0

    loaded = eb.load_archive(tmp_path, day)
    assert len(loaded) == 1
    assert loaded[0]["station"] == "Brussels"

    path = tmp_path / "buienradar" / "2026" / "09" / "03.jsonl"
    assert path.exists()
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1

    # a genuinely new row (different valid_epoch) does get appended.
    rows2 = [_mk_row("Brussels", issue_epoch, valid_epoch + 300)]
    written3 = eb.append_archive(rows2, tmp_path)
    assert written3 == 1
    assert len(eb.load_archive(tmp_path, day)) == 2


def test_load_archive_missing_day_returns_empty(tmp_path):
    assert eb.load_archive(tmp_path, dt.date(2020, 1, 1)) == []


def test_append_archive_splits_by_utc_day_of_valid_epoch(tmp_path):
    # issue on day 1, valid_epoch on day 2 (near-midnight step): must land
    # in day 2's file since that's what a later truth-join keys on.
    issue_epoch = int(dt.datetime(2026, 9, 3, 23, 55, tzinfo=dt.timezone.utc).timestamp())
    valid_epoch = issue_epoch + 600  # crosses into 2026-09-04 UTC
    rows = [_mk_row("Brussels", issue_epoch, valid_epoch)]
    eb.append_archive(rows, tmp_path)

    assert eb.load_archive(tmp_path, dt.date(2026, 9, 3)) == []
    assert len(eb.load_archive(tmp_path, dt.date(2026, 9, 4))) == 1


# ---------------------------------------------------------------------------
# score_against_truth
# ---------------------------------------------------------------------------

def test_score_against_truth_arithmetic():
    rows = [
        _mk_row("A", 0, 300, mm_per_h=1.0),   # lead 5
        _mk_row("B", 0, 300, mm_per_h=0.0),   # lead 5
        _mk_row("C", 0, 600, mm_per_h=2.0),   # lead 10
    ]
    # truth: station A observed 0.5 (over-forecast at 1.0 vs 0.5), B
    # observed 0.2 (predicted 0.0), C observed 2.0 (perfect hit). A/B/C
    # share the same lat/lon in _mk_row, so the lookup is keyed by call
    # order (rows are scored in the order passed in).
    truth_sequence = iter([0.5, 0.2, 2.0])

    def truth_lookup(lat, lon, valid_epoch):
        return next(truth_sequence)

    result = eb.score_against_truth(rows, truth_lookup, thresholds=(0.1, 1.0))

    assert set(result.keys()) == {5, 10}

    lead5 = result[5]
    assert lead5["n"] == 2
    # errors: (1.0 - 0.5) = 0.5, (0.0 - 0.2) = -0.2
    assert lead5["bias"] == pytest.approx((0.5 - 0.2) / 2)
    assert lead5["rmse"] == pytest.approx(((0.5**2 + 0.2**2) / 2) ** 0.5)
    # threshold 0.1: A predicted 1.0>=0.1 & obs 0.5>=0.1 -> hit.
    #                B predicted 0.0<0.1  & obs 0.2>=0.1 -> miss.
    # CSI = hits / (hits+misses+FA) = 1 / 2
    assert lead5["csi_0.1"] == pytest.approx(0.5)
    # threshold 1.0: A predicted 1.0>=1.0 & obs 0.5<1.0 -> false alarm.
    #                B predicted 0.0<1.0 & obs 0.2<1.0 -> correct negative (ignored).
    # CSI = 0 / 1 = 0.0
    assert lead5["csi_1.0"] == pytest.approx(0.0)

    lead10 = result[10]
    assert lead10["n"] == 1
    assert lead10["bias"] == pytest.approx(0.0)
    assert lead10["rmse"] == pytest.approx(0.0)
    assert lead10["csi_0.1"] == pytest.approx(1.0)
    assert lead10["csi_1.0"] == pytest.approx(1.0)


def test_score_against_truth_skips_missing_truth():
    rows = [_mk_row("A", 0, 300, mm_per_h=1.0)]

    def truth_lookup(lat, lon, valid_epoch):
        return None

    result = eb.score_against_truth(rows, truth_lookup)
    assert result == {}


# ---------------------------------------------------------------------------
# Station list sanity
# ---------------------------------------------------------------------------

def test_stations_are_roughly_twenty_and_in_belgium_netherlands_bbox():
    assert 18 <= len(eb.STATIONS) <= 22
    names = [s[0] for s in eb.STATIONS]
    assert len(names) == len(set(names))  # unique names
    for _name, lat, lon in eb.STATIONS:
        assert 49.4 <= lat <= 53.6
        assert 2.5 <= lon <= 7.3
