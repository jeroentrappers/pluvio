"""Tests for tools.external_baselines. No network access: HTTP is mocked and
all payloads come from tests/fixtures/."""

from __future__ import annotations

import datetime as dt
import itertools
import math
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
    # 0 is below the dry floor -> snapped to exactly 0.0.
    assert eb.value_to_mm_per_h(0) == 0.0
    # 255 is the byte's formal ceiling; the log scale sends it to an
    # enormous, never-observed rate -- that's the encoding, not a bug.
    assert eb.value_to_mm_per_h(255) == pytest.approx(36517.4, rel=1e-3)


def test_value_to_mm_per_h_dry_floor_boundary():
    # value 45 -> raw exactly 0.01 == DRY_FLOOR_MM_H -> kept, not floored.
    assert eb.value_to_mm_per_h(45) == pytest.approx(0.01)
    # value 44 -> raw just under the floor -> snapped to 0.0.
    assert eb.value_to_mm_per_h(44) == 0.0


def test_value_to_mm_per_h_monotonic():
    values = [0, 30, 44, 45, 50, 109, 150, 200, 255]
    rates = [eb.value_to_mm_per_h(v) for v in values]
    assert rates == sorted(rates)


# ---------------------------------------------------------------------------
# parse_raintext: real payload (live-recorded, all-zero-rain)
# ---------------------------------------------------------------------------

def test_parse_raintext_live_fixture_no_rain():
    text = read_fixture("buienradar_brussels_live_20260903.txt")
    issue = dt.datetime(2026, 9, 3, 8, 27, 44, tzinfo=dt.UTC)
    rows = eb.parse_raintext(text, issue)

    non_blank_lines = [l for l in text.splitlines() if l.strip()]
    assert len(rows) == len(non_blank_lines) == 24

    # first line is 10:30 local (CEST, UTC+2) on issue day.
    first_epoch, first_rate = rows[0]
    first_dt = dt.datetime.fromtimestamp(first_epoch, dt.UTC)
    assert first_dt == dt.datetime(2026, 9, 3, 8, 30, tzinfo=dt.UTC)
    assert first_rate == 0.0  # value 0 is below the dry floor

    # steps are 5 minutes apart, strictly increasing.
    epochs = [e for e, _ in rows]
    assert epochs == sorted(epochs)
    assert all(b - a == 300 for a, b in itertools.pairwise(epochs))


# ---------------------------------------------------------------------------
# parse_raintext: local-midnight rollover (no DST involved)
# ---------------------------------------------------------------------------

def test_parse_raintext_day_rollover():
    text = read_fixture("buienradar_rollover_20260630.txt")
    # issued a few minutes before local midnight, CEST (UTC+2).
    issue = dt.datetime(2026, 6, 30, 21, 52, 0, tzinfo=dt.UTC)
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

    # epochs strictly increase across the rollover, every step 5 minutes.
    epochs = [e for e, _ in rows]
    assert epochs == sorted(epochs)
    assert len(set(epochs)) == len(epochs)
    assert all(b - a == 300 for a, b in itertools.pairwise(epochs))


# ---------------------------------------------------------------------------
# parse_raintext: DST fall-back (October) -- the repeated local hour
# ---------------------------------------------------------------------------

def test_parse_raintext_dst_fallback_repeated_hour():
    text = read_fixture("buienradar_fallback_20261025.txt")
    # issued just before the repeated hour, while still CEST.
    issue = dt.datetime(2026, 10, 24, 23, 50, 0, tzinfo=dt.UTC)
    rows = eb.parse_raintext(text, issue)
    # 01:50, 01:55, then 02:00..02:55 (CEST, 12 lines), then 02:00..02:55
    # again (CET, 12 lines), then 03:00, 03:05 -- 28 lines total.
    assert len(rows) == 28

    epochs = [e for e, _ in rows]
    # monotonic and evenly spaced by 5 minutes throughout, including
    # across the fold -- this is the whole point of the closest-epoch
    # anchoring: a naive "clock went backwards -> new day" heuristic would
    # instead jump the date forward here and blow the lead out to ~1512 min.
    assert epochs == sorted(epochs)
    assert len(set(epochs)) == len(epochs)
    assert all(b - a == 300 for a, b in itertools.pairwise(epochs))

    # first "02:00" (fold=0, still CEST/UTC+2) and second "02:00" (fold=1,
    # now CET/UTC+1) are exactly one hour apart in absolute time.
    first_0200_epoch = epochs[2]
    second_0200_epoch = epochs[14]
    assert second_0200_epoch - first_0200_epoch == 3600

    # sanity: 28 lines is 27 gaps of 5 real minutes each -- the repeated
    # hour is already accounted for by the fixture listing it twice, no
    # separate adjustment needed.
    assert epochs[-1] - epochs[0] == 27 * 300

    # and every epoch really does land in UTC where we expect.
    first_dt = dt.datetime.fromtimestamp(epochs[0], dt.UTC)
    assert first_dt == dt.datetime(2026, 10, 24, 23, 50, tzinfo=dt.UTC)
    last_dt = dt.datetime.fromtimestamp(epochs[-1], dt.UTC)
    assert last_dt == dt.datetime(2026, 10, 25, 2, 5, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# parse_raintext: DST spring-forward (March) -- the skipped local hour
# ---------------------------------------------------------------------------

def test_parse_raintext_dst_springforward_skipped_hour():
    text = read_fixture("buienradar_springforward_20260329.txt")
    # issued just before the gap, while still CET.
    issue = dt.datetime(2026, 3, 29, 0, 50, 0, tzinfo=dt.UTC)
    rows = eb.parse_raintext(text, issue)
    assert len(rows) == 5

    epochs = [e for e, _ in rows]
    # the labels jump from 01:55 straight to 03:00 (02:xx never happened
    # locally) but the real cadence stays a perfectly even 5 minutes.
    assert epochs == sorted(epochs)
    assert all(b - a == 300 for a, b in itertools.pairwise(epochs))

    first_dt = dt.datetime.fromtimestamp(epochs[0], dt.UTC)
    assert first_dt == dt.datetime(2026, 3, 29, 0, 50, tzinfo=dt.UTC)
    last_dt = dt.datetime.fromtimestamp(epochs[-1], dt.UTC)
    assert last_dt == dt.datetime(2026, 3, 29, 1, 10, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# parse_raintext: malformed / out-of-range input tolerance
# ---------------------------------------------------------------------------

def test_parse_raintext_malformed_lines_are_skipped():
    text = read_fixture("buienradar_malformed.txt")
    issue = dt.datetime(2026, 9, 3, 10, 25, 0, tzinfo=dt.UTC)
    rows = eb.parse_raintext(text, issue)
    # valid: "000|10:30", "109|10:40", "109|10:50".
    # skipped: blank line, no-pipe line, non-numeric value ("abc|10:35"),
    # bad time ("000|xx:yy"), and "999|10:45" -- 999 is outside the feed's
    # defined 0-255 byte range and is rejected like any other malformed line
    # (not parsed as a legal, if absurd, rain rate).
    assert len(rows) == 3
    minutes = sorted(dt.datetime.fromtimestamp(e, eb.AMSTERDAM).minute for e, _ in rows)
    assert minutes == [30, 40, 50]


def test_parse_raintext_rejects_out_of_range_byte():
    issue = dt.datetime(2026, 9, 3, 10, 25, 0, tzinfo=dt.UTC)
    rows = eb.parse_raintext("999|10:30\n256|10:35\n-1|10:40\n109|10:45\n", issue)
    assert len(rows) == 1
    assert rows[0][1] == pytest.approx(eb.value_to_mm_per_h(109))


def test_parse_raintext_requires_integer_byte_token():
    issue = dt.datetime(2026, 9, 3, 10, 25, 0, tzinfo=dt.UTC)
    # "50.5" and "1e2" are not legal encodings of a 0-255 byte and must be
    # rejected like any other malformed line, not silently coerced by
    # float(). " 50 " (incidental whitespace around an otherwise-plain
    # integer) is fine and should still parse.
    text = "50.5|10:30\n1e2|10:35\n 50 |10:40\n109|10:45\n"
    rows = eb.parse_raintext(text, issue)
    assert len(rows) == 2
    rates = sorted(rate for _, rate in rows)
    assert rates == sorted([eb.value_to_mm_per_h(50), eb.value_to_mm_per_h(109)])


def test_parse_raintext_empty_string():
    issue = dt.datetime(2026, 9, 3, 10, 25, 0, tzinfo=dt.UTC)
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
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["user_agent"] = request.get_header("User-agent")
        return _FakeResponse(text)

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    source = eb.BuienradarSource()
    issue = dt.datetime(2026, 9, 3, 8, 27, 44, tzinfo=dt.UTC)
    rows = source.fetch_point(50.85, 4.35, issue)
    assert rows is not None
    assert len(rows) == 24
    assert "lat=50.85" in seen["url"] and "lon=4.35" in seen["url"]
    assert seen["user_agent"] == eb.USER_AGENT


def test_buienradar_source_fetch_point_http_error_returns_none(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    source = eb.BuienradarSource()
    issue = dt.datetime.now(dt.UTC)
    assert source.fetch_point(50.85, 4.35, issue) is None


def test_buienradar_source_fetch_point_incomplete_read_returns_none(monkeypatch):
    import http.client

    def fake_urlopen(request, timeout=None):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    source = eb.BuienradarSource()
    issue = dt.datetime.now(dt.UTC)
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


def test_sample_all_lead_min_is_relative_to_feed_t0_not_fetch_time():
    fetch_time = dt.datetime(2026, 9, 3, 10, 0, 0, tzinfo=dt.UTC)
    # feed's first line ("t0") is 3 minutes after the fetch instant, as
    # happens when the feed floors to its own 5-minute grid.
    t0 = int(fetch_time.timestamp()) + 180
    stations = [("A", 1.0, 2.0), ("B", 3.0, 4.0)]
    points = {
        (1.0, 2.0): [(t0, 0.5), (t0 + 300, 1.5), (t0 + 600, 2.5)],
        (3.0, 4.0): None,  # simulates a failed station
    }
    source = _FakeSource(points)
    rows = eb.sample_all(issue_time=fetch_time, stations=stations, source=source, delay_s=0)

    assert len(rows) == 3
    assert {r["station"] for r in rows} == {"A"}
    leads = sorted(r["lead_min"] for r in rows)
    assert leads == [0, 5, 10]  # exactly on-grid, not [3, 8, 13]

    row0 = rows[0]
    assert row0["source"] == "buienradar"
    assert row0["lat"] == 1.0 and row0["lon"] == 2.0
    assert row0["issue_epoch"] == t0            # snapped to the feed's own t0
    assert row0["fetch_epoch"] == int(fetch_time.timestamp())  # wall clock kept separately
    assert row0["lead_min"] == 0


def test_sample_all_drops_rows_with_lead_outside_sane_range():
    fetch_time = dt.datetime(2026, 9, 3, 10, 0, 0, tzinfo=dt.UTC)
    t0 = int(fetch_time.timestamp())
    stations = [("A", 1.0, 2.0)]
    points = {
        (1.0, 2.0): [
            (t0, 1.0),                 # lead 0: fine
            (t0 + 300, 1.0),           # lead 5: fine
            (t0 + 200 * 60, 1.0),      # lead 200: way outside [-5, 130] -> dropped
        ],
    }
    source = _FakeSource(points)
    rows = eb.sample_all(issue_time=fetch_time, stations=stations, source=source, delay_s=0)
    leads = sorted(r["lead_min"] for r in rows)
    assert leads == [0, 5]


# ---------------------------------------------------------------------------
# CLI station selection
# ---------------------------------------------------------------------------

def test_parse_stations_arg_preserves_requested_order():
    names_in_order = [name for name, _, _ in eb.STATIONS]
    last, first = names_in_order[-1], names_in_order[0]
    result = eb._parse_stations_arg(f"{last},{first}")
    assert [r[0] for r in result] == [last, first]


def test_parse_stations_arg_all():
    assert eb._parse_stations_arg("all") == eb.STATIONS


def test_parse_stations_arg_unknown_raises():
    with pytest.raises(SystemExit):
        eb._parse_stations_arg("NotAStation")


# ---------------------------------------------------------------------------
# CLI dry-run: no archive write happens
# ---------------------------------------------------------------------------

def test_cli_sample_dry_run_does_not_write(monkeypatch, tmp_path, capsys):
    called = {"append": False}

    def fake_sample_all(**kwargs):
        return [{"station": "Brussels"}]

    def fake_append_archive(rows, root):
        called["append"] = True
        return len(rows)

    monkeypatch.setattr(eb, "sample_all", fake_sample_all)
    monkeypatch.setattr(eb, "append_archive", fake_append_archive)

    eb.main(["sample", "--archive", str(tmp_path), "--dry-run", "--stations", "Brussels"])
    out = capsys.readouterr().out
    assert "not written" in out
    assert called["append"] is False
    assert list(tmp_path.iterdir()) == []


def test_cli_sample_requires_archive_or_env(monkeypatch):
    monkeypatch.delenv("PLUVIO_EXTERNAL_ROOT", raising=False)
    with pytest.raises(SystemExit):
        eb.main(["sample", "--stations", "Brussels"])


def test_cli_sample_uses_env_default_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("PLUVIO_EXTERNAL_ROOT", str(tmp_path))

    def fake_sample_all(**kwargs):
        return []

    monkeypatch.setattr(eb, "sample_all", fake_sample_all)
    # no --archive flag passed: should pick up PLUVIO_EXTERNAL_ROOT.
    eb.main(["sample", "--stations", "Brussels"])
    # no exception means the env default was accepted as the archive root.


# ---------------------------------------------------------------------------
# Archive: append + idempotency + load
# ---------------------------------------------------------------------------

def _mk_row(station, issue_epoch, valid_epoch, mm_per_h=1.0, fetch_epoch=None):
    return {
        "source": "buienradar",
        "station": station,
        "lat": 50.0,
        "lon": 4.0,
        "issue_epoch": issue_epoch,
        "fetch_epoch": fetch_epoch if fetch_epoch is not None else issue_epoch,
        "valid_epoch": valid_epoch,
        "lead_min": round((valid_epoch - issue_epoch) / 60),
        "mm_per_h": mm_per_h,
    }


def test_append_archive_writes_and_is_idempotent(tmp_path):
    day = dt.date(2026, 9, 3)
    issue_epoch = int(dt.datetime(2026, 9, 3, 10, 0, tzinfo=dt.UTC).timestamp())
    valid_epoch = issue_epoch + 600
    rows = [_mk_row("Brussels", issue_epoch, valid_epoch)]

    written1 = eb.append_archive(rows, tmp_path)
    assert written1 == 1

    # exact same batch again (same station + issue_epoch): no-op.
    written2 = eb.append_archive(rows, tmp_path)
    assert written2 == 0

    loaded = eb.load_archive(tmp_path, day)
    assert len(loaded) == 1
    assert loaded[0]["station"] == "Brussels"

    path = tmp_path / "buienradar" / "2026" / "09" / "03.jsonl"
    assert path.exists()
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1

    # a new fetch (different issue_epoch/t0) for the same station is a new
    # batch and does get appended.
    rows2 = [_mk_row("Brussels", issue_epoch + 600, valid_epoch + 600)]
    written3 = eb.append_archive(rows2, tmp_path)
    assert written3 == 1
    assert len(eb.load_archive(tmp_path, day)) == 2


def test_append_archive_batch_covers_all_rows_from_one_fetch(tmp_path):
    # multiple lead-time rows from a single station/fetch share (station,
    # issue_epoch) and are idempotent together, not row-by-row.
    issue_epoch = int(dt.datetime(2026, 9, 3, 10, 0, tzinfo=dt.UTC).timestamp())
    rows = [
        _mk_row("Brussels", issue_epoch, issue_epoch),
        _mk_row("Brussels", issue_epoch, issue_epoch + 300),
        _mk_row("Brussels", issue_epoch, issue_epoch + 600),
    ]
    written1 = eb.append_archive(rows, tmp_path)
    assert written1 == 3
    written2 = eb.append_archive(rows, tmp_path)
    assert written2 == 0
    assert len(eb.load_archive(tmp_path, dt.date(2026, 9, 3))) == 3


def test_append_archive_survives_deleted_index(tmp_path):
    issue_epoch = int(dt.datetime(2026, 9, 3, 10, 0, tzinfo=dt.UTC).timestamp())
    rows = [
        _mk_row("Brussels", issue_epoch, issue_epoch),
        _mk_row("Brussels", issue_epoch, issue_epoch + 300),
        _mk_row("Antwerp", issue_epoch, issue_epoch),
    ]
    written1 = eb.append_archive(rows, tmp_path)
    assert written1 == 3

    path = tmp_path / "buienradar" / "2026" / "09" / "03.jsonl"
    idx_path = eb._index_path(path)
    assert idx_path.exists()
    idx_path.unlink()

    # re-appending the exact same rows with the index gone must not
    # duplicate: the code has to fall back to scanning the JSONL (the
    # ground truth) instead of assuming "no index -> nothing written yet".
    written2 = eb.append_archive(rows, tmp_path)
    assert written2 == 0
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3  # not 6

    # and the index has been rebuilt, so a third call stays cheap/correct too.
    assert idx_path.exists()
    written3 = eb.append_archive(rows, tmp_path)
    assert written3 == 0
    assert len([l for l in path.read_text().splitlines() if l.strip()]) == 3


def test_append_archive_survives_truncated_index(tmp_path):
    issue_epoch = int(dt.datetime(2026, 9, 3, 10, 0, tzinfo=dt.UTC).timestamp())
    rows = [
        _mk_row("Brussels", issue_epoch, issue_epoch),
        _mk_row("Antwerp", issue_epoch, issue_epoch),
        _mk_row("Ghent", issue_epoch, issue_epoch),
    ]
    written1 = eb.append_archive(rows, tmp_path)
    assert written1 == 3

    path = tmp_path / "buienradar" / "2026" / "09" / "03.jsonl"
    idx_path = eb._index_path(path)
    original = idx_path.read_text()
    assert len(original.splitlines()) > 1

    # simulate a partial write / corruption: drop the last data line but
    # keep the (now-wrong) header claiming the old batch count.
    truncated = "\n".join(original.splitlines()[:-1]) + "\n"
    idx_path.write_text(truncated)

    written2 = eb.append_archive(rows, tmp_path)
    assert written2 == 0  # must not re-append rows the JSONL already has
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


def test_append_archive_index_header_mismatch_forces_rebuild(tmp_path):
    # a header whose recorded size no longer matches the JSONL (as if the
    # JSONL was appended to without updating the index) must not be trusted.
    issue_epoch = int(dt.datetime(2026, 9, 3, 10, 0, tzinfo=dt.UTC).timestamp())
    rows = [_mk_row("Brussels", issue_epoch, issue_epoch)]
    eb.append_archive(rows, tmp_path)

    path = tmp_path / "buienradar" / "2026" / "09" / "03.jsonl"
    idx_path = eb._index_path(path)
    idx_path.write_text(f"# size=999999 batches=1\nBrussels|{issue_epoch}\n")

    # re-appending the same row: the bogus header must trigger a JSONL
    # rebuild rather than being trusted at face value.
    written = eb.append_archive(rows, tmp_path)
    assert written == 0
    assert len([l for l in path.read_text().splitlines() if l.strip()]) == 1


def test_load_archive_missing_day_returns_empty(tmp_path):
    assert eb.load_archive(tmp_path, dt.date(2020, 1, 1)) == []


def test_append_archive_splits_by_utc_day_of_valid_epoch(tmp_path):
    # issue on day 1, valid_epoch on day 2 (near-midnight step): must land
    # in day 2's file since that's what a later truth-join keys on.
    issue_epoch = int(dt.datetime(2026, 9, 3, 23, 55, tzinfo=dt.UTC).timestamp())
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


def test_score_against_truth_skips_nan_truth():
    rows = [
        _mk_row("A", 0, 300, mm_per_h=1.0),  # lead 5, truth is NaN -> skipped
        _mk_row("B", 0, 300, mm_per_h=2.0),  # lead 5, truth is real -> counted
    ]
    truth_sequence = iter([math.nan, 2.0])

    def truth_lookup(lat, lon, valid_epoch):
        return next(truth_sequence)

    result = eb.score_against_truth(rows, truth_lookup, thresholds=(1.0,))
    assert result[5]["n"] == 1  # the NaN row did not poison the count
    assert result[5]["bias"] == pytest.approx(0.0)
    assert result[5]["rmse"] == pytest.approx(0.0)
    assert result[5]["csi_1.0"] == pytest.approx(1.0)  # not a false alarm


def test_score_against_truth_skips_numpy_float32_nan_truth():
    # truth arrays elsewhere in this repo (regional_eval.py) are float32,
    # not the builtin float -- an isinstance(truth, float) gate would let
    # this straight through and poison the bucket (n counted wrong,
    # rmse/bias -> nan).
    np = pytest.importorskip("numpy")
    rows = [
        _mk_row("A", 0, 300, mm_per_h=1.0),
        _mk_row("B", 0, 300, mm_per_h=2.0),
    ]
    truth_sequence = iter([np.float32("nan"), np.float32(2.0)])

    def truth_lookup(lat, lon, valid_epoch):
        return next(truth_sequence)

    result = eb.score_against_truth(rows, truth_lookup, thresholds=(1.0,))
    assert result[5]["n"] == 1
    assert not math.isnan(result[5]["rmse"])
    assert result[5]["rmse"] == pytest.approx(0.0)


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
