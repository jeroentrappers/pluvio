"""Tests for tools.buienradar_eu. No network: every HTTP call goes through a
fake ``_open_url``, and the images are tiny synthetic PNGs built with Pillow."""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import pathlib
import sqlite3
import urllib.error

import numpy as np
import pytest
from PIL import Image
from tools import buienradar_eu as br

UTC = dt.UTC

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------
FRESH_KEY = "11111111-2222-3333-4444-555555555555"
COMPOSITE_URL = "https://processing-cdn.buienradar.nl/processing/eu/raincombined/current/webm/{}.png"
FORECAST_URL = (
    "https://processing-cdn.buienradar.nl/processing/eu/raincombined/forecast/runs/webm/{run}/{v}.png"
)


def composite_doc(newest: str, n: int = 4, offset: float = 2.0) -> dict:
    """A RadarMapRain15mEU document ending at ``newest`` (compact UTC)."""
    end = dt.datetime.strptime(newest, br.COMPACT).replace(tzinfo=UTC)
    times = []
    for i in range(n - 1, -1, -1):
        t = end - dt.timedelta(minutes=15 * i)
        times.append(
            {"timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"), "url": COMPOSITE_URL.format(t.strftime(br.COMPACT))}
        )
    return {
        "imagetype": "RadarMapRain15mEU",
        "timeOffset": offset,
        "ext": "png",
        "width": br.WIDTH,
        "height": br.HEIGHT,
        "timestamp": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "still": times[-1]["url"],
        "stilltimestamp": times[-1]["timestamp"],
        "times": times,
    }


def forecast_doc(run: str, n: int = 3, first_lead: int = 35, offset: float = 2.0) -> dict:
    """A RadarMapRain5mEU document for forecast run ``run`` (compact UTC)."""
    r = dt.datetime.strptime(run, br.COMPACT).replace(tzinfo=UTC)
    times = []
    for i in range(n):
        t = r + dt.timedelta(minutes=first_lead + 5 * i)
        times.append(
            {
                "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"),
                "url": FORECAST_URL.format(run=run, v=t.strftime(br.COMPACT)),
            }
        )
    return {
        "imagetype": "RadarMapRain5mEU",
        "timeOffset": offset,
        "ext": "png",
        "width": br.WIDTH,
        "height": br.HEIGHT,
        "timestamp": r.strftime("%Y-%m-%dT%H:%M:%S"),
        "still": times[0]["url"],
        "stilltimestamp": times[0]["timestamp"],
        "times": times,
    }


def png_bytes(colour=(0, 0, 0, 0), size=(2, 2)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, format="PNG")
    return buf.getvalue()


class FakeHttp:
    """Routes URLs to canned payloads and records every request."""

    def __init__(self, composite: dict, forecast: dict, *, page_key: str | None = None):
        self.composite = composite
        self.forecast = forecast
        self.page_key = page_key
        self.calls: list[str] = []
        self.frame_payloads: dict[str, bytes] = {}
        self.default_frame = png_bytes((10, 20, 30, 255))
        self.fail_urls: set[str] = set()
        self.metadata_status: int | None = None

    def __call__(self, url: str, timeout: float) -> bytes:
        self.calls.append(url)
        if url in self.fail_urls:
            raise urllib.error.URLError("boom")
        if url.startswith(br.PAGE_URL):
            if self.page_key is None:
                raise urllib.error.HTTPError(url, 404, "nope", None, None)
            return f"var x=1; window.apiKey = '{self.page_key}'; more".encode()
        if "/metadata/" in url:
            if self.metadata_status is not None:
                raise urllib.error.HTTPError(url, self.metadata_status, "no", None, None)
            doc = self.composite if "15mEU" in url else self.forecast
            return json.dumps(doc).encode()
        return self.frame_payloads.get(url, self.default_frame)

    @property
    def frame_calls(self) -> list[str]:
        return [u for u in self.calls if u.endswith(".png")]

    @property
    def metadata_calls(self) -> list[str]:
        return [u for u in self.calls if "/metadata/" in u]


@pytest.fixture()
def http(monkeypatch):
    fake = FakeHttp(composite_doc("202606151200"), forecast_doc("202606151145"))
    monkeypatch.setattr(br, "_open_url", fake)
    return fake


def run_tick(root, http, *, now="2026-06-15T12:10:00+00:00", **kw):
    kw.setdefault("sleep_between", 0.0)
    kw.setdefault("sleep", lambda _s: None)
    return br.collect(root, now=dt.datetime.fromisoformat(now), **kw)


# ---------------------------------------------------------------------------
# Metadata parsing + timestamp semantics
# ---------------------------------------------------------------------------
def test_parse_metadata_fields_and_ordering():
    meta = br.parse_metadata(composite_doc("202606151200", n=4))
    assert meta.imagetype == "RadarMapRain15mEU"
    assert (meta.width, meta.height) == (766, 652)
    assert meta.timestamp == dt.datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    assert [f.valid_id for f in meta.frames] == [
        "202606151115",
        "202606151130",
        "202606151145",
        "202606151200",
    ]
    assert meta.run is None  # composite frames carry no run id
    assert all(f.run_id is None for f in meta.frames)


def test_parse_metadata_timestamps_are_utc_not_local():
    # The document's naive timestamps are UTC; timeOffset is the *display*
    # offset the site adds on top (verified against the page's own labels).
    meta = br.parse_metadata(composite_doc("202606151200", offset=2.0))
    assert meta.timestamp == dt.datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    shown = br.local_from_utc(meta.timestamp, meta.time_offset_h)
    assert shown.strftime("%H:%M") == "14:00"


def test_parse_metadata_forecast_run_from_url():
    meta = br.parse_metadata(forecast_doc("202606151145", n=3))
    assert meta.run == dt.datetime(2026, 6, 15, 11, 45, tzinfo=UTC)
    assert [f.run_id for f in meta.frames] == ["202606151145"] * 3
    assert [f.valid_id for f in meta.frames] == [
        "202606151220",
        "202606151225",
        "202606151230",
    ]


def test_check_run_id_accepts_utc_run_and_rejects_local_shift(caplog):
    # Run id == document timestamp: the identity the UTC archive naming rests on.
    assert br.check_run_id(br.parse_metadata(forecast_doc("202606151145"))) is True

    # Simulate Buienradar switching the run id in the URL to local time
    # (+2 h in June): the check must fail loudly rather than silently
    # mis-filing two hours of archive.
    doc = forecast_doc("202606151145")
    for entry in doc["times"]:
        entry["url"] = entry["url"].replace("/webm/202606151145/", "/webm/202606151345/")
    with caplog.at_level("WARNING"):
        assert br.check_run_id(br.parse_metadata(doc)) is False
    assert "run id" in caplog.text


@pytest.mark.parametrize(
    ("utc", "expected_offset"),
    [
        ("2026-01-15T12:00:00", 1.0),  # CET
        ("2026-03-29T00:30:00", 1.0),  # 30 min before the spring-forward
        ("2026-03-29T01:30:00", 2.0),  # 30 min after it
        ("2026-06-15T12:00:00", 2.0),  # CEST
        ("2026-10-25T00:30:00", 2.0),  # before the fall-back
        ("2026-10-25T01:30:00", 1.0),  # after it
    ],
)
def test_time_offset_matches_amsterdam_across_dst(utc, expected_offset):
    when = dt.datetime.fromisoformat(utc).replace(tzinfo=UTC)
    assert br.amsterdam_offset_hours(when) == expected_offset

    doc = composite_doc(when.strftime(br.COMPACT), n=2, offset=expected_offset)
    meta = br.parse_metadata(doc)
    assert br.check_time_offset(meta) is True
    # The UTC instants (and therefore the archive paths) do not move with DST.
    assert meta.frames[-1].valid == when
    assert br.composite_relpath(when).endswith(when.strftime(br.COMPACT) + "Z.png")
    # ...but the clock time the site shows does.
    assert br.local_from_utc(when, expected_offset) == when + dt.timedelta(hours=expected_offset)


def test_check_time_offset_flags_a_stale_offset(caplog):
    # A summer timestamp carrying the winter offset means the feed's timestamp
    # convention changed under us -> warn, do not silently keep archiving.
    meta = br.parse_metadata(composite_doc("202606151200", offset=1.0))
    with caplog.at_level("WARNING"):
        assert br.check_time_offset(meta) is False
    assert "timeOffset" in caplog.text


def test_dst_fall_back_hour_is_unambiguous_in_utc():
    # 02:30 local occurs twice on 2026-10-25; in UTC the two frames are
    # distinct instants and get distinct archive paths.
    first = dt.datetime(2026, 10, 25, 0, 30, tzinfo=UTC)
    second = dt.datetime(2026, 10, 25, 1, 30, tzinfo=UTC)
    assert br.local_from_utc(first, br.amsterdam_offset_hours(first)).strftime("%H:%M") == "02:30"
    assert br.local_from_utc(second, br.amsterdam_offset_hours(second)).strftime("%H:%M") == "02:30"
    assert br.composite_relpath(first) != br.composite_relpath(second)


def test_parse_metadata_trusts_the_url_id_over_a_disagreeing_timestamp(caplog):
    doc = composite_doc("202606151200", n=1)
    doc["times"][0]["timestamp"] = "2026-06-15T14:00:00"  # local leaked in
    with caplog.at_level("WARNING"):
        meta = br.parse_metadata(doc)
    assert meta.frames[0].valid_id == "202606151200"
    assert "url id" in caplog.text


def test_parse_metadata_skips_broken_entries_but_keeps_the_rest(caplog):
    doc = composite_doc("202606151200", n=3)
    doc["times"].insert(1, {"timestamp": "not-a-time", "url": "x"})
    with caplog.at_level("WARNING"):
        meta = br.parse_metadata(doc)
    assert len(meta.frames) == 3


def test_parse_metadata_rejects_a_document_with_no_frames():
    doc = composite_doc("202606151200", n=1)
    doc["times"] = []
    with pytest.raises(ValueError):
        br.parse_metadata(doc)


def test_parse_metadata_rejects_a_malformed_document():
    with pytest.raises(ValueError):
        br.parse_metadata({"imagetype": "x"})


def test_metadata_url_uses_full_size_and_the_documented_windows():
    url = br.metadata_url("composite", "KEY")
    assert "RadarMapRain15mEU" in url and "size=full" in url
    assert "forecast=0&history=12" in url and "ak=KEY" in url
    url = br.metadata_url("forecast", "KEY")
    assert "RadarMapRain5mEU" in url and "forecast=36&history=0" in url


# ---------------------------------------------------------------------------
# HTTP politeness
# ---------------------------------------------------------------------------
def test_user_agent_names_pluvio_and_a_contact():
    assert "pluvio" in br.USER_AGENT
    assert "github.com/jeroentrappers/pluvio" in br.USER_AGENT


def test_http_get_retries_with_exponential_backoff(monkeypatch):
    attempts = []
    delays = []

    def flaky(url, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise urllib.error.URLError("transient")
        return b"ok"

    monkeypatch.setattr(br, "_open_url", flaky)
    got = br.http_get("https://x/y", retries=3, backoff=1.0, sleep=delays.append)
    assert got == b"ok"
    assert len(attempts) == 3
    assert delays == [1.0, 2.0]


def test_http_get_gives_up_and_reraises(monkeypatch):
    monkeypatch.setattr(br, "_open_url", lambda u, t: (_ for _ in ()).throw(urllib.error.URLError("no")))
    with pytest.raises(urllib.error.URLError):
        br.http_get("https://x/y", retries=2, backoff=0.0, sleep=lambda _s: None)


def test_http_get_does_not_retry_a_client_error(monkeypatch):
    calls = []

    def denied(url, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 403, "denied", None, None)

    monkeypatch.setattr(br, "_open_url", denied)
    with pytest.raises(br.HttpError) as exc:
        br.http_get("https://x/y", retries=3, backoff=0.0, sleep=lambda _s: None)
    assert exc.value.status == 403
    assert len(calls) == 1  # a rotated key is not fixed by hammering


def test_http_get_retries_a_429(monkeypatch):
    calls = []

    def throttled(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "slow down", None, None)
        return b"ok"

    monkeypatch.setattr(br, "_open_url", throttled)
    assert br.http_get("https://x/y", retries=2, backoff=0.0, sleep=lambda _s: None) == b"ok"


def test_collect_spaces_image_fetches_apart(tmp_path, http):
    slept = []
    run_tick(tmp_path, http, sleep_between=0.2, sleep=slept.append)
    # 4 composite + 3 forecast frames -> one gap before each fetch but the
    # first of each feed.
    assert slept == [0.2] * 5


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
def test_api_key_is_cached_and_reused(tmp_path):
    assert br.load_api_key(tmp_path) == br.DEFAULT_API_KEY
    br.store_api_key(tmp_path, "cached-key")
    assert br.load_api_key(tmp_path) == "cached-key"
    assert br.api_key_path(tmp_path).read_text().strip() == "cached-key"


@pytest.mark.parametrize("status", [400, 401, 403])
def test_rotated_api_key_is_rescraped_and_cached(tmp_path, monkeypatch, status):
    fake = FakeHttp(composite_doc("202606151200"), forecast_doc("202606151145"), page_key=FRESH_KEY)
    calls = {"n": 0}
    inner = fake.__call__

    def gated(url, timeout):
        if "/metadata/" in url and f"ak={FRESH_KEY}" not in url:
            calls["n"] += 1
            raise urllib.error.HTTPError(url, status, "stale key", None, None)
        return inner(url, timeout)

    monkeypatch.setattr(br, "_open_url", gated)
    doc, _payload, key = br.fetch_metadata("composite", tmp_path, retries=1, backoff=0.0)
    assert key == FRESH_KEY
    assert doc["imagetype"] == "RadarMapRain15mEU"
    assert br.load_api_key(tmp_path) == FRESH_KEY
    assert calls["n"] == 1
    assert any(u.startswith(br.PAGE_URL) for u in fake.calls)


def test_api_key_rescrape_failure_surfaces_the_http_error(tmp_path, monkeypatch):
    fake = FakeHttp(composite_doc("202606151200"), forecast_doc("202606151145"), page_key=None)
    fake.metadata_status = 401
    monkeypatch.setattr(br, "_open_url", fake)
    with pytest.raises(br.HttpError):
        br.fetch_metadata("composite", tmp_path, retries=1, backoff=0.0)


def test_scrape_api_key_extracts_window_apikey(monkeypatch):
    monkeypatch.setattr(
        br,
        "_open_url",
        lambda u, t: b"<script>window.apiKey = '3c4a3037-85e6-4d1e-ad6c-f3f6e4b75f2f';</script>",
    )
    assert br.scrape_api_key(retries=1) == "3c4a3037-85e6-4d1e-ad6c-f3f6e4b75f2f"


# ---------------------------------------------------------------------------
# Collection: layout, dedupe, revisions, resilience
# ---------------------------------------------------------------------------
def test_collect_writes_the_documented_layout(tmp_path, http):
    stats, fatal = run_tick(tmp_path, http)
    assert fatal == []
    assert stats.downloaded == 7  # 4 composite + 3 forecast
    assert (tmp_path / "composite/2026/06/15/202606151200Z.png").exists()
    assert (tmp_path / "forecast/2026/06/15/202606151145Z/202606151220Z.png").exists()
    assert (tmp_path / "forecast/runs.jsonl").exists()
    assert (tmp_path / "georeference.json").exists()
    metas = sorted((tmp_path / "meta").rglob("*.json"))
    assert [p.parent.parent.parent.parent.name for p in metas] == ["composite", "forecast"]
    assert metas[0].parts[-4:-1] == ("2026", "06", "15")
    assert metas[0].name == "20260615121000Z.json"


def test_second_tick_fetches_nothing(tmp_path, http):
    run_tick(tmp_path, http)
    before = len(http.calls)
    stats, fatal = run_tick(tmp_path, http, now="2026-06-15T12:12:00+00:00")
    assert fatal == []
    assert stats.downloaded == 0
    assert stats.meta_stored == 0  # identical metadata -> not stored twice
    assert stats.runs_new == 0
    new = http.calls[before:]
    # Only the two metadata polls plus the revision-window re-checks.
    assert len(new) == 2 + 2 * 2
    assert stats.skipped == 7


def test_changed_metadata_is_archived_again(tmp_path, http):
    run_tick(tmp_path, http)
    http.composite = composite_doc("202606151215")
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:20:00+00:00")
    assert stats.meta_stored == 1
    assert stats.downloaded == 1
    assert len(list((tmp_path / "meta/composite").rglob("*.json"))) == 2
    assert len(list((tmp_path / "meta/forecast").rglob("*.json"))) == 1


def test_revised_frame_is_archived_beside_the_original(tmp_path, http):
    run_tick(tmp_path, http)
    newest = COMPOSITE_URL.format("202606151200")
    original = (tmp_path / "composite/2026/06/15/202606151200Z.png").read_bytes()
    http.frame_payloads[newest] = png_bytes((99, 88, 77, 255))

    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:12:00+00:00")
    assert stats.revised == 1
    assert stats.downloaded == 0
    revised = tmp_path / "composite/2026/06/15/202606151200Z.r1.png"
    assert revised.exists()
    assert (tmp_path / "composite/2026/06/15/202606151200Z.png").read_bytes() == original
    rows = sqlite3.connect(tmp_path / "index.sqlite").execute(
        "SELECT revision FROM frames WHERE valid_id = '202606151200' ORDER BY revision"
    ).fetchall()
    assert [r[0] for r in rows] == [0, 1]

    # A third tick with the revised bytes still in place adds nothing.
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:17:00+00:00")
    assert (stats.revised, stats.downloaded) == (0, 0)


def test_frames_outside_the_revision_window_are_never_refetched(tmp_path, http):
    run_tick(tmp_path, http)
    oldest = COMPOSITE_URL.format("202606151115")
    http.frame_payloads[oldest] = png_bytes((1, 2, 3, 255))
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:12:00+00:00", revision_window=1)
    assert oldest not in http.calls[len(http.calls) - 4 :]
    assert stats.revised == 0


def test_one_failing_frame_does_not_abort_the_tick(tmp_path, http):
    http.fail_urls.add(COMPOSITE_URL.format("202606151130"))
    stats, fatal = run_tick(tmp_path, http, http_kw={"retries": 1, "backoff": 0.0})
    assert fatal == []
    assert stats.failed == 1
    assert stats.downloaded == 6
    assert not (tmp_path / "composite/2026/06/15/202606151130Z.png").exists()

    # ...and it is picked up on the next tick, once the frame is available.
    http.fail_urls.clear()
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:12:00+00:00")
    assert stats.downloaded == 1
    assert (tmp_path / "composite/2026/06/15/202606151130Z.png").exists()


def test_metadata_failure_is_fatal_for_that_feed_only(tmp_path, monkeypatch):
    fake = FakeHttp(composite_doc("202606151200"), forecast_doc("202606151145"))
    forecast_meta = br.metadata_url("forecast", br.DEFAULT_API_KEY)
    inner = fake.__call__

    def partly_broken(url, timeout):
        if url == forecast_meta:
            raise urllib.error.URLError("down")
        return inner(url, timeout)

    monkeypatch.setattr(br, "_open_url", partly_broken)
    stats, fatal = run_tick(tmp_path, fake, http_kw={"retries": 1, "backoff": 0.0})
    assert len(fatal) == 1 and fatal[0].startswith("forecast:")
    assert stats.downloaded == 4  # the composite still got archived
    assert not (tmp_path / "forecast/runs.jsonl").exists()


def test_cli_collect_exit_codes(tmp_path, monkeypatch, http):
    argv = ["collect", "--root", str(tmp_path), "--sleep", "0"]
    assert br.main(argv) == 0
    http.fail_urls.add(COMPOSITE_URL.format("202606151130"))
    # A frame failure is a warning, not a failure of the tick.
    assert br.main([*argv, "--retries", "1"]) == 0
    http.metadata_status = 500
    assert br.main([*argv, "--retries", "1"]) == 1


def test_dry_run_touches_nothing(tmp_path, http):
    stats, fatal = run_tick(tmp_path, http, dry_run=True)
    assert fatal == []
    assert stats.downloaded == 7
    assert http.frame_calls == []  # metadata only
    assert list(tmp_path.iterdir()) == []


def test_dry_run_against_a_populated_archive_reports_only_the_gap(tmp_path, http):
    run_tick(tmp_path, http)
    http.composite = composite_doc("202606151215")
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:20:00+00:00", dry_run=True)
    assert stats.downloaded == 1
    assert stats.skipped == 6


def test_image_size_change_is_flagged(tmp_path, http, caplog):
    http.composite = composite_doc("202606151200")
    http.composite["width"] = 550
    http.composite["height"] = 468
    with caplog.at_level("WARNING"):
        run_tick(tmp_path, http)
    assert "georeference may be stale" in caplog.text


# ---------------------------------------------------------------------------
# Run ledger + cadence
# ---------------------------------------------------------------------------
def test_run_ledger_records_first_sight_once(tmp_path, http):
    run_tick(tmp_path, http, now="2026-06-15T12:10:00+00:00")
    run_tick(tmp_path, http, now="2026-06-15T12:15:00+00:00")
    entries = [json.loads(line) for line in br.runs_ledger(tmp_path).read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["run"] == "202606151145"
    assert entries[0]["first_seen"] == "2026-06-15T12:10:00+00:00"
    assert entries[0]["frames"] == 3
    assert entries[0]["lead_min"] == [35, 45]
    assert entries[0]["first_valid"] == "202606151220"


def test_run_ledger_appends_an_update_when_a_run_gains_frames(tmp_path, http):
    run_tick(tmp_path, http)
    http.forecast = forecast_doc("202606151145", n=5)
    run_tick(tmp_path, http, now="2026-06-15T12:15:00+00:00")
    entries = [json.loads(line) for line in br.runs_ledger(tmp_path).read_text().splitlines()]
    assert len(entries) == 2
    assert entries[1]["update"] is True
    assert entries[1]["frames"] == 5
    # first_seen is preserved so the cadence signal is not corrupted...
    assert entries[1]["first_seen"] == entries[0]["first_seen"]
    # ...and read_runs ignores update lines.
    assert len(br.read_runs(tmp_path)) == 1


def test_new_run_appends_a_new_ledger_line(tmp_path, http):
    run_tick(tmp_path, http, now="2026-06-15T12:10:00+00:00")
    http.forecast = forecast_doc("202606151200")
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:25:00+00:00")
    assert stats.runs_new == 1
    runs = br.read_runs(tmp_path)
    assert [r["run"] for r in runs] == ["202606151145", "202606151200"]
    assert (tmp_path / "forecast/2026/06/15/202606151200Z").is_dir()


def test_cadence_summary_from_the_ledger(tmp_path):
    ledger = br.runs_ledger(tmp_path)
    ledger.parent.mkdir(parents=True)
    lines = [
        {
            "run": "202606151100",
            "run_utc": "2026-06-15T11:00:00+00:00",
            "first_valid": "202606151135",
            "last_valid": "202606151315",
            "frames": 30,
            "first_seen": "2026-06-15T11:26:00+00:00",
            "lead_min": [35, 135],
        },
        {
            "run": "202606151115",
            "run_utc": "2026-06-15T11:15:00+00:00",
            "first_valid": "202606151150",
            "last_valid": "202606151330",
            "frames": 30,
            "first_seen": "2026-06-15T11:40:00+00:00",
            "lead_min": [35, 135],
        },
        {
            "run": "202606151145",
            "run_utc": "2026-06-15T11:45:00+00:00",
            "first_valid": "202606151220",
            "last_valid": "202606151400",
            "frames": 28,
            "first_seen": "2026-06-15T12:12:00+00:00",
            "lead_min": [35, 135],
        },
        # An update line must not be counted as another run.
        {"run": "202606151145", "run_utc": "2026-06-15T11:45:00+00:00", "frames": 29, "update": True},
    ]
    ledger.write_text("".join(json.dumps(x) + "\n" for x in lines))

    summary = br.cadence_summary(tmp_path)
    assert summary["runs"] == 3
    assert summary["first_run"] == "202606151100"
    assert summary["last_run"] == "202606151145"
    assert summary["gap_min"] == {"min": 15.0, "median": 30.0, "max": 30.0, "mean": 22.5}
    assert summary["first_seen_lag_min"]["min"] == 25.0
    assert summary["first_seen_lag_min"]["max"] == 27.0
    assert summary["frames_per_run"] == [28, 30]
    assert summary["lead_min"] == [(35, 135)]


def test_cadence_summary_on_an_empty_archive(tmp_path):
    assert br.cadence_summary(tmp_path) == {"runs": 0}


def test_read_runs_tolerates_a_torn_line(tmp_path, caplog):
    ledger = br.runs_ledger(tmp_path)
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"run": "202606151100", "run_utc": "2026-06-15T11:00:00+00:00"}\n{"run":\n')
    with caplog.at_level("WARNING"):
        runs = br.read_runs(tmp_path)
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# Index integrity
# ---------------------------------------------------------------------------
def test_index_matches_the_files_on_disk(tmp_path, http):
    run_tick(tmp_path, http)
    report = br.index_check(tmp_path)
    assert report["frames_indexed"] == 7
    assert report["frames_on_disk"] == 7
    assert report == {
        "frames_indexed": 7,
        "frames_on_disk": 7,
        "missing_files": [],
        "sha_mismatch": [],
        "unindexed_files": [],
        "revisions": 0,
    }


def test_index_check_reports_damage(tmp_path, http):
    run_tick(tmp_path, http)
    (tmp_path / "composite/2026/06/15/202606151115Z.png").unlink()
    (tmp_path / "composite/2026/06/15/202606151130Z.png").write_bytes(b"corrupt")
    (tmp_path / "composite/2026/06/15/stray.png").write_bytes(b"x")
    report = br.index_check(tmp_path)
    assert report["missing_files"] == ["composite/2026/06/15/202606151115Z.png"]
    assert report["sha_mismatch"] == ["composite/2026/06/15/202606151130Z.png"]
    assert report["unindexed_files"] == ["composite/2026/06/15/stray.png"]


def test_index_rows_carry_hash_size_and_fetch_time(tmp_path, http):
    run_tick(tmp_path, http)
    conn = sqlite3.connect(tmp_path / "index.sqlite")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM frames WHERE kind='forecast' AND valid_id='202606151220'"
    ).fetchone()
    payload = (tmp_path / row["path"]).read_bytes()
    assert row["sha256"] == br.sha256(payload)
    assert row["bytes"] == len(payload)
    assert row["run_id"] == "202606151145"
    assert row["fetched_at"] == "2026-06-15T12:10:00+00:00"
    assert row["url"].endswith("/202606151145/202606151220.png")


def test_a_deleted_file_is_restored_in_place(tmp_path, http, caplog):
    # An index row whose file has gone missing must not mask the gap: the
    # frame is refetched, and because the bytes match the row we already have
    # it is restored at its own path rather than minted as a revision.
    run_tick(tmp_path, http)
    gone = tmp_path / "composite/2026/06/15/202606151115Z.png"
    original = gone.read_bytes()
    gone.unlink()
    with caplog.at_level("WARNING"):
        stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:12:00+00:00")
    assert "the file is missing" in caplog.text
    assert (stats.restored, stats.downloaded, stats.revised) == (1, 0, 0)
    assert gone.read_bytes() == original
    assert br.index_check(tmp_path)["missing_files"] == []


def test_a_deleted_file_with_no_index_row_is_redownloaded(tmp_path, http):
    run_tick(tmp_path, http)
    gone = tmp_path / "composite/2026/06/15/202606151115Z.png"
    gone.unlink()
    conn = sqlite3.connect(tmp_path / "index.sqlite")
    conn.execute("DELETE FROM frames WHERE valid_id = '202606151115'")
    conn.commit()
    conn.close()
    stats, _ = run_tick(tmp_path, http, now="2026-06-15T12:12:00+00:00")
    assert stats.downloaded == 1
    assert gone.exists()


# ---------------------------------------------------------------------------
# Georeference
# ---------------------------------------------------------------------------
def test_mercator_bounds_match_the_image_aspect():
    left, bottom, right, top = br.mercator_bounds()
    aspect = (top - bottom) / (right - left)
    assert aspect == pytest.approx(br.HEIGHT / br.WIDTH, rel=1e-3)
    # Web Mercator pixels are square; the equirectangular reading is not.
    a, _b, _c, _d, e, _f = br.affine_transform()
    assert a == pytest.approx(-e, rel=1e-3)
    assert a == pytest.approx(7048.3, rel=1e-3)
    assert pytest.approx(0.5567, rel=1e-3) == (br.NORTH - br.SOUTH) / (br.EAST - br.WEST)


def test_affine_transform_maps_corners_to_the_bounds():
    a, b, c, d, e, f = br.affine_transform()
    left, bottom, right, top = br.mercator_bounds()
    assert (c, f) == (left, top)  # pixel (0, 0) is the north-west corner
    assert c + a * br.WIDTH == pytest.approx(right)
    assert f + e * br.HEIGHT == pytest.approx(bottom)
    assert (b, d) == (0.0, 0.0)  # north-up, no rotation


def test_mercator_helpers_against_the_closed_form():
    assert br.mercator_x(0.0) == 0.0
    assert br.mercator_y(0.0) == pytest.approx(0.0, abs=1e-6)
    assert br.mercator_x(35.0) == pytest.approx(35.0 * br.EARTH_RADIUS_M * math.pi / 180.0)
    assert br.mercator_y(-34.0) == pytest.approx(-br.mercator_y(34.0))


def test_georeference_sidecar_is_self_describing(tmp_path, http):
    run_tick(tmp_path, http)
    doc = json.loads((tmp_path / "georeference.json").read_text())
    assert doc["crs"] == "EPSG:3857"
    assert doc["row_0"] == "north"
    assert (doc["width"], doc["height"]) == (766, 652)
    assert doc["latlon_corner_bounds"] == {"north": 61.0, "west": -13.5, "south": 34.0, "east": 35.0}
    assert doc["transform"] == list(br.affine_transform())


def test_world_file_references_pixel_centres(tmp_path, http):
    run_tick(tmp_path, http)
    values = [float(v) for v in (tmp_path / "frame.pgw").read_text().split()]
    a, b, c, d, e, f = br.affine_transform()
    assert values == pytest.approx([a, d, b, e, c + a / 2, f + e / 2])


def test_rasterio_transform_agrees_with_the_tuple():
    rio = pytest.importorskip("rasterio")
    assert tuple(br.rasterio_transform())[:6] == pytest.approx(br.affine_transform())
    assert rio  # silence the unused-import lint


# ---------------------------------------------------------------------------
# Palette decoding
# ---------------------------------------------------------------------------
def ramp_png(path: pathlib.Path, colours) -> pathlib.Path:
    """A 1-pixel-tall RGBA PNG of ``colours`` (an opaque pixel per colour)."""
    img = Image.new("RGBA", (len(colours), 1))
    img.putdata([(*c, 204) for c in colours])
    img.save(path)
    return path


def test_palette_table_is_the_published_legend():
    assert [p[0] for p in br.PALETTE] == ["#e4e5ff", "#4d5dff", "#000770", "#fe1600", "#c01cc4"]
    assert [(p[1], p[2]) for p in br.PALETTE] == [
        (0.0, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
        (10.0, 100.0),
        (100.0, None),
    ]
    assert br.hex_to_rgb("#e4e5ff") == (228, 229, 255)
    assert br.hex_to_rgb("#c01cc4") == br.hex_to_rgb("#C01CC4")


def test_ramp_anchors_are_exactly_on_the_ramp():
    t, dist = br.ramp_position(np.asarray(br.RAMP))
    assert dist == pytest.approx(np.zeros(len(br.RAMP)), abs=1e-3)
    assert list(t) == sorted(t)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(1.0)


def test_ramp_position_is_monotonic_along_the_ramp():
    # Walk the polyline in 200 steps: t must increase monotonically.
    pts = np.asarray(br.RAMP, dtype="float64")
    samples = []
    for i in range(len(pts) - 1):
        for u in np.linspace(0, 1, 50, endpoint=False):
            samples.append(pts[i] + u * (pts[i + 1] - pts[i]))
    samples.append(pts[-1])
    t, dist = br.ramp_position(np.asarray(samples))
    assert dist.max() < 1e-3
    assert np.all(np.diff(t) > 0)
    rate = br.rate_from_position(t)
    # Never decreasing, and strictly increasing once past the dry floor the
    # lightest colours are clamped to.
    assert np.all(np.diff(rate) >= 0)
    above = t > br.RATE_ANCHORS[0][0]
    assert np.all(np.diff(rate[above]) > 0)


def test_rate_anchors_reproduce_the_published_class_boundaries():
    ts = [a[0] for a in br.RATE_ANCHORS]
    assert br.rate_from_position(ts) == pytest.approx([0.1, 2.0, 5.0, 10.0, 100.0], rel=1e-4)


def test_ramp_colour_inverts_ramp_position():
    for t in (0.0, 0.05, 0.2501, 0.5, 0.7591, 1.0):
        back, dist = br.ramp_position(np.asarray([br.ramp_colour(t)]))
        assert dist[0] < 1.0
        assert back[0] == pytest.approx(t, abs=2e-3)


def test_png_to_rate_round_trips_the_rate_anchors(tmp_path):
    # Paint the exact colour each mm/h anchor sits at; the decoder must give
    # that anchor's rate back.
    anchors = [(t, rate) for t, rate in br.RATE_ANCHORS]
    path = ramp_png(tmp_path / "anchors.png", [br.ramp_colour(t) for t, _ in anchors])
    rate = br.png_to_rate(path)
    assert rate.dtype == np.dtype("float32")
    assert rate.shape == (1, len(anchors))
    assert rate[0] == pytest.approx([r for _, r in anchors], rel=2e-2)


def test_png_to_rate_round_trips_the_ramp_control_points(tmp_path):
    path = ramp_png(tmp_path / "ramp.png", br.RAMP)
    rate = br.png_to_rate(path)
    t, _ = br.ramp_position(np.asarray(br.RAMP))
    assert rate[0] == pytest.approx(br.rate_from_position(t), rel=1e-3)


def test_png_to_rate_round_trips_intermediate_ramp_colours(tmp_path):
    pts = np.asarray(br.RAMP, dtype="float64")
    mids = [tuple(np.round((pts[i] + pts[i + 1]) / 2).astype(int)) for i in range(len(pts) - 1)]
    path = ramp_png(tmp_path / "mid.png", mids)
    t, _ = br.ramp_position(np.asarray(mids))
    assert br.png_to_rate(path)[0] == pytest.approx(br.rate_from_position(t), rel=1e-3)


def test_png_to_rate_is_nan_for_transparent_and_off_ramp_pixels(tmp_path):
    img = Image.new("RGBA", (3, 1))
    img.putdata([
        (0, 0, 0, 0),  # transparent -> no data
        (0, 128, 0, 255),  # a colour nowhere near the ramp -> no data
        (253, 23, 2, 204),  # the red anchor -> 10 mm/h
    ])
    img.save(tmp_path / "mixed.png")
    rate = br.png_to_rate(tmp_path / "mixed.png")
    assert math.isnan(rate[0, 0])
    assert math.isnan(rate[0, 1])
    assert rate[0, 2] == pytest.approx(10.0, rel=1e-3)


def test_png_to_class_matches_the_published_classes(tmp_path):
    # One colour per legend class, taken at the midpoint of the class's span
    # along the ramp, plus a transparent pixel.
    edges = [0.0] + [a[0] for a in br.RATE_ANCHORS[1:]]
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)] + [1.0]
    img = Image.new("RGBA", (len(mids) + 1, 1))
    img.putdata([(0, 0, 0, 0)] + [(*br.ramp_colour(t), 204) for t in mids])
    img.save(tmp_path / "cls.png")
    cls = br.png_to_class(tmp_path / "cls.png")
    assert cls.dtype == np.dtype("int8")
    assert list(cls[0]) == [-1, 0, 1, 2, 3, 4]


def test_class_and_rate_agree_on_the_class_a_rate_falls_in(tmp_path):
    colours = [br.ramp_colour(t) for t in np.linspace(0.02, 1.0, 40)]
    path = ramp_png(tmp_path / "ramp2.png", colours)
    rate = br.png_to_rate(path)[0]
    cls = br.png_to_class(path)[0]
    for r, c in zip(rate, cls, strict=True):
        lo, hi = br.PALETTE[c][1], br.PALETTE[c][2]
        assert r >= lo - 1e-3
        if hi is not None:
            assert r <= hi + 1e-3


def test_png_to_rate_accepts_a_palette_png_like_the_real_frames(tmp_path):
    # The real frames are mode "P" with a per-index transparency table; the
    # decoder must not care, because it converts to RGBA first.
    img = Image.new("P", (2, 1))
    img.putpalette([0, 0, 0, 253, 23, 2])
    img.putdata([0, 1])
    img.save(tmp_path / "p.png", transparency=0)
    rate = br.png_to_rate(tmp_path / "p.png")
    assert rate.shape == (1, 2)
    assert math.isnan(rate[0, 0])
    assert rate[0, 1] == pytest.approx(10.0, rel=1e-3)


def test_cli_decode_and_verify(tmp_path, http, capsys):
    run_tick(tmp_path, http)
    assert br.main(["verify", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["frames_indexed"] == 7

    ramp_png(tmp_path / "ramp.png", br.RAMP)
    out = tmp_path / "rate.npy"
    assert br.main(["decode", "--png", str(tmp_path / "ramp.png"), "--out", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["wet_px"] == len(br.RAMP)
    assert report["max_mm_h"] == pytest.approx(100.0, rel=1e-3)
    assert "PROVISIONAL" in report["note"]
    assert np.load(out).shape == (1, len(br.RAMP))


def test_cli_verify_fails_on_a_damaged_archive(tmp_path, http, capsys):
    run_tick(tmp_path, http)
    (tmp_path / "composite/2026/06/15/202606151115Z.png").unlink()
    assert br.main(["verify", "--root", str(tmp_path)]) == 1
    capsys.readouterr()


def test_cli_cadence(tmp_path, http, capsys):
    run_tick(tmp_path, http)
    assert br.main(["cadence", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 1
    assert br.main(["cadence", "--root", str(tmp_path), "--days", "1"]) == 0
    capsys.readouterr()
