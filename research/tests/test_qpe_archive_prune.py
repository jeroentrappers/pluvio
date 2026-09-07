"""Retention policy of the QPE archiver — the part that deletes things.

Two bugs motivated these: the daily prune unit ran a 96-stamp backfill first and
regularly hit its timeout before pruning anything, and the cache sweep was gated
on archive coverage, so a stretch of un-archived days kept its (re-downloadable)
cache forever.
"""

from __future__ import annotations

import datetime as dt
import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture()
def qa(tmp_path, monkeypatch):
    """qpe_archive re-imported with its store roots pointed at tmp_path."""
    monkeypatch.setenv("PLUVIO_QPE_ROOT", str(tmp_path / "qpe"))
    mod = importlib.reload(importlib.import_module("tools.qpe_archive"))
    mod.RAW_STORES = ((tmp_path / "radar_volumes", "daydir"),
                      (tmp_path / "dwd_vol", "stampfile"))
    mod.CACHE_STORES = ((tmp_path / "knmi_vol", "stampfile"),)
    mod.CACHE_RETAIN_DAYS = 2
    mod.CACHE_HORIZON_DAYS = 400
    mod.CACHE_KEEP_SINCE = None
    mod.RETAIN_DAYS = 5
    return mod


def _cache_file(mod, day: dt.date) -> pathlib.Path:
    root = mod.CACHE_STORES[0][0] / "nlhrw"
    root.mkdir(parents=True, exist_ok=True)
    f = root / f"vol-{day:%Y%m%d}T1200.h5"
    f.write_bytes(b"x")
    return f


def _raw_daydir(mod, day: dt.date) -> pathlib.Path:
    d = mod.RAW_STORES[0][0] / f"{day:%Y/%m/%d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sweep.h5").write_bytes(b"x")
    return d


TODAY = dt.date(2026, 9, 7)


def test_cache_prune_is_age_based_and_ignores_archive_coverage(qa, monkeypatch):
    monkeypatch.setattr(qa, "day_coverage", lambda day: 0.0)   # nothing archived
    old = _cache_file(qa, TODAY - dt.timedelta(days=30))
    assert qa.prune_cache(TODAY) == 1
    assert not old.exists()


def test_cache_prune_keeps_files_inside_the_retention_window(qa, monkeypatch):
    monkeypatch.setattr(qa, "day_coverage", lambda day: 1.0)
    fresh = _cache_file(qa, TODAY - dt.timedelta(days=1))
    assert qa.prune_cache(TODAY) == 0
    assert fresh.exists()


def test_cache_prune_reaches_back_past_a_long_backlog(qa, monkeypatch):
    monkeypatch.setattr(qa, "day_coverage", lambda day: 0.0)
    files = [_cache_file(qa, TODAY - dt.timedelta(days=age)) for age in (10, 100, 300)]
    assert qa.prune_cache(TODAY) == 3
    assert not any(f.exists() for f in files)


def test_cache_keep_since_holds_back_a_window_queued_for_backprocessing(qa, monkeypatch):
    monkeypatch.setattr(qa, "day_coverage", lambda day: 0.0)
    qa.CACHE_KEEP_SINCE = dt.date(2026, 8, 16)
    kept = _cache_file(qa, dt.date(2026, 8, 20))
    gone = _cache_file(qa, dt.date(2026, 8, 10))
    assert qa.prune_cache(TODAY) == 1
    assert kept.exists()
    assert not gone.exists()


def test_raw_prune_still_requires_archive_coverage(qa, monkeypatch):
    """Raw volumes are unrecoverable, so their gate must NOT be relaxed."""
    monkeypatch.setattr(qa, "day_coverage", lambda day: 0.5)
    d = _raw_daydir(qa, TODAY - dt.timedelta(days=20))
    qa.prune_raw(TODAY)
    assert d.exists()
    monkeypatch.setattr(qa, "day_coverage", lambda day: 0.99)
    qa.prune_raw(TODAY)
    assert not d.exists()


def test_prune_only_never_archives(qa, monkeypatch):
    """The daily unit runs --prune-only: a slow backfill cannot starve the prune."""
    calls = []
    monkeypatch.setattr(qa, "archive", lambda *a, **k: calls.append(a) or 0)
    monkeypatch.setattr(qa, "day_coverage", lambda day: 1.0)
    old = _cache_file(qa, TODAY - dt.timedelta(days=30))
    assert qa.main(["--prune-only"]) == 0
    assert calls == []
    assert not old.exists()
