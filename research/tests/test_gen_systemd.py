"""The ops manifest renders valid units and the diff spots drift."""

from __future__ import annotations

import pathlib

import pytest

from tools import gen_systemd as gs

MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "ops" / "schedule.yaml"


def test_manifest_loads_and_covers_every_cron_job_and_timer():
    doc = gs.load_manifest(MANIFEST)
    names = {j["name"] for j in doc["jobs"]}
    for n in ("collect-knmi-radar", "collect-kmi-aws", "collect-knmi-aws", "collect-meteosat",
              "collect-alaro", "collect-sst", "collect-netatmo", "append-infer", "rotate-to-nas",
              "observed", "qc", "qc-inputs", "qpe-archive", "qpe-prune", "wide-archive",
              "forecast-archive", "external-baselines", "buienradar-eu", "scoreboard"):
        assert n in names, n


def test_render_produces_service_and_timer_with_log_redirect_and_env():
    doc = gs.load_manifest(MANIFEST)
    files = gs.render_all(doc)
    svc = files["pluvio-append-infer.service"]
    assert "User=ansible" in svc and "Environment=PYTHONPATH=/opt/pluvio/research" in svc
    assert "/opt/pluvio/research/.venv/bin/python -m tools.build_zarr --append" in svc
    assert ">> /opt/pluvio/logs/serve.log 2>&1" in svc
    assert "RequiresMountsFor" not in svc            # local store + serve dir: runs even if the box is down
    assert "RequiresMountsFor=/mnt/storagebox" in files["pluvio-rotate-to-nas.service"]
    assert svc.count("/bin/sh -c") == 1              # one shell wrapper, not nested
    tmr = files["pluvio-append-infer.timer"]
    assert "OnCalendar=*:0/5" in tmr and "Persistent=true" in tmr
    # hand-written units are not rendered
    assert "pluvio-scoreboard.service" not in files


def test_diff_reports_missing_changed_and_unknown(tmp_path):
    doc = gs.load_manifest(MANIFEST)
    files = gs.render_all(doc)
    for n, t in files.items():
        (tmp_path / n).write_text(t)
    (tmp_path / "pluvio-collect-sst.timer").write_text("[Timer]\nOnCalendar=07:00\n")
    (tmp_path / "pluvio-append-infer.service").unlink()
    (tmp_path / "pluvio-mystery.timer").write_text("[Timer]\n")
    res = gs.diff_installed(doc, tmp_path)
    assert res["changed"] == ["pluvio-collect-sst.timer"]
    assert res["missing"] == ["pluvio-append-infer.service"]
    assert res["unknown_timers"] == ["pluvio-mystery.timer"]


def test_manifest_validation(tmp_path):
    bad = tmp_path / "m.yaml"
    bad.write_text("jobs:\n  - name: a\n    on_calendar: '*:0/5'\n  - name: a\n    on_calendar: '*:0/5'\n    exec: x\n")
    with pytest.raises(ValueError, match="duplicate"):
        gs.load_manifest(bad)
    bad.write_text("jobs:\n  - name: b\n    on_calendar: '*:0/5'\n")
    with pytest.raises(ValueError, match="exec"):
        gs.load_manifest(bad)
