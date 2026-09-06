"""The storage-box watchdog: probe, protect, recover, restore (2026-09-06)."""

from __future__ import annotations

import json
import pathlib
import subprocess

from tools import storagebox_watchdog as wd


class FakeRunner:
    """Records systemctl/chattr/timeout calls and replies from a script."""

    def __init__(self, replies=None):
        self.calls: list[list[str]] = []
        self.replies = replies or {}

    def __call__(self, cmd, capture_output=False, **kw):
        self.calls.append(list(cmd))
        key = " ".join(cmd[:3])
        code = self.replies.get(key, 0)
        if cmd[0] == "timeout":
            code = self.replies.get("io", 0)
        elif cmd[:2] == ["systemctl", "is-active"]:
            code = self.replies.get(f"is-active {cmd[-1]}", 0)
        elif cmd[:2] == ["systemctl", "start"]:
            code = self.replies.get(f"start {cmd[-1]}", 0)
        return subprocess.CompletedProcess(cmd, code)

    def ran(self, *needles) -> bool:
        return any(all(n in " ".join(c) for n in needles) for c in self.calls)


def test_io_probe_reports_a_hung_mount_as_a_timeout(tmp_path):
    r = FakeRunner({"io": 124})
    ok, detail = wd.io_probe(tmp_path, runner=r)
    assert not ok and "hung" in detail
    assert r.ran("timeout", "bash")            # killable subprocess, not an in-process call


def test_io_probe_distinguishes_read_only_from_dead(tmp_path):
    assert wd.io_probe(tmp_path, runner=FakeRunner({"io": 4}))[1] == "cannot write"
    assert wd.io_probe(tmp_path, runner=FakeRunner({"io": 3}))[1] == "cannot list"
    assert wd.io_probe(tmp_path, runner=FakeRunner({"io": 0})) == (True, "ok")


def test_unreachable_box_protects_the_mountpoint_and_does_not_touch_the_mount(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "port_open", lambda *a, **k: False)
    r = FakeRunner({"is-active mnt-storagebox.mount": 1})
    state = wd.check(host="h", mount=tmp_path, unit="mnt-storagebox.mount",
                     units_to_restore=["pluvio-observed.timer"], runner=r)
    assert state["status"] == "box_unreachable"
    assert r.ran("chattr", "+i")
    assert not r.ran("systemctl", "start", "mnt-storagebox.mount")
    assert not r.ran("systemctl", "start", "pluvio-observed.timer")


def test_protect_refuses_when_local_files_shadow_the_mountpoint(tmp_path):
    (tmp_path / "radar_volumes").mkdir()
    r = FakeRunner()
    msg = wd.protect(tmp_path, runner=r)
    assert "NOT protected" in msg and "radar_volumes" in msg
    assert not r.ran("chattr", "+i")           # never hide a shadow copy behind +i


def test_hung_mount_with_a_reachable_box_is_recovered_and_timers_re_armed(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "port_open", lambda *a, **k: True)
    seq = iter([124, 0])                        # probe fails, then succeeds after remount
    r = FakeRunner()
    r.replies["io"] = 124

    def io(cmd, capture_output=False, **kw):
        r.calls.append(list(cmd))
        if cmd[0] == "timeout":
            return subprocess.CompletedProcess(cmd, next(seq))
        if cmd[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(cmd, 0 if "mount" in cmd[-1] else 1)
        return subprocess.CompletedProcess(cmd, 0)

    state = wd.check(host="h", mount=tmp_path, unit="mnt-storagebox.mount",
                     units_to_restore=["pluvio-observed.timer"], runner=io)
    assert state["status"] == "recovered" and state["io_ok"]
    assert any("chattr -i" in " ".join(c) for c in r.calls)
    assert any(c[:3] == ["systemctl", "start", "mnt-storagebox.mount"] for c in r.calls)
    assert any(c[:3] == ["systemctl", "start", "pluvio-observed.timer"] for c in r.calls)


def test_healthy_box_re_arms_stopped_timers_only(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "port_open", lambda *a, **k: True)
    r = FakeRunner({"is-active pluvio-observed.timer": 1, "is-active pluvio-qc.timer": 0})
    state = wd.check(host="h", mount=tmp_path, unit="mnt-storagebox.mount",
                     units_to_restore=["pluvio-observed.timer", "pluvio-qc.timer"], runner=r)
    assert state["status"] == "ok"
    assert r.ran("systemctl", "start", "pluvio-observed.timer")
    assert not r.ran("systemctl", "start", "pluvio-qc.timer")   # already armed


def test_probe_only_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "port_open", lambda *a, **k: False)
    r = FakeRunner({"is-active mnt-storagebox.mount": 1})
    state = wd.check(host="h", mount=tmp_path, unit="mnt-storagebox.mount",
                     units_to_restore=["pluvio-observed.timer"], act=False, runner=r)
    assert state["status"] == "box_unreachable" and state["actions"] == []
    assert not r.ran("chattr")


def test_cli_writes_the_state_file_and_exits_nonzero_when_broken(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(wd, "mount_active", lambda *a, **k: False)
    state_path = tmp_path / "state.json"
    rc = wd.main(["--host", "h", "--mount", str(tmp_path), "--state", str(state_path),
                  "--restore-units", "", "--probe-only"])
    assert rc == 1
    body = json.loads(state_path.read_text())
    assert body["status"] == "box_unreachable" and body["host"] == "h"
    assert pathlib.Path(body["mount"]) == tmp_path
