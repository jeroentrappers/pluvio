"""Watch the storage-box mount and recover it automatically (2026-09-06).

Why: on 2026-09-06 the Hetzner storage box stopped answering (port 445 dead,
neither IPv4 nor IPv6). CIFS hung rather than failing, so every worker that
touched `/mnt/storagebox` blocked in `cifs_wait_for_server_reconnect`; the
observed-composite producer was killed on its 900 s timeout every run and the
served radar composite silently stopped advancing. Nothing noticed for ~40
minutes. Worse, a manual `umount -l` turned the mountpoint into a plain local
directory and collectors wrote 2.3 GB onto the root disk before it was caught.

So this watchdog does four things, in this order:

  probe    is the box's SMB port reachable, is the mount active, and can we
           actually stat/read/write through it inside a timeout? (a hung CIFS
           mount still reports "active", so liveness needs a real I/O probe
           with its own timeout — that is the whole lesson of the incident)
  protect  while the box is away the mountpoint stays EMPTY and immutable
           (`chattr +i`), so a job that runs anyway fails loudly instead of
           filling the root filesystem with a shadow copy
  recover  when the port answers again: drop the immutable bit, start the
           mount unit, verify with the same I/O probe
  restore  once healthy, re-arm the timers that were stopped for the outage
           (they are named in --restore-units) and clear the alert file

State goes to a JSON file the QC report picks up, so an outage shows up in
`qc_inputs`'s verdict rather than only in journald.

    python -m tools.storagebox_watchdog            # probe, protect, recover
    python -m tools.storagebox_watchdog --probe-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys

LOG = logging.getLogger("pluvio.storagebox_watchdog")

DEFAULT_HOST = "u614373-sub1.your-storagebox.de"
DEFAULT_MOUNT = pathlib.Path("/mnt/storagebox")
DEFAULT_UNIT = "mnt-storagebox.mount"
DEFAULT_STATE = pathlib.Path("/opt/pluvio/serve/storagebox_watchdog.json")
# Timers stopped during the 2026-09-06 outage; re-armed once the box is back.
DEFAULT_RESTORE_UNITS = (
    "pluvio-observed.timer", "pluvio-qpe-archive.timer", "pluvio-wide-archive.timer",
    "pluvio-forecast-archive.timer", "pluvio-external-baselines.timer",
    "pluvio-buienradar-eu.timer", "pluvio-rotate-to-nas.timer", "pluvio-scoreboard.timer",
    "pluvio-qpe-prune.timer", "aifs-forward.timer", "dwd-sweeps.timer", "era5-forward.timer",
    "icon-d2-forward.timer", "knmi-rtcor-forward.timer", "mtg-l2-forward.timer",
)


def port_open(host: str, port: int = 445, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def mount_active(unit: str, runner=subprocess.run) -> bool:
    r = runner(["systemctl", "is-active", "--quiet", unit], capture_output=True)
    return r.returncode == 0


def io_probe(mount: pathlib.Path, timeout: float = 20.0, runner=subprocess.run) -> tuple[bool, str]:
    """Read AND write through the mount, under an external timeout.

    Run in a subprocess with `timeout(1)`: a hung CIFS mount blocks in the
    kernel, where a Python-level timeout cannot help — the call itself must be
    killable. A write probe matters as much as a read: the box has come back
    read-only before.
    """
    probe = mount / ".watchdog_probe"
    script = (
        f"ls -1 {mount} >/dev/null 2>&1 || exit 3; "
        f"printf ok > {probe} 2>/dev/null || exit 4; "
        f"grep -q ok {probe} 2>/dev/null || exit 5; "
        f"rm -f {probe} 2>/dev/null || exit 6"
    )
    r = runner(["timeout", str(int(timeout)), "bash", "-c", script], capture_output=True)
    codes = {0: "ok", 124: "timed out (hung mount)", 3: "cannot list", 4: "cannot write",
             5: "write did not read back", 6: "cannot remove probe file"}
    return r.returncode == 0, codes.get(r.returncode, f"exit {r.returncode}")


def protect(mount: pathlib.Path, runner=subprocess.run) -> str:
    """Keep the detached mountpoint empty and immutable so nothing shadows it."""
    entries = sorted(p.name for p in mount.iterdir()) if mount.is_dir() else []
    if entries:
        return f"NOT protected: {len(entries)} local entr(y|ies) at the mountpoint ({entries[:3]})"
    runner(["chattr", "+i", str(mount)], capture_output=True)
    return "mountpoint empty and immutable"


def unprotect(mount: pathlib.Path, runner=subprocess.run) -> None:
    runner(["chattr", "-i", str(mount)], capture_output=True)


def recover(unit: str, mount: pathlib.Path, runner=subprocess.run) -> tuple[bool, str]:
    unprotect(mount, runner)
    r = runner(["systemctl", "start", unit], capture_output=True)
    if r.returncode != 0:
        return False, f"{unit} failed to start (exit {r.returncode})"
    ok, detail = io_probe(mount, runner=runner)
    return ok, f"mounted, probe: {detail}"


def restore_units(units, runner=subprocess.run) -> list[str]:
    started = []
    for u in units:
        if not mount_active(u, runner):
            r = runner(["systemctl", "start", u], capture_output=True)
            if r.returncode == 0:
                started.append(u)
    return started


def check(*, host: str, mount: pathlib.Path, unit: str, units_to_restore,
         act: bool = True, runner=subprocess.run) -> dict:
    """One pass. Returns the state dict that is written to the state file."""
    state: dict = {"checked_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                   "host": host, "mount": str(mount), "unit": unit}
    state["port_open"] = port_open(host)
    state["mount_active"] = mount_active(unit, runner)
    if state["mount_active"]:
        ok, detail = io_probe(mount, runner=runner)
        state["io_ok"], state["io_detail"] = ok, detail
    else:
        state["io_ok"], state["io_detail"] = False, "mount unit inactive"

    actions: list[str] = []
    if state["io_ok"]:
        state["status"] = "ok"
        if act:
            started = restore_units(units_to_restore, runner)
            if started:
                actions.append(f"re-armed {len(started)} timer(s): {', '.join(started)}")
    elif not state["port_open"]:
        # The box itself is unreachable: do not thrash the mount, just make
        # sure nothing can write a shadow copy onto the root disk.
        state["status"] = "box_unreachable"
        if act:
            actions.append(protect(mount, runner))
    else:
        # Port answers but I/O does not: a stale/hung mount worth recovering.
        state["status"] = "mount_broken"
        if act:
            ok, detail = recover(unit, mount, runner)
            actions.append(f"recovery {'succeeded' if ok else 'failed'}: {detail}")
            state["io_ok"] = ok
            if ok:
                state["status"] = "recovered"
                started = restore_units(units_to_restore, runner)
                if started:
                    actions.append(f"re-armed {len(started)} timer(s)")
    state["actions"] = actions
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("PLUVIO_STORAGEBOX_HOST", DEFAULT_HOST))
    p.add_argument("--mount", default=str(DEFAULT_MOUNT))
    p.add_argument("--unit", default=DEFAULT_UNIT)
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--restore-units", default=",".join(DEFAULT_RESTORE_UNITS))
    p.add_argument("--probe-only", action="store_true",
                   help="report, change nothing (no protect, no recover, no timer restore)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    state = check(host=args.host, mount=pathlib.Path(args.mount), unit=args.unit,
                  units_to_restore=[u for u in args.restore_units.split(",") if u],
                  act=not args.probe_only)
    sp = pathlib.Path(args.state)
    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_name(sp.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(sp)
    for a in state["actions"]:
        LOG.warning("%s", a)
    LOG.info("storagebox %s (port_open=%s mount_active=%s io=%s)", state["status"],
             state["port_open"], state["mount_active"], state["io_detail"])
    return 0 if state["status"] in ("ok", "recovered") else 1


if __name__ == "__main__":
    sys.exit(main())
