"""Render systemd unit + timer files from research/ops/schedule.yaml (TODO 1.8).

    python -m tools.gen_systemd render --out /tmp/units      # write pluvio-<name>.{service,timer}
    python -m tools.gen_systemd list                         # every job and its cadence
    python -m tools.gen_systemd diff --installed /etc/systemd/system   # what differs from disk

Only jobs with ``managed: true`` (the default) are rendered; jobs marked
``managed: false`` are pre-existing hand-written units listed so the manifest
is the complete schedule, and ``diff`` reports any pluvio-* timer on disk that
the manifest does not know about.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

UNIT_PREFIX = "pluvio-"


def load_manifest(path: str | pathlib.Path) -> dict:
    doc = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(doc, dict) or "jobs" not in doc:
        raise ValueError(f"{path}: expected a mapping with a 'jobs' list")
    names = [j["name"] for j in doc["jobs"]]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate job names: {sorted(dupes)}")
    for j in doc["jobs"]:
        if j.get("managed", True) and not j.get("exec"):
            raise ValueError(f"job {j['name']}: managed jobs need an 'exec'")
        if not j.get("on_calendar"):
            raise ValueError(f"job {j['name']}: needs 'on_calendar'")
    return doc


def render_job(job: dict, defaults: dict) -> tuple[str, str]:
    """(service_text, timer_text) for one managed job."""
    env = dict(defaults.get("environment") or {})
    env.update(job.get("environment") or {})
    python = defaults.get("python", "python3")
    exec_line = str(job["exec"]).replace("{python}", python).strip()
    log = job.get("log")
    if log:
        # append to the same log file cron used, so nothing downstream changes
        exec_line = f"/bin/sh -c {_sh_quote(exec_line + f' >> {log} 2>&1')}"
    mounts = job.get("requires_mounts", defaults.get("requires_mounts") or [])
    lines = ["[Unit]", f"Description=Pluvio {job.get('description', job['name'])}"]
    if mounts:
        lines.append("RequiresMountsFor=" + " ".join(mounts))
    lines += ["", "[Service]", "Type=oneshot",
              f"User={job.get('user', defaults.get('user', 'root'))}",
              f"WorkingDirectory={job.get('working_directory', defaults.get('working_directory', '/'))}"]
    for k, v in sorted(env.items()):
        lines.append(f"Environment={k}={v}")
    lines.append(f"ExecStart={exec_line}")
    if job.get("timeout_sec"):
        lines.append(f"TimeoutStartSec={int(job['timeout_sec'])}")
    service = "\n".join(lines) + "\n"
    timer = "\n".join([
        "[Unit]", f"Description=Pluvio {job.get('description', job['name'])} (timer)", "",
        "[Timer]", f"OnCalendar={job['on_calendar']}", "Persistent=true", "",
        "[Install]", "WantedBy=timers.target",
    ]) + "\n"
    return service, timer


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def render_all(doc: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for job in doc["jobs"]:
        if not job.get("managed", True):
            continue
        service, timer = render_job(job, doc.get("defaults") or {})
        out[f"{UNIT_PREFIX}{job['name']}.service"] = service
        out[f"{UNIT_PREFIX}{job['name']}.timer"] = timer
    return out


def diff_installed(doc: dict, installed_dir: pathlib.Path) -> dict:
    rendered = render_all(doc)
    known = {f"{UNIT_PREFIX}{j['name']}.timer" for j in doc["jobs"]}
    on_disk = {p.name for p in installed_dir.glob(f"{UNIT_PREFIX}*.timer")}
    changed = [n for n, text in rendered.items()
               if (installed_dir / n).exists() and (installed_dir / n).read_text() != text]
    missing = [n for n in rendered if not (installed_dir / n).exists()]
    return {"changed": sorted(changed), "missing": sorted(missing),
            "unknown_timers": sorted(on_disk - known)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(pathlib.Path(__file__).resolve().parents[1] / "ops" / "schedule.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("--out", required=True)
    sub.add_parser("list")
    d = sub.add_parser("diff"); d.add_argument("--installed", default="/etc/systemd/system")
    args = p.parse_args(argv)
    doc = load_manifest(args.manifest)
    if args.cmd == "list":
        for j in doc["jobs"]:
            flag = "" if j.get("managed", True) else "  (hand-written unit)"
            print(f"{UNIT_PREFIX}{j['name']:<22} {j['on_calendar']:<14} {j.get('description','')}{flag}")
        return 0
    if args.cmd == "render":
        out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
        for name, text in render_all(doc).items():
            (out / name).write_text(text)
        print(f"rendered {len(render_all(doc))} files to {out}")
        return 0
    res = diff_installed(doc, pathlib.Path(args.installed))
    for k, v in res.items():
        print(f"{k}: {v}")
    return 1 if any(res.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
