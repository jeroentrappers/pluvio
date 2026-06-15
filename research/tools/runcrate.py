"""Runcrate.ai GPU launch helper for Pluvio training.

Wraps the Runcrate Infrastructure API (https://runcrate.ai/api/v1) to run the
training on a persistent-volume pattern so the 12 GB store is uploaded ONCE and
reused across runs (the GPU only bills during training, not the upload):

    1. volume-create   make a persistent volume (mounts at /workspace)
    2. ssh-key         register your SSH public key
    3. populate        boot a cheap CPU instance with the volume; you scp the
                       bundle (zarr + model code) onto it, untar, then stop it
    4. train           boot an RTX-4090 with the volume + a launch script that
                       runs model/gpu_train.sh (train + evaluate → /workspace)
    5. status / stop   poll, then terminate (the volume + checkpoints persist)

Reads RUNCRATE_API_KEY / RUNCRATE_API_BASE from research/.env.

Usage:
    python -m tools.runcrate gpus
    python -m tools.runcrate volume-create --name pluvio-data --size 40 --region <id>
    python -m tools.runcrate ssh-key --name me --pub ~/.ssh/id_ed25519.pub
    python -m tools.runcrate populate --volume <vid> --ssh-key <kid>
    python -m tools.runcrate train --volume <vid> --ssh-key <kid> --gpu <type_id> --epochs 40
    python -m tools.runcrate status <instance_id>
    python -m tools.runcrate stop <instance_id>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = {}
    p = REPO_ROOT / ".env"
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    if "RUNCRATE_API_KEY" not in env:
        sys.exit("RUNCRATE_API_KEY not in .env")
    return env


def _client() -> tuple[httpx.Client, str]:
    env = _env()
    base = env.get("RUNCRATE_API_BASE", "https://runcrate.ai/api/v1")
    c = httpx.Client(headers={"Authorization": f"Bearer {env['RUNCRATE_API_KEY']}"},
                     timeout=60)
    return c, base


def _req(method: str, path: str, **body):
    c, base = _client()
    r = c.request(method, base + path, json=body or None)
    if r.status_code >= 300:
        sys.exit(f"{method} {path} → {r.status_code}: {r.text[:300]}")
    return r.json() if r.content and "json" in r.headers.get("content-type", "") else {}


# ───────────────────────────────────────────────────────────── commands

def cmd_gpus(args):
    data = _req("GET", "/instances/types").get("data", [])
    def rows(pred):
        return sorted([d for d in data if pred(d)], key=lambda d: d.get("hourly_rate") or 9e9)
    print("── single RTX-4090 ──")
    for d in rows(lambda d: "4090" in str(d.get("gpu_type", "")).lower() and d.get("gpu_count") == 1):
        print(f"  {d['id']:42s} {d.get('region',''):22s} {d.get('memory_gb')}GB  ${d.get('hourly_rate')}/hr")
    print("── cheapest CPU (for populate) ──")
    for d in rows(lambda d: d.get("gpu_type") == "CPU")[:3]:
        print(f"  {d['id']:42s} {d.get('region',''):22s} ${d.get('hourly_rate')}/hr")


def cmd_volume_create(args):
    out = _req("POST", "/storage", name=args.name, size_gb=args.size, region=args.region)
    print(json.dumps(out, indent=2))


def cmd_ssh_key(args):
    pub = pathlib.Path(args.pub).expanduser().read_text().strip()
    out = _req("POST", "/ssh-keys", name=args.name, public_key=pub)
    print(json.dumps(out, indent=2))


def _create_instance(**body):
    out = _req("POST", "/instances", **body)
    print(json.dumps(out, indent=2))
    return out


def cmd_populate(args):
    inst = _create_instance(
        name="pluvio-populate", instance_type_id=args.type,
        storage_id=args.volume, ssh_key_id=args.ssh_key)
    iid = inst.get("id") or inst.get("data", {}).get("id")
    print(f"\nCPU instance {iid} booting with the volume at /workspace.")
    print("Once it shows running (python -m tools.runcrate status <id>), note its SSH")
    print("host from the status output, then from THIS server upload the bundle:\n")
    print("  tar -C research/data -cf - timeseries.zarr \\")
    print("      -C ../ model notebooks/_lib.py | \\")
    print("      ssh <user>@<host> 'tar -xf - -C /workspace'\n")
    print(f"Then stop it: python -m tools.runcrate stop {iid}")


def cmd_train(args):
    launch = (
        "cd /workspace && PYTHONPATH=/workspace "
        f"EPOCHS={args.epochs} BASE_CHANNELS={args.base_channels} "
        f"BATCH_SIZE={args.batch_size} RAIN_FRAC={args.rain_frac} "
        "bash model/gpu_train.sh > /workspace/launch.log 2>&1"
    )
    inst = _create_instance(
        name="pluvio-train", instance_type_id=args.gpu,
        storage_id=args.volume, ssh_key_id=args.ssh_key,
        launch_script=launch)
    iid = inst.get("id") or inst.get("data", {}).get("id")
    print(f"\n4090 instance {iid} launching training. It writes to the volume:")
    print("  /workspace/checkpoints/{pluvio_unet.pt, train.log, eval_report.txt, STATUS}")
    print(f"Poll: python -m tools.runcrate status {iid}")
    print(f"When STATUS=DONE, pull the checkpoint, then: python -m tools.runcrate stop {iid}")


def cmd_status(args):
    print(json.dumps(_req("GET", f"/instances/{args.id}"), indent=2))
    print(json.dumps(_req("GET", f"/instances/{args.id}/status"), indent=2))


def cmd_stop(args):
    _req("DELETE", f"/instances/{args.id}")
    print(f"stopped/terminated {args.id}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus").set_defaults(fn=cmd_gpus)
    s = sub.add_parser("volume-create"); s.set_defaults(fn=cmd_volume_create)
    s.add_argument("--name", required=True); s.add_argument("--size", type=int, default=40)
    s.add_argument("--region", required=True)
    s = sub.add_parser("ssh-key"); s.set_defaults(fn=cmd_ssh_key)
    s.add_argument("--name", required=True); s.add_argument("--pub", required=True)
    s = sub.add_parser("populate"); s.set_defaults(fn=cmd_populate)
    s.add_argument("--volume", required=True); s.add_argument("--ssh-key", required=True)
    s.add_argument("--type", required=True, help="cheap CPU instance_type_id (see `gpus`)")
    s = sub.add_parser("train"); s.set_defaults(fn=cmd_train)
    s.add_argument("--volume", required=True); s.add_argument("--ssh-key", required=True)
    s.add_argument("--gpu", required=True, help="RTX-4090 instance_type_id (see `gpus`)")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--base-channels", type=int, default=32)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--rain-frac", type=float, default=0.02)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status); s.add_argument("id")
    s = sub.add_parser("stop"); s.set_defaults(fn=cmd_stop); s.add_argument("id")
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
