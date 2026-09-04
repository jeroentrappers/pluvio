"""Canary tick for a candidate checkpoint (TODO 3.5, ops half).

Every 5 min (systemd `pluvio-canary`): if `<candidate>` exists, run
`model.infer_latest` with it to `<canary_dir>/candidate/<issue>Z.npz` and copy
the SERVED nowcast (the incumbent's output for the same issue) to
`<canary_dir>/incumbent/<issue>Z.npz`. After an hour, `promote_checkpoint
canary --incumbent-dir … --candidate-dir …` has its paired ticks; nothing here
touches what is served. Exit 0 with no candidate (the timer is always armed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import shutil
import subprocess
import sys

import numpy as np

LOG = logging.getLogger("pluvio.canary")


def issue_stamp(npz_path: pathlib.Path) -> str:
    with np.load(npz_path, allow_pickle=True) as d:
        epoch = int(d["issue_epoch"])
    return dt.datetime.fromtimestamp(epoch, dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def run_candidate(python: str, zarr: str, candidate: pathlib.Path, out: pathlib.Path,
                  cwd: pathlib.Path) -> None:
    cmd = [python, "-m", "model.infer_latest", "--zarr", zarr, "--checkpoint", str(candidate),
           "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=str(cwd), timeout=600)


def tick(*, candidate: pathlib.Path, served: pathlib.Path, canary_dir: pathlib.Path,
         zarr: str, python: str, cwd: pathlib.Path, runner=run_candidate) -> dict:
    if not candidate.exists():
        return {"ran": False, "reason": "no candidate checkpoint"}
    if not served.exists():
        return {"ran": False, "reason": "no served nowcast to pair with"}
    stamp = issue_stamp(served)
    inc_dir, cand_dir = canary_dir / "incumbent", canary_dir / "candidate"
    inc_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_out = cand_dir / f"{stamp}.npz"
    inc_out = inc_dir / f"{stamp}.npz"
    if cand_out.exists() and inc_out.exists():
        return {"ran": False, "reason": f"issue {stamp} already paired"}
    tmp = cand_dir / f".{stamp}.partial.npz"   # np.savez appends .npz to other suffixes
    runner(python, zarr, candidate, tmp, cwd)
    if issue_stamp(tmp) != stamp:
        tmp.unlink(missing_ok=True)
        return {"ran": False, "reason": "candidate resolved a different issue than the served npz — retry next tick"}
    tmp.replace(cand_out)
    shutil.copy2(served, inc_out)
    return {"ran": True, "issue": stamp, "pairs": len(list(cand_dir.glob("*.npz")))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", default="/opt/pluvio/research/checkpoints/candidate.pt")
    p.add_argument("--served", default="/opt/pluvio/serve/model_nowcast.npz")
    p.add_argument("--canary-dir", default="/opt/pluvio/serve/canary")
    p.add_argument("--zarr", default="/opt/pluvio/zarr/timeseries.zarr")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--cwd", default="/opt/pluvio/research")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    res = tick(candidate=pathlib.Path(args.candidate), served=pathlib.Path(args.served),
               canary_dir=pathlib.Path(args.canary_dir), zarr=args.zarr, python=args.python,
               cwd=pathlib.Path(args.cwd))
    LOG.info("canary %s", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
