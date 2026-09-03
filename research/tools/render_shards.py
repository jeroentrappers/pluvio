"""One-time render of the training sample set to sharded float16 tensors.

    python -m tools.render_shards --zarr ../data/timeseries_v3.zarr \
        --out ../data/shards_v3 --split train,val --workers 8

Why
───
``ZarrCorrectionDataset`` rebuilds every sample from the zarr on every epoch
(one chunk read per history frame, one per aux channel, plus normalisation and
the time-encoding planes). At 192² with 113k train samples that assembly *is*
the epoch cost — measured ~47 min/epoch at batch 8 with 6 workers, GPU idle.
This script does the assembly once; ``model/shard_dataset.py`` then streams the
result with one memmap slice and one dtype cast per sample.

What it writes (under ``--out/<split>/``)
─────────────────────────────────────────
    x_00000.npy   (n, C, H, W) float16   — model inputs
    y_00000.npy   (n, 1, H, W) float16   — targets
    ...
    index.npy     structured (issue_epoch, lead_min, issue_idx, target_idx)
    manifest.json recipe + grid + sample count + source-store hash
                  + per-shard sha256, and ``"complete": true`` only when done

Semantics preserved exactly
───────────────────────────
Samples come from ``ZarrCorrectionDataset``'s own index, in its own order, with
the same ``issue_time_split`` boundary; the input and target arrays come from
its own ``build_input`` / ``build_target``. The only transformation is the cast
to the shard dtype, applied by ``shard_dataset.cast_for_shard`` — the same
function the tests apply to the zarr side, which is why the two agree
bit-for-bit. Shard boundaries never split an issue-time, so a shard is a whole
number of issues (matching the store's own per-issue chunking).

Resumability
────────────
Shards are written to ``*.tmp`` and atomically renamed, and the manifest is
rewritten after every completed shard. A rerun with the same arguments verifies
the recipe, keeps the shards already listed (checking file size, and sha256 with
``--verify``), and renders only what is missing. Interrupting mid-shard leaves a
``.tmp`` behind, which the rerun overwrites.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.shard_dataset import (  # noqa: E402
    INDEX_DTYPE,
    INDEX_NAME,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    SHARD_DTYPES,
    ShardDataset,
    ShardRecipeMismatch,
    cast_for_shard,
    compare_recipes,
    recipe_from_dataset,
    recipe_hash,
    sha256_file,
)
from model.zarr_dataset import (  # noqa: E402
    DEFAULT_LEADS,
    RADAR_HISTORY_STEPS,
    ZarrCorrectionDataset,
    issue_time_split,
)

LOG = logging.getLogger("pluvio.render_shards")

_DT_MIN = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DT_MAX = datetime(2100, 1, 1, tzinfo=timezone.utc)

SPLITS = ("train", "val", "all")
DEFAULT_SAMPLES_PER_SHARD = 512


# ─────────────────────────────────────────────────────── source store hash


def source_store_hash(zarr_path: str | pathlib.Path) -> dict[str, Any]:
    """Structural fingerprint of the source store.

    Deliberately NOT a content hash of every byte: the real store is ~14 GB and
    re-reading it would cost more than the render. It covers what a rebuild
    changes in practice — the array inventory (name, shape, dtype), the group
    attrs, and the full ``issue_time`` vector (so an appended, truncated or
    re-timed store fingerprints differently). A silent in-place value edit at
    identical shape/timestamps would slip past; note that in the manifest so
    nobody mistakes this for a full checksum."""
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    parts: list[str] = [json.dumps(dict(sorted(root.attrs.items())), sort_keys=True, default=str)]
    for name in sorted(root.array_keys()):
        arr = root[name]
        parts.append(f"{name}:{tuple(arr.shape)}:{arr.dtype!s}")
    issue = np.asarray(root["issue_time"][:], dtype="int64")
    import hashlib

    h = hashlib.sha256("|".join(parts).encode())
    h.update(issue.tobytes())
    return {
        "path": str(zarr_path),
        "hash": h.hexdigest(),
        "hash_mode": "structural",
        "hash_covers": "group attrs + array names/shapes/dtypes + full issue_time vector "
                       "(NOT array contents)",
        "n_issues": int(issue.size),
    }


# ───────────────────────────────────────────────────────── dataset plumbing


def _open_dataset(args: argparse.Namespace, split: str, boundary: datetime | None,
                  *, build_index: bool, aux_channels: list[str] | None = None):
    if split == "train":
        time_range = (_DT_MIN, boundary)
    elif split == "val":
        time_range = (boundary, _DT_MAX)
    else:
        time_range = None
    return ZarrCorrectionDataset(
        args.zarr,
        time_range=time_range,
        leads_min=tuple(args.leads),
        history_steps=args.history_steps,
        include_static=not args.no_static,
        aux_channels=aux_channels,
        # The dry filter belongs to the TRAIN index only, exactly as train.py
        # applies it (val is never filtered) — otherwise the rendered val set
        # would not be the one the loss curve was measured on.
        require_rain_fraction=args.require_rain_fraction if split == "train" else None,
        build_index=build_index,
        lagrangian_channels=args.lagrangian_channels,
    )


def _shard_plan(index: list, samples_per_shard: int) -> list[tuple[int, int]]:
    """(start, stop) sample ranges. A shard closes only on an issue boundary, so
    every issue's leads stay together in one shard."""
    plan: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(index) + 1):
        at_end = i == len(index)
        issue_boundary = at_end or index[i].issue_idx != index[i - 1].issue_idx
        if issue_boundary and (i - start) >= samples_per_shard:
            plan.append((start, i))
            start = i
    if start < len(index):
        plan.append((start, len(index)))
    return plan


# ───────────────────────────────────────────────────────────── shard render

_WORKER: dict[str, Any] = {}


def _worker_init(payload: dict[str, Any]) -> None:
    """Each worker process opens the store once, WITHOUT rebuilding the index
    (the parent already did that, and on the real store the index build reads
    every candidate target)."""
    args = argparse.Namespace(**payload["args"])
    _WORKER["ds"] = _open_dataset(
        args, "all", None, build_index=False, aux_channels=payload["aux_channels"]
    )
    _WORKER["dtype"] = payload["dtype"]


def _render_shard(job: dict[str, Any]) -> dict[str, Any]:
    ds = _WORKER["ds"]
    dtype = _WORKER["dtype"]
    out_dir = pathlib.Path(job["out_dir"])
    samples = job["samples"]
    n = len(samples)
    c, (h, w) = ds.n_channels, ds.grid_hw

    x_path = out_dir / job["x"]
    y_path = out_dir / job["y"]
    x_tmp, y_tmp = x_path.with_suffix(".npy.tmp"), y_path.with_suffix(".npy.tmp")
    xm = np.lib.format.open_memmap(x_tmp, mode="w+", dtype=np.dtype(dtype), shape=(n, c, h, w))
    ym = np.lib.format.open_memmap(y_tmp, mode="w+", dtype=np.dtype(dtype), shape=(n, 1, h, w))
    try:
        for i, s in enumerate(samples):
            xm[i] = cast_for_shard(
                ds.build_input(s["issue_idx"], s["lead_min"], tuple(s["history_idx"])), dtype
            )
            ym[i] = cast_for_shard(ds.build_target(s["target_idx"]), dtype)
        xm.flush()
        ym.flush()
    finally:
        del xm, ym
    os.replace(x_tmp, x_path)
    os.replace(y_tmp, y_path)
    return {
        "id": job["id"],
        "n_samples": n,
        "first_sample": job["first_sample"],
        "x": job["x"],
        "y": job["y"],
        "sha256_x": sha256_file(x_path),
        "sha256_y": sha256_file(y_path),
        "issue_epoch_first": int(samples[0]["issue_epoch"]),
        "issue_epoch_last": int(samples[-1]["issue_epoch"]),
    }


# ─────────────────────────────────────────────────────────────── manifest


def _write_manifest(out_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    tmp = out_dir / (MANIFEST_NAME + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, out_dir / MANIFEST_NAME)


def _load_existing(out_dir: pathlib.Path, recipe: dict[str, Any], *, force: bool
                   ) -> dict[int, dict[str, Any]]:
    """Shards already on disk that we can keep, keyed by shard id."""
    path = out_dir / MANIFEST_NAME
    if force or not path.exists():
        return {}
    with open(path) as fh:
        old = json.load(fh)
    if old.get("manifest_version") != MANIFEST_VERSION:
        LOG.warning("existing manifest has version %r — re-rendering from scratch",
                    old.get("manifest_version"))
        return {}
    diffs = compare_recipes(old.get("recipe", {}), recipe)
    if diffs:
        raise ShardRecipeMismatch(
            f"{out_dir}: existing shards were rendered with a different recipe "
            f"({len(diffs)} field(s)):\n  " + "\n  ".join(diffs)
            + "\nPass --force to discard them, or render into a fresh --out directory."
        )
    keep: dict[int, dict[str, Any]] = {}
    for shard in old.get("shards", []):
        ok = True
        for key in ("x", "y"):
            f = out_dir / shard[key]
            chans = recipe["n_channels"] if key == "x" else 1
            want_items = shard["n_samples"] * chans * recipe["grid_hw"][0] * recipe["grid_hw"][1]
            want_bytes = want_items * np.dtype(recipe["dtype"]).itemsize
            if not f.exists() or f.stat().st_size < want_bytes:
                ok = False
        if ok:
            keep[int(shard["id"])] = shard
    return keep


# ─────────────────────────────────────────────────────────────────── render


def render_split(args: argparse.Namespace, split: str, boundary: datetime | None) -> pathlib.Path:
    out_dir = pathlib.Path(args.out) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    ds = _open_dataset(args, split, boundary, build_index=True)
    index = ds.index
    if args.max_samples is not None:
        index = index[: args.max_samples]
    LOG.info("[%s] indexed %d samples in %.1fs (%d channels @ %s)",
             split, len(index), time.monotonic() - t0, ds.n_channels, ds.grid_hw)

    recipe = recipe_from_dataset(
        ds,
        split=split,
        dtype=args.dtype,
        split_boundary_epoch=int(boundary.timestamp()) if boundary else None,
        val_frac=args.val_frac,
        max_samples=args.max_samples,
    )
    keep = _load_existing(out_dir, recipe, force=args.force)

    plan = _shard_plan(index, args.samples_per_shard)
    jobs = []
    for sid, (start, stop) in enumerate(plan):
        if sid in keep and keep[sid]["n_samples"] == stop - start:
            continue
        jobs.append({
            "id": sid,
            "out_dir": str(out_dir),
            "first_sample": start,
            "x": f"x_{sid:05d}.npy",
            "y": f"y_{sid:05d}.npy",
            "samples": [
                {"issue_idx": s.issue_idx, "lead_min": s.lead_min,
                 "history_idx": list(s.history_idx), "target_idx": s.target_idx,
                 "issue_epoch": s.issue_epoch}
                for s in index[start:stop]
            ],
        })
    LOG.info("[%s] %d shard(s) planned, %d already on disk, %d to render",
             split, len(plan), len(plan) - len(jobs), len(jobs))

    # index.npy: per-sample metadata, same order as the samples themselves.
    meta = np.zeros(len(index), dtype=INDEX_DTYPE)
    for i, s in enumerate(index):
        meta[i] = (s.issue_epoch, s.lead_min, s.issue_idx, s.target_idx)
    np.save(out_dir / INDEX_NAME, meta)

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "complete": False,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "split": split,
        "n_samples": len(index),
        "grid_hw": recipe["grid_hw"],
        "n_channels": recipe["n_channels"],
        "dtype": args.dtype,
        "samples_per_shard": args.samples_per_shard,
        "channel_recipe": ds.channel_recipe(),
        "recipe": recipe,
        "recipe_hash": recipe_hash(recipe),
        "source_store": source_store_hash(args.zarr),
        "index": INDEX_NAME,
        "shards": [],
    }

    def _flush(done: dict[int, dict[str, Any]], *, complete: bool) -> None:
        manifest["shards"] = [done[sid] for sid in sorted(done)]
        manifest["complete"] = complete
        _write_manifest(out_dir, manifest)

    done = dict(keep)
    _flush(done, complete=False)

    started = time.monotonic()
    n_rendered = 0
    payload = {"args": _worker_args(args), "aux_channels": list(ds.aux_channels),
               "dtype": args.dtype}
    total = sum(len(j["samples"]) for j in jobs)
    if args.workers <= 1:
        _worker_init(payload)
        for job in jobs:
            done[job["id"]] = _render_shard(job)
            n_rendered += len(job["samples"])
            _flush(done, complete=False)
            _log_progress(split, n_rendered, total, started)
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                 initargs=(payload,)) as pool:
            for shard in pool.map(_render_shard, jobs):
                done[shard["id"]] = shard
                n_rendered += shard["n_samples"]
                _flush(done, complete=False)
                _log_progress(split, n_rendered, total, started)

    if sorted(done) != list(range(len(plan))):
        missing = sorted(set(range(len(plan))) - set(done))
        raise RuntimeError(f"{out_dir}: shards {missing} did not render")
    _flush(done, complete=True)
    LOG.info("[%s] complete → %s (%d samples, %d shards, %.1f min)",
             split, out_dir, len(index), len(plan), (time.monotonic() - started) / 60)
    return out_dir


def _worker_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "zarr": str(args.zarr),
        "leads": list(args.leads),
        "history_steps": args.history_steps,
        "no_static": args.no_static,
        "lagrangian_channels": args.lagrangian_channels,
        "require_rain_fraction": None,   # index-time only; workers never index
    }


def _log_progress(split: str, done: int, total: int, started: float) -> None:
    if not done:
        return
    rate = done / max(time.monotonic() - started, 1e-9)
    eta_min = (total - done) / rate / 60 if rate else float("nan")
    LOG.info("[%s] %d/%d samples (%.0f samples/s, ETA %.1f min)",
             split, done, total, rate, eta_min)


# ──────────────────────────────────────────────────────────────────── main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", type=pathlib.Path, required=True, help="source timeseries zarr store")
    p.add_argument("--out", type=pathlib.Path, required=True,
                   help="shard root; each split lands in <out>/<split>/")
    p.add_argument("--split", default="train,val",
                   help="train, val, all — or a comma-separated list (default train,val)")
    p.add_argument("--workers", type=int, default=1,
                   help="render processes (each opens the store; index is built once, here)")
    p.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD,
                   help=f"target samples per shard, rounded up to an issue boundary "
                        f"(default {DEFAULT_SAMPLES_PER_SHARD})")
    p.add_argument("--dtype", default="float16", choices=SHARD_DTYPES,
                   help="shard dtype; float16 halves the disk and matches CUDA autocast "
                        "precision, float32 is bit-exact with the zarr dataset")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="must match train.py --val-frac: sets the issue_time split boundary")
    p.add_argument("--leads", default=",".join(str(v) for v in DEFAULT_LEADS),
                   help="comma-separated forecast leads in minutes")
    p.add_argument("--history-steps", type=int, default=RADAR_HISTORY_STEPS)
    p.add_argument("--no-static", action="store_true", help="drop the static channels")
    p.add_argument("--lagrangian-channels", type=int, default=0, choices=(0, 1, 2),
                   help="Bake the Lagrangian-persistence planes (2.3) into the shards: 1 = "
                        "the latest analysis advected to each sample's lead, 2 = that plus "
                        "the per-step flow magnitude. Recorded in the recipe, so a train run "
                        "asking for a different count is refused. This is also where the "
                        "flow estimate stops being a per-epoch cost — it is computed once "
                        "per issue here.")
    p.add_argument("--require-rain-fraction", type=float, default=None,
                   help="train-split dry-sample filter, same meaning as train.py's flag")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap samples per split (smoke tests); recorded in the recipe")
    p.add_argument("--force", action="store_true", help="discard existing shards and re-render")
    p.add_argument("--verify", action="store_true",
                   help="after rendering, re-hash every shard against the manifest")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    splits = [s.strip() for s in str(args.split).split(",") if s.strip()]
    bad = [s for s in splits if s not in SPLITS]
    if bad or not splits:
        raise SystemExit(f"--split must be one or more of {SPLITS}, got {args.split!r}")
    if "all" in splits and len(splits) > 1:
        raise SystemExit("--split all renders the unsplit sample set; do not combine it "
                         "with train/val")
    if args.samples_per_shard < 1:
        raise SystemExit(f"--samples-per-shard must be >= 1, got {args.samples_per_shard}")
    if args.workers < 1:
        raise SystemExit(f"--workers must be >= 1, got {args.workers}")
    args.leads = tuple(int(v) for v in str(args.leads).split(",") if v.strip())
    if not args.leads:
        raise SystemExit("--leads must list at least one lead")

    boundary = None
    if {"train", "val"} & set(splits):
        boundary = issue_time_split(args.zarr, args.val_frac)
        LOG.info("issue_time split: train < %s <= val", boundary.isoformat())

    for split in splits:
        out_dir = render_split(args, split, boundary)
        if args.verify:
            ShardDataset(out_dir, verify_checksums=True)
            LOG.info("[%s] checksums verified", split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
