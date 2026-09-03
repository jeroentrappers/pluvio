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
Default (``--layout dedup``) — 29 of the 33 channels are a function of the
ISSUE, not of the lead, so they are stored once per issue instead of once per
sample (2.39 → 0.86 MiB/sample; 332 → 119 GiB for the full v3 sample set,
which is what makes it fit on a 194 G disk):

    inv_00000.npy (n_issues, C_inv, H, W) float16  — lead-invariant channels
    x_00000.npy   (n, C_var, H, W) float16         — lead-dependent planes
    y_00000.npy   (n, 1, H, W) float16             — targets
    ...
    index.npy     structured (issue_epoch, lead_min, issue_idx, target_idx)
    manifest.json layout + recipe + grid + sample count + source-store hash
                  + per-shard sha256, and ``"complete": true`` only when done

With ``--layout flat``, ``x_*.npy`` holds the whole ``(n, C, H, W)`` input and
there is no ``inv_*.npy``. ``ShardDataset`` reassembles a dedup sample in
``build_input`` channel order and hands out the identical tensor either way.

Which channels are lead-invariant comes from
``zarr_dataset.lead_varying_channel_indices`` over the dataset's own
``channel_names`` — and this renderer VERIFIES it: for every issue it builds
the full input at each of the issue's leads and refuses to write if a channel
it was about to store once per issue is not in fact identical across them.

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
the recipe **and the source store's fingerprint**, keeps the shards already
listed (checking file size, and sha256 with ``--verify``), and renders only
what is missing. Interrupting mid-shard leaves a ``.tmp`` behind, which the
rerun overwrites.

The source fingerprint is the load-bearing half of that check. The recipe says
what a sample *means*; it cannot see a store rebuilt in place with the same
arrays, shapes, attrs and ``issue_time`` but different values. Resuming into
that renders the missing shards from the new numbers and leaves a directory
that is half one store and half another, with no error anywhere. So
``source_store_hash`` covers sampled CONTENT as well as structure, and a resume
against a differently-fingerprinting store refuses and points at ``--force``.
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
    DEFAULT_LAYOUT,
    INDEX_DTYPE,
    INDEX_NAME,
    LAYOUT_DEDUP,
    LAYOUTS,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    SHARD_DTYPES,
    ShardDataset,
    ShardRecipeMismatch,
    cast_for_shard,
    compare_recipes,
    compare_source_store,
    recipe_from_dataset,
    recipe_hash,
    sha256_file,
    shard_file_keys,
    source_store_hash,
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


def _issue_groups(samples: list[dict[str, Any]]) -> list[list[int]]:
    """Positions within one shard grouped by issue, in order. The indexer emits
    an issue's leads consecutively and ``_shard_plan`` never splits an issue, so
    every group is a contiguous run — which is what lets ``ShardDataset``
    recover the mapping from ``index.npy`` alone."""
    groups: list[list[int]] = []
    for i, s in enumerate(samples):
        if i and s["issue_idx"] == samples[i - 1]["issue_idx"]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


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
    _WORKER["layout"] = payload["layout"]
    _WORKER["var_idx"] = np.asarray(payload["lead_varying_channels"], dtype="int64")
    _WORKER["inv_idx"] = np.asarray(
        [c for c in range(_WORKER["ds"].n_channels)
         if c not in set(payload["lead_varying_channels"])], dtype="int64")
    _WORKER["channel_names"] = payload["channel_names"]


def _render_shard(job: dict[str, Any]) -> dict[str, Any]:
    ds = _WORKER["ds"]
    dtype = _WORKER["dtype"]
    dedup = _WORKER["layout"] == LAYOUT_DEDUP
    inv_idx, var_idx = _WORKER["inv_idx"], _WORKER["var_idx"]
    out_dir = pathlib.Path(job["out_dir"])
    samples = job["samples"]
    n = len(samples)
    c, (h, w) = ds.n_channels, ds.grid_hw
    groups = _issue_groups(samples)

    paths = {k: out_dir / job[k] for k in shard_file_keys(_WORKER["layout"])}
    tmps = {k: v.with_suffix(".npy.tmp") for k, v in paths.items()}
    maps = {
        "x": np.lib.format.open_memmap(
            tmps["x"], mode="w+", dtype=np.dtype(dtype),
            shape=(n, len(var_idx) if dedup else c, h, w)),
        "y": np.lib.format.open_memmap(
            tmps["y"], mode="w+", dtype=np.dtype(dtype), shape=(n, 1, h, w)),
    }
    if dedup:
        maps["inv"] = np.lib.format.open_memmap(
            tmps["inv"], mode="w+", dtype=np.dtype(dtype),
            shape=(len(groups), len(inv_idx), h, w))
    try:
        for row, group in enumerate(groups):
            first_inv: np.ndarray | None = None
            for i in group:
                s = samples[i]
                x = cast_for_shard(
                    ds.build_input(s["issue_idx"], s["lead_min"], tuple(s["history_idx"])), dtype
                )
                if not dedup:
                    maps["x"][i] = x
                else:
                    maps["x"][i] = x[var_idx]
                    # Verify the lead-invariance we are about to rely on, per
                    # issue, instead of trusting the channel-name list: the
                    # block is written from the issue's first lead and every
                    # other lead must agree with it plane for plane.
                    inv = x[inv_idx]
                    if first_inv is None:
                        first_inv = inv
                        maps["inv"][row] = inv
                    elif not np.array_equal(inv, first_inv):
                        bad = [int(k) for k in np.flatnonzero(
                            [not np.array_equal(inv[j], first_inv[j])
                             for j in range(len(inv_idx))])]
                        names = _WORKER["channel_names"]
                        raise RuntimeError(
                            f"issue {s['issue_idx']} lead {s['lead_min']}: channel(s) "
                            + ", ".join(f"{int(inv_idx[k])} ({names[int(inv_idx[k])]})"
                                        for k in bad)
                            + " differ from the issue's first lead but are stored ONCE per "
                            "issue by --layout dedup. build_input's channel semantics and "
                            "zarr_dataset.LEAD_VARYING_CHANNEL_NAMES have drifted apart; "
                            "fix that set, or render with --layout flat."
                        )
                maps["y"][i] = cast_for_shard(ds.build_target(s["target_idx"]), dtype)
        for m in maps.values():
            m.flush()
    finally:
        maps.clear()
    for key, path in paths.items():
        os.replace(tmps[key], path)
    out = {
        "id": job["id"],
        "n_samples": n,
        "n_issues": len(groups),
        "first_sample": job["first_sample"],
        "issue_epoch_first": int(samples[0]["issue_epoch"]),
        "issue_epoch_last": int(samples[-1]["issue_epoch"]),
    }
    for key, path in paths.items():
        out[key] = job[key]
        out[f"sha256_{key}"] = sha256_file(path)
    return out


# ─────────────────────────────────────────────────────────────── manifest


def _write_manifest(out_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    tmp = out_dir / (MANIFEST_NAME + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, out_dir / MANIFEST_NAME)


def _load_existing(out_dir: pathlib.Path, recipe: dict[str, Any], *, force: bool,
                   layout: str, n_inv: int, n_var: int, samples_per_shard: int,
                   source_store: dict[str, Any] | None = None,
                   ) -> dict[int, dict[str, Any]]:
    """Shards already on disk that we can keep, keyed by shard id.

    Four ways a resume must refuse rather than fill in the gaps: a different
    recipe, a different LAYOUT, a different ``--samples-per-shard``, and — the
    one this missed until review — a source store whose fingerprint has moved.

    The recipe cannot see either of the last two. An in-place value rebuild
    leaves shapes, attrs and ``issue_time`` identical, so without the source
    check a resume renders the missing shards from the new numbers and the
    loader accepts the mixture. And ``samples_per_shard`` is a shard-BOUNDARY
    choice, not a sample-semantics one, so it is deliberately not in
    RECIPE_KEYS — but changing it re-cuts the plan, and a kept shard whose
    sample count happens to match the new range at a different offset holds
    the wrong samples. Measured: a partial render at ``--samples-per-shard 2``
    resumed at 3 kept shard 3 (``first_sample`` 6) at offset 12, i.e. 3 of 81
    samples silently wrong, manifest ``complete``, ``--verify`` clean."""
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
    old_sps = old.get("samples_per_shard")
    if old_sps is not None and int(old_sps) != int(samples_per_shard):
        raise ShardRecipeMismatch(
            f"{out_dir}: existing shards were cut at --samples-per-shard {int(old_sps)}, "
            f"this run asks for {int(samples_per_shard)}. That re-cuts every shard "
            "boundary, so a kept shard would sit at the wrong offset in the new plan — "
            "pass --force to re-render, or render into a fresh --out directory."
        )
    old_layout = str(old.get("layout", "flat"))
    if old_layout != layout:
        raise ShardRecipeMismatch(
            f"{out_dir}: existing shards use layout {old_layout!r}, this run asks for "
            f"{layout!r}. The two store different files per shard and cannot be mixed — "
            "pass --force to re-render, or render into a fresh --out directory."
        )
    compare_source_store(
        old.get("source_store"), source_store, what=str(out_dir),
        remedy="Pass --force to discard the existing shards and re-render them all from "
               "the current store, or render into a fresh --out directory. Resuming would "
               "leave this directory half one store and half the other.",
    )
    keep: dict[int, dict[str, Any]] = {}
    chans = {"x": n_var if layout == LAYOUT_DEDUP else recipe["n_channels"],
             "y": 1, "inv": n_inv}
    for shard in old.get("shards", []):
        ok = True
        for key in shard_file_keys(layout):
            if key not in shard:
                ok = False
                continue
            f = out_dir / shard[key]
            rows = shard.get("n_issues", 0) if key == "inv" else shard["n_samples"]
            want_items = rows * chans[key] * recipe["grid_hw"][0] * recipe["grid_hw"][1]
            want_bytes = want_items * np.dtype(recipe["dtype"]).itemsize
            if not f.exists() or f.stat().st_size < want_bytes:
                ok = False
        if ok:
            keep[int(shard["id"])] = shard
    return keep


def _prune_stale(out_dir: pathlib.Path, keep_names: set[str]) -> int:
    """Delete every ``*.npy`` / ``*.npy.tmp`` in ``out_dir`` that the new plan
    does not name. Only called under ``--force``: without this, re-rendering a
    smaller sample set (fewer leads, ``--max-samples``, ``--layout flat`` over a
    dedup store) leaves the previous run's higher-numbered shards and its
    orphaned ``inv_*.npy`` sitting there — not referenced by the manifest, so
    harmless to the loader, but tens of gigabytes of it on the real store."""
    removed = 0
    for f in sorted(out_dir.iterdir()):
        if not f.is_file():
            continue
        if not (f.name.endswith(".npy") or f.name.endswith(".npy.tmp")):
            continue
        if f.name in keep_names:
            continue
        f.unlink()
        removed += 1
    if removed:
        LOG.info("--force: removed %d stale file(s) from %s", removed, out_dir)
    return removed


# ─────────────────────────────────────────────────────────────────── render


def render_split(args: argparse.Namespace, split: str, boundary: datetime | None) -> pathlib.Path:
    out_dir = pathlib.Path(args.out) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    ds = _open_dataset(args, split, boundary, build_index=True)
    index = ds.index
    if args.max_samples is not None:
        index = index[: args.max_samples]
    if not index:
        raise SystemExit(
            f"[{split}] no samples to render (index is empty after --max-samples="
            f"{args.max_samples!r}). Writing a 'complete' manifest with zero shards "
            "would produce a store that looks finished and trains on nothing."
        )
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
    var_idx = list(ds.lead_varying_channels())
    n_var = len(var_idx)
    n_inv = ds.n_channels - n_var
    source_store = (getattr(args, "source_store_hash", None)
                    or source_store_hash(args.zarr))
    keep = _load_existing(out_dir, recipe, force=args.force, layout=args.layout,
                          n_inv=n_inv, n_var=n_var, source_store=source_store,
                          samples_per_shard=args.samples_per_shard)

    plan = _shard_plan(index, args.samples_per_shard)
    jobs = []
    for sid, (start, stop) in enumerate(plan):
        job = {
            "id": sid,
            "out_dir": str(out_dir),
            "first_sample": start,
            "x": f"x_{sid:05d}.npy",
            "y": f"y_{sid:05d}.npy",
        }
        if args.layout == LAYOUT_DEDUP:
            job["inv"] = f"inv_{sid:05d}.npy"
        # BOTH the count and the OFFSET must match: a shard is identified by
        # where it starts in the index, not by how many samples it holds.
        if (sid in keep and keep[sid]["n_samples"] == stop - start
                and int(keep[sid].get("first_sample", -1)) == start):
            continue
        job["samples"] = [
            {"issue_idx": s.issue_idx, "lead_min": s.lead_min,
             "history_idx": list(s.history_idx), "target_idx": s.target_idx,
             "issue_epoch": s.issue_epoch}
            for s in index[start:stop]
        ]
        jobs.append(job)
    LOG.info("[%s] %d shard(s) planned, %d already on disk, %d to render "
             "(layout=%s: %d lead-invariant channel(s) per issue, %d per sample)",
             split, len(plan), len(plan) - len(jobs), len(jobs), args.layout,
             n_inv if args.layout == LAYOUT_DEDUP else 0,
             n_var if args.layout == LAYOUT_DEDUP else ds.n_channels)

    if args.force:
        planned_names = {INDEX_NAME}
        for sid in range(len(plan)):
            planned_names |= {f"x_{sid:05d}.npy", f"y_{sid:05d}.npy"}
            if args.layout == LAYOUT_DEDUP:
                planned_names.add(f"inv_{sid:05d}.npy")
        _prune_stale(out_dir, planned_names)

    # index.npy: per-sample metadata, same order as the samples themselves.
    # tmp + rename like the shards, so a kill mid-write cannot leave a
    # truncated index next to a complete manifest.
    meta = np.zeros(len(index), dtype=INDEX_DTYPE)
    for i, s in enumerate(index):
        meta[i] = (s.issue_epoch, s.lead_min, s.issue_idx, s.target_idx)
    index_tmp = out_dir / (INDEX_NAME + ".tmp")
    with open(index_tmp, "wb") as fh:          # np.save(path) would append .npy
        np.save(fh, meta, allow_pickle=False)
    os.replace(index_tmp, out_dir / INDEX_NAME)

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "complete": False,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "split": split,
        "layout": args.layout,
        "n_samples": len(index),
        "grid_hw": recipe["grid_hw"],
        "n_channels": recipe["n_channels"],
        "dtype": args.dtype,
        "samples_per_shard": args.samples_per_shard,
        "channel_recipe": ds.channel_recipe(),
        "lead_varying_channels": var_idx,
        "recipe": recipe,
        "recipe_hash": recipe_hash(recipe),
        "source_store": source_store,
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
               "dtype": args.dtype, "layout": args.layout,
               "lead_varying_channels": var_idx,
               "channel_names": ds.channel_names()}
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
                        f"(default {DEFAULT_SAMPLES_PER_SHARD}). Changing it re-cuts every "
                        f"boundary, so it cannot be changed on a resume — a rerun with a "
                        f"different value refuses and asks for --force")
    p.add_argument("--layout", default=DEFAULT_LAYOUT, choices=LAYOUTS,
                   help="on-disk layout. 'dedup' (default) stores the lead-invariant "
                        "channels — the history stack, aux, statics, 29 of 33 with the "
                        "current store — once per ISSUE and only the lead-dependent planes "
                        "plus the target per sample: 0.86 vs 2.39 MiB/sample, 119 vs 332 GiB "
                        "for the full v3 set. 'flat' stores every channel of every sample. "
                        "ShardDataset hands out the identical tensor either way.")
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

    # Fingerprint the store ONCE for every split: the content half reads ~64x3
    # planes, and rendering train+val must compare both halves against the same
    # store anyway (a rebuild between the two splits is exactly the failure the
    # fingerprint exists to catch, and it would show up as a mismatch on the
    # second split).
    args.source_store_hash = source_store_hash(args.zarr)
    LOG.info("source store: %s (%s, %d issues, %d sampled)",
             args.source_store_hash["hash"][:16], args.source_store_hash["hash_mode"],
             args.source_store_hash["n_issues"],
             args.source_store_hash["sampled"]["n_issues_sampled"])

    for split in splits:
        out_dir = render_split(args, split, boundary)
        if args.verify:
            ShardDataset(out_dir, verify_checksums=True)
            LOG.info("[%s] checksums verified", split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
