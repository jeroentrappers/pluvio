"""Streaming Dataset over pre-rendered training shards (``tools/render_shards.py``).

Why this exists
───────────────
``ZarrCorrectionDataset`` re-assembles every sample from the zarr store on every
epoch: one chunk read per history frame, one per aux channel, plus the
normalisation arithmetic and the time-encoding planes — ~35 array reads and a
handful of numpy ops per sample, ×113k samples, ×every epoch. On the v3 store
that is the epoch cost (~47 min at 192², batch 8, 6 workers; the GPU is not the
bottleneck).

A shard store is that work done exactly once. ``render_shards.py`` walks the
*same* ``ZarrCorrectionDataset`` index and writes the assembled
``(n_channels, H, W)`` inputs and ``(1, H, W)`` targets into ``.npy`` memmaps.
Reading a sample here is a memmap slice or two plus one dtype cast — no zarr,
no normalisation, no datetime arithmetic.

Layout
──────
Most of a sample does not depend on the lead. With the current store 29 of the
33 channels ``build_input`` writes are a function of the ISSUE alone — the
radar history stack, the aux planes, the statics (and ``lagrangian_flow_mag``
when the Lagrangian planes are on) — and only ``nowcast_at_lead``,
``lead_over_120``, ``tod_sin``, ``tod_cos`` (and ``lagrangian_rate``) change
between an issue's four leads. So the default ``"dedup"`` layout stores the
lead-invariant block ONCE PER ISSUE (``inv_%05d.npy``) and the lead-dependent
planes plus the target per sample (``x_%05d.npy``, ``y_%05d.npy``), and
``__getitem__`` reassembles the full ``(C, H, W)`` input in ``build_input``
channel order. That is bit-for-bit the flat sample at ~2.8x less disk — the
difference between 332 GiB and 119 GiB for the full v3 sample set, which is
what makes it fit on the render box at all. ``"flat"`` (one whole sample per
row, no ``inv`` file) stays available behind ``--layout flat``; it is also how
every store rendered before this change reads, since a manifest with no
``"layout"`` key means flat.

Which channels are lead-invariant is NOT hard-coded here: it comes from
``zarr_dataset.lead_varying_channel_indices`` over the manifest's own
``channel_names``, and the renderer verifies the split per issue by comparing
an issue's leads plane by plane before it writes anything.

Contract
────────
``ShardDataset`` is drop-in for ``ZarrCorrectionDataset``: same ``__len__``,
same ``__getitem__`` → ``(x, y)`` float32 ``torch.Tensor`` pair, same
``n_channels`` / ``grid_hw`` attributes, and the same sample *order* (shards are
concatenated in render order, which is the dataset's index order).

── The float16 question ──────────────────────────────────────────────────
The cast to the shard dtype happens **once, at render time** (in
``render_shards.py``, via ``cast_for_shard``); ``ShardDataset`` casts straight
back to float32 on read. So with the default ``--dtype float16``:

    ShardDataset[i] == cast_for_shard(ZarrCorrectionDataset[i]).astype(float32)

bit-for-bit — the equality the tests assert, with the identical cast applied to
both sides.

How much does that cast actually lose? Less than the first version of this
docstring claimed. ``_normalise`` divides float16 store arrays by python
floats, which under numpy's weak scalar promotion stays float16 arithmetic —
so the normalised aux (``/255``), SST and static channels are float16 values
*already*, and the cast is exact. The radar history and the nowcast plane come
straight out of a float16 store. ``lead/120`` is exact for every lead the
30-min cadence allows (0.25/0.5/0.75/1.0). What genuinely quantises is:

* ``tod_sin`` / ``tod_cos`` — computed in float64/float32, error ≤ 2.4e-4;
* with ``--lagrangian-channels``, ``lagrangian_rate`` and
  ``lagrangian_flow_mag`` — the warp and the hypot run in float32.

That is deliberate and harmless: CUDA training runs under
``autocast(float16)``, so the model never saw more precision than this. Render
with ``--dtype float32`` when exact float32 equality is wanted (2× the disk).

Recipe safety
─────────────
Everything that determines *what a sample means* (channel list and order, the
history/lead geometry, the dry-sample filter, the normalisation version, the
train/val boundary) is recorded in the manifest as the **recipe**. Loading
shards whose recipe disagrees with what the caller expects raises
``ShardRecipeMismatch`` — silently training on stale shards (a store rebuild, a
changed ``_normalise``, a different lead set) is the failure mode this module
exists to make impossible.

Augmentation
────────────
``ZarrCorrectionDataset`` has no augmentation hooks — every sample is
deterministic — which is why shards are a faithful substitute. If augmentation
is ever added it must be applied *after* the shard read (pass ``transform``
here) and never baked into the shards, or every epoch would see the same fixed
draw. Record any such transform in the run config, not in the recipe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

LOG = logging.getLogger("pluvio.shard_dataset")

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.npy"
MANIFEST_VERSION = 1

# Fields that pin down the meaning of a sample. Two shard stores with equal
# recipes hold interchangeable samples; anything else must fail loudly.
RECIPE_KEYS: tuple[str, ...] = (
    "dataset_class",
    "grid_hw",
    "n_channels",
    "history_steps",
    "history_step_min",
    "history_tolerance_s",
    "leads_min",
    "aux_channels",
    "static_channels",
    "lagrangian_channels",
    "has_truth",
    "require_rain_fraction",
    "normalise_version",
    "dtype",
    "split",
    "split_boundary_epoch",
    "val_frac",
    "max_samples",
)

# Per-sample metadata written next to the shards (structured .npy, so the
# loader never parses a 100k-entry JSON list).
INDEX_DTYPE = np.dtype([
    ("issue_epoch", "int64"),
    ("lead_min", "int32"),
    ("issue_idx", "int32"),
    ("target_idx", "int32"),
])

SHARD_DTYPES = ("float16", "float32")

# ── on-disk layouts ────────────────────────────────────────────────────────
# "dedup" (default): 29 of the 33 channels build_input writes are a function of
#   the ISSUE, not of the lead (see zarr_dataset.LEAD_VARYING_CHANNEL_NAMES), so
#   they are stored ONCE PER ISSUE in ``inv_%05d.npy`` and the 4 lead-dependent
#   planes once per sample in ``x_%05d.npy``. ShardDataset reassembles the full
#   (C, H, W) input in build_input channel order — bit-for-bit the flat sample,
#   at 2.8x less disk (see docs/training_run_v2.md).
# "flat": the whole (C, H, W) input per sample in ``x_%05d.npy``, no inv file.
#   Kept because it is one branch here and one in the renderer, and because it
#   is the layout every store rendered before this change already has (a
#   manifest with no "layout" key reads as flat).
LAYOUT_DEDUP = "dedup"
LAYOUT_FLAT = "flat"
LAYOUTS = (LAYOUT_DEDUP, LAYOUT_FLAT)
DEFAULT_LAYOUT = LAYOUT_DEDUP

# Issue indices sampled by ``source_store_hash`` for its content half.
SOURCE_HASH_SAMPLES = 64


class ShardRecipeMismatch(ValueError):
    """Raised when a shard store's recipe disagrees with what was expected."""


class ShardSourceMismatch(ShardRecipeMismatch):
    """Raised when a shard store's SOURCE zarr no longer fingerprints the same.

    Separate from a recipe mismatch because the recipe can be identical while
    the store's *values* have been rebuilt underneath it: same arrays, same
    shapes, same ``issue_time``, different numbers. That store renders
    different samples, so half-rendered shards from before the rebuild and
    shards rendered after it must never end up in one directory.
    """


class ShardStoreIncomplete(RuntimeError):
    """Raised when a shard store's manifest is not marked complete."""


def cast_for_shard(arr: np.ndarray, dtype: str) -> np.ndarray:
    """The one and only quantisation step between the zarr dataset and a shard.

    Both the renderer and the tests go through this, so "the same cast on both
    sides" is a property of the code rather than of a convention."""
    if dtype not in SHARD_DTYPES:
        raise ValueError(f"unsupported shard dtype {dtype!r}; expected one of {SHARD_DTYPES}")
    return np.ascontiguousarray(arr, dtype=np.dtype(dtype))


def recipe_from_dataset(
    ds: Any,
    *,
    split: str,
    dtype: str,
    split_boundary_epoch: int | None,
    val_frac: float | None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Build the recipe dict for a ``ZarrCorrectionDataset`` instance.

    Shared by the renderer (writes it) and train.py (checks it), so the two can
    never disagree about what a field means."""
    from model.zarr_dataset import NORMALISE_VERSION

    return {
        "dataset_class": type(ds).__name__,
        "grid_hw": [int(v) for v in ds.grid_hw],
        "n_channels": int(ds.n_channels),
        "history_steps": int(ds.history_steps),
        "history_step_min": int(ds.history_step_min),
        "history_tolerance_s": int(ds.history_tolerance_s),
        # The EFFECTIVE leads: a requested lead the store does not carry is
        # skipped by the indexer, so recording the requested superset would
        # make an equivalent shard store look like a mismatch (e.g. the default
        # 30/60/90/120 against a store whose lead axis stops at 90).
        "leads_min": [int(v) for v in ds.leads_min if int(v) in ds._lead_to_idx],
        "aux_channels": list(ds.aux_channels),
        "static_channels": list(ds.static_channels),
        # The Lagrangian planes (2.3) are assembled per sample from the flow
        # between two history frames, so shards rendered with them are NOT
        # interchangeable with shards rendered without — and the channel count
        # alone does not separate the two once an aux channel differs too.
        "lagrangian_channels": int(ds.lagrangian_channels),
        "has_truth": bool(ds._has_truth),
        "require_rain_fraction": ds.require_rain_fraction,
        "normalise_version": int(NORMALISE_VERSION),
        "dtype": dtype,
        "split": split,
        "split_boundary_epoch": split_boundary_epoch,
        "val_frac": val_frac,
        "max_samples": max_samples,
    }


def _canonical(value: Any) -> Any:
    """JSON round-trip normalisation, so a tuple/list or int/float difference in
    how a recipe was constructed never reads as a mismatch."""
    return json.loads(json.dumps(value, sort_keys=True))


def recipe_hash(recipe: dict[str, Any]) -> str:
    payload = json.dumps({k: _canonical(recipe.get(k)) for k in RECIPE_KEYS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def compare_recipes(
    got: dict[str, Any], expected: dict[str, Any], *, ignore: frozenset[str] = frozenset()
) -> list[str]:
    """Human-readable list of recipe field differences (empty → equal)."""
    diffs = []
    for key in RECIPE_KEYS:
        if key in ignore:
            continue
        a, b = _canonical(got.get(key)), _canonical(expected.get(key))
        if a != b:
            diffs.append(f"{key}: shards={a!r} expected={b!r}")
    return diffs


def sha256_file(path: pathlib.Path, *, block: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


# ────────────────────────────────────────────────────── source-store identity


def _sampled_issue_indices(n_issues: int, n_samples: int = SOURCE_HASH_SAMPLES) -> np.ndarray:
    """``n_samples`` evenly spaced issue indices (all of them if the store is
    smaller). Deterministic, so two runs over the same store sample the same
    slices and the digests are comparable."""
    if n_issues <= 0:
        return np.zeros(0, dtype="int64")
    if n_issues <= n_samples:
        return np.arange(n_issues, dtype="int64")
    return np.unique(np.linspace(0, n_issues - 1, n_samples).round().astype("int64"))


def source_store_hash(zarr_path: str | pathlib.Path,
                      *, n_samples: int = SOURCE_HASH_SAMPLES) -> dict[str, Any]:
    """Fingerprint of the source store: structural **plus sampled content**.

    A full content hash is out of the question — the real store is ~14 GB and
    re-reading it would cost more than the render. But a purely structural hash
    (which is what this was) misses the failure found in review: rebuilding the
    store IN PLACE with the same arrays, shapes, attrs and ``issue_time`` leaves
    the fingerprint identical, so a resumed render fills the missing shards in
    from the NEW values and the loader accepts a directory that is half one
    store and half another.

    So the digest covers, in order:

    * the group attrs;
    * every array's name, shape and dtype;
    * the full ``issue_time`` vector (an appended, truncated or re-timed store
      fingerprints differently);
    * and the CONTENT of ``radar[i, 0]``, ``truth[i]`` and one aux array at
      ``n_samples`` evenly spaced issue indices ``i``.

    The content half is ~64 x 3 planes — seconds on the real store, and it is
    the half that catches an in-place value rebuild. It is a sample, not a
    proof: an edit confined to the issues between two probes still slips past.
    ``hash_mode`` says which mode produced the digest so an old manifest stays
    recognisable, and ``sampled`` records exactly what was read.
    """
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    parts: list[str] = [json.dumps(dict(sorted(root.attrs.items())), sort_keys=True, default=str)]
    array_keys = sorted(root.array_keys())
    for name in array_keys:
        arr = root[name]
        parts.append(f"{name}:{tuple(arr.shape)}:{arr.dtype!s}")
    issue = np.asarray(root["issue_time"][:], dtype="int64")

    h = hashlib.sha256("|".join(parts).encode())
    h.update(issue.tobytes())

    # One aux plane alongside radar+truth: enough to notice an aux rebuild
    # without reading a dozen arrays per probe. Picked deterministically (the
    # first per-issue 3-D array that is not radar/truth) and recorded by name.
    per_issue = [n for n in array_keys
                 if n not in ("radar", "truth", "issue_time", "leads_min")
                 and len(root[n].shape) == 3 and root[n].shape[0] == issue.size]
    probes: list[tuple[str, bool]] = [("radar", True)]
    sampled_names = ["radar[:, 0]"]
    if "truth" in array_keys:
        probes.append(("truth", False))
        sampled_names.append("truth")
    if per_issue:
        probes.append((per_issue[0], False))
        sampled_names.append(per_issue[0])

    idx = _sampled_issue_indices(int(issue.size), n_samples)
    for i in idx.tolist():
        h.update(np.int64(i).tobytes())
        for name, is_radar in probes:
            block = root[name][i, 0] if is_radar else root[name][i]
            h.update(np.ascontiguousarray(block).tobytes())

    return {
        "path": str(zarr_path),
        "hash": h.hexdigest(),
        "hash_mode": "structural+sampled",
        "hash_covers": "group attrs + array names/shapes/dtypes + full issue_time vector "
                       f"+ content of {', '.join(sampled_names)} at {idx.size} evenly "
                       "spaced issue indices",
        "n_issues": int(issue.size),
        "sampled": {"arrays": sampled_names, "n_issues_sampled": int(idx.size),
                    "issue_indices": [int(v) for v in idx]},
    }


def compare_source_store(got: dict[str, Any] | None, expected: dict[str, Any] | None,
                         *, what: str, remedy: str) -> None:
    """Raise ``ShardSourceMismatch`` when two source fingerprints disagree.

    A missing fingerprint on either side is not a mismatch (a store rendered
    before the hash existed, or a caller with no zarr to hash) — only two
    present-and-different digests are."""
    if not got or not expected:
        return
    a, b = got.get("hash"), expected.get("hash")
    if not a or not b or a == b:
        return
    raise ShardSourceMismatch(
        f"{what}: the source store no longer fingerprints the same "
        f"(shards={a[:16]} store={b[:16]}).\n"
        f"  shards rendered from: {got.get('path')} "
        f"({got.get('hash_mode')}, {got.get('n_issues')} issues)\n"
        f"  store now:            {expected.get('path')} "
        f"({expected.get('hash_mode')}, {expected.get('n_issues')} issues)\n"
        "The recipe is unchanged, so this is a store REBUILD: same arrays and "
        "issue_time, different values. Every sample rendered before it means "
        f"something different from every sample rendered after it. {remedy}"
    )


def shard_file_keys(layout: str) -> tuple[str, ...]:
    """Manifest keys naming the files one shard of ``layout`` consists of."""
    return ("x", "y", "inv") if layout == LAYOUT_DEDUP else ("x", "y")


def read_manifest(shard_dir: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(shard_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {shard_dir!r} — not a shard store "
            "(render one with tools/render_shards.py)"
        )
    with open(path) as fh:
        manifest = json.load(fh)
    version = manifest.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ShardRecipeMismatch(
            f"{path}: manifest_version {version!r}, this build reads "
            f"{MANIFEST_VERSION} — re-render the shards"
        )
    return manifest


class ShardDataset(Dataset):
    """Reads pre-rendered shards. See the module docstring for the contract.

    ``expected_recipe`` (e.g. from ``recipe_from_dataset``) is compared field by
    field; any difference raises ``ShardRecipeMismatch`` naming the fields.
    ``expected_source_store`` (from ``source_store_hash``) additionally catches
    the case the recipe cannot see: a source store rebuilt in place with the
    same arrays, shapes and ``issue_time`` but different values.
    """

    def __init__(
        self,
        shard_dir: str | pathlib.Path,
        *,
        expected_recipe: dict[str, Any] | None = None,
        expected_channels: int | None = None,
        expected_source_store: dict[str, Any] | None = None,
        ignore_recipe_keys: frozenset[str] = frozenset(),
        verify_checksums: bool = False,
        max_open_shards: int = 64,
        transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ):
        self.shard_dir = pathlib.Path(shard_dir)
        self.manifest = read_manifest(self.shard_dir)
        self.recipe: dict[str, Any] = self.manifest["recipe"]
        self.transform = transform
        self._max_open = max(1, int(max_open_shards))
        self._open_shards: OrderedDict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray | None]
        ] = OrderedDict()
        self._pid: int | None = None

        if not self.manifest.get("complete"):
            done = len(self.manifest.get("shards", []))
            raise ShardStoreIncomplete(
                f"{self.shard_dir}: shard store is incomplete ({done} shard(s) written) — "
                "re-run tools/render_shards.py to finish it (the render is resumable)"
            )

        if expected_recipe is not None:
            diffs = compare_recipes(self.recipe, expected_recipe, ignore=ignore_recipe_keys)
            if diffs:
                raise ShardRecipeMismatch(
                    f"{self.shard_dir}: shard recipe does not match the expected recipe "
                    f"({len(diffs)} field(s)):\n  " + "\n  ".join(diffs)
                    + "\nThese shards were rendered from different sample semantics; "
                    "re-render them or point at the matching store."
                )
        stored_hash = self.manifest.get("recipe_hash")
        if stored_hash and stored_hash != recipe_hash(self.recipe):
            # Name the schema drift as well as the fact of it: a store rendered
            # by an older build is missing whatever RECIPE_KEYS have been added
            # since (a pre-2.3 store has no `lagrangian_channels`), and that
            # reads very differently from a hand-edited manifest.
            missing = [k for k in RECIPE_KEYS if k not in self.recipe]
            added = [k for k in self.recipe if k not in RECIPE_KEYS]
            detail = ""
            if missing:
                detail += f"\n  recipe keys MISSING vs this build: {', '.join(missing)}"
            if added:
                detail += f"\n  recipe keys this build does not know: {', '.join(added)}"
            if missing or added:
                detail += ("\n  → these shards were rendered by an older build; re-render "
                           "them (tools/render_shards.py) rather than editing the manifest.")
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: manifest recipe_hash does not match its own recipe "
                "— the manifest was edited or written by a different build" + detail
            )

        # NORMALISE_VERSION is a property of THIS BUILD, not of the store, so
        # this one check needs no zarr at all — which matters, because "the
        # loader refuses shards baked under a different _normalise" was only
        # true with --zarr while it lived in the store-derived recipe alone.
        from model.zarr_dataset import NORMALISE_VERSION
        stored_nv = self.recipe.get("normalise_version")
        if stored_nv is not None and int(stored_nv) != int(NORMALISE_VERSION):
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: shards were rendered under normalise_version "
                f"{int(stored_nv)}, this build is at {int(NORMALISE_VERSION)} — "
                "zarr_dataset._normalise (or build_input's channel order/scaling) has "
                "changed since, so every rendered channel means something else. "
                "Re-render the shards."
            )

        compare_source_store(
            self.manifest.get("source_store"), expected_source_store,
            what=str(self.shard_dir),
            remedy="Re-render the shards from the current store "
                   "(tools/render_shards.py --force), or point --zarr at the store they "
                   "were rendered from.",
        )

        self.dtype = str(self.recipe["dtype"])
        self.n_channels = int(self.recipe["n_channels"])
        self.grid_hw: tuple[int, int] = tuple(int(v) for v in self.recipe["grid_hw"])

        # Layout is a STORAGE choice, not sample semantics — the reassembled
        # sample is identical either way — so it lives in the manifest and not
        # in the recipe (a dedup store and a flat store of the same samples
        # must stay interchangeable for train.py's recipe comparison).
        self.layout = str(self.manifest.get("layout", LAYOUT_FLAT))
        if self.layout not in LAYOUTS:
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: manifest layout {self.layout!r} is not one of "
                f"{LAYOUTS} — written by a newer build?"
            )
        self._inv_idx = np.zeros(0, dtype="int64")
        self._var_idx = np.zeros(0, dtype="int64")
        if self.layout == LAYOUT_DEDUP:
            self._inv_idx, self._var_idx = self._channel_split()

        expected = expected_channels
        if expected is None:
            env_expected = os.environ.get("PLUVIO_EXPECTED_CHANNELS")
            expected = int(env_expected) if env_expected else None
        if expected is not None and self.n_channels != expected:
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: shards hold {self.n_channels} channels, expected "
                f"{expected} (aux={self.recipe['aux_channels']}, "
                f"static={self.recipe['static_channels']})"
            )

        # Shard table → flat O(1) (shard, offset) lookup arrays.
        self._shards = list(self.manifest["shards"])
        if not self._shards:
            raise ShardStoreIncomplete(f"{self.shard_dir}: manifest lists no shards")
        counts = np.asarray([int(s["n_samples"]) for s in self._shards], dtype="int64")
        self._shard_of = np.repeat(np.arange(len(counts), dtype="int64"), counts)
        self._offset_of = np.concatenate(
            [np.arange(c, dtype="int64") for c in counts]
        ) if counts.size else np.zeros(0, dtype="int64")
        self._n = int(counts.sum())
        if self._n != int(self.manifest["n_samples"]):
            raise ShardStoreIncomplete(
                f"{self.shard_dir}: shard sample counts sum to {self._n} but the manifest "
                f"says n_samples={self.manifest['n_samples']}"
            )

        for shard in self._shards:
            for key in shard_file_keys(self.layout):
                if key not in shard:
                    raise ShardStoreIncomplete(
                        f"{self.shard_dir}: shard {shard['id']} has no {key!r} file recorded, "
                        f"which layout {self.layout!r} requires — re-render the store"
                    )
                path = self.shard_dir / shard[key]
                if not path.exists():
                    raise ShardStoreIncomplete(f"{self.shard_dir}: missing shard file {path.name}")
        if verify_checksums:
            self.verify_checksums()

        # Per-sample metadata (issue_epoch / lead_min / issue_idx) — same order
        # as the samples, so callers can stratify a shard run exactly like a
        # zarr run. REQUIRED, not optional: it used to fall back to a
        # zero-filled array, which turned "the index never got written" into a
        # store whose every sample claims issue_epoch 0 and issue_idx 0 — a
        # silently wrong stratification, and (under the dedup layout) a silently
        # wrong issue→row mapping.
        index_path = self.shard_dir / INDEX_NAME
        if not index_path.exists():
            raise ShardStoreIncomplete(
                f"{self.shard_dir}: no {INDEX_NAME} — the per-sample index is part of a "
                "shard store, not an optional extra (it carries issue_epoch/lead_min for "
                "stratification and the issue grouping the dedup layout reads). Re-run "
                "tools/render_shards.py to write it."
            )
        self.index = np.load(index_path)
        if len(self.index) != self._n:
            raise ShardStoreIncomplete(
                f"{self.shard_dir}: {INDEX_NAME} has {len(self.index)} rows, expected {self._n}"
            )
        # Which row of a shard's per-issue block each sample reads.
        self._inv_row_of = (self._issue_rows() if self.layout == LAYOUT_DEDUP
                            else np.zeros(0, dtype="int64"))

        LOG.info(
            "shard_dataset: %s — %d samples in %d shard(s), %d channels @ %s, dtype=%s, "
            "layout=%s (%d lead-invariant channel(s) stored per issue)",
            self.shard_dir, self._n, len(self._shards), self.n_channels, self.grid_hw,
            self.dtype, self.layout, len(self._inv_idx),
        )

    # ──────────────────────────────────────────────────── dedup bookkeeping

    def _channel_split(self) -> tuple[np.ndarray, np.ndarray]:
        """(lead-invariant, lead-varying) channel indices for the dedup layout.

        Derived from the manifest's own ``channel_names`` through the dataset's
        ``lead_varying_channel_indices`` — and cross-checked against the split
        the RENDERER recorded. Those two disagreeing means the channel set
        changed under a rendered store, which would reassemble every sample
        with two planes swapped; that has to be a load error, not a silent
        one."""
        from model.zarr_dataset import lead_varying_channel_indices

        names = list((self.manifest.get("channel_recipe") or {}).get("channel_names") or [])
        if len(names) != self.n_channels:
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: layout {LAYOUT_DEDUP!r} needs "
                f"channel_recipe.channel_names to reassemble a sample, but the manifest "
                f"lists {len(names)} name(s) for {self.n_channels} channels — re-render"
            )
        var = lead_varying_channel_indices(names)
        recorded = self.manifest.get("lead_varying_channels")
        if recorded is not None and [int(v) for v in recorded] != list(var):
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: the store was rendered splitting channels "
                f"{[int(v) for v in recorded]} as lead-varying, but this build derives "
                f"{list(var)} from the same channel names — build_input's channel "
                "semantics changed under the store; re-render it"
            )
        var_set = set(var)
        inv = [i for i in range(self.n_channels) if i not in var_set]
        return np.asarray(inv, dtype="int64"), np.asarray(var, dtype="int64")

    def _issue_rows(self) -> np.ndarray:
        """Row of its shard's per-issue block that each sample reads.

        Not stored per sample: an issue's leads are consecutive in the index by
        construction (the indexer loops issues outer, leads inner) and a shard
        boundary only ever falls on an issue boundary, so the row is the count
        of issue changes so far within the shard. Both of those properties are
        asserted here rather than assumed — a store that violates them would
        hand out samples assembled from another issue's history."""
        rows = np.zeros(self._n, dtype="int64")
        issue_idx = np.asarray(self.index["issue_idx"], dtype="int64")
        pos = 0
        for shard in self._shards:
            n = int(shard["n_samples"])
            block = issue_idx[pos:pos + n]
            changed = np.empty(n, dtype=bool)
            changed[0] = True
            changed[1:] = block[1:] != block[:-1]
            row = np.cumsum(changed) - 1
            n_issues = int(row[-1]) + 1
            if np.unique(block).size != n_issues:
                raise ShardStoreIncomplete(
                    f"{self.shard_dir}: shard {shard['id']} has an issue whose samples are "
                    f"not consecutive in {INDEX_NAME} ({np.unique(block).size} distinct "
                    f"issues in {n_issues} runs) — the dedup layout cannot address it"
                )
            recorded = shard.get("n_issues")
            if recorded is not None and int(recorded) != n_issues:
                raise ShardStoreIncomplete(
                    f"{self.shard_dir}: shard {shard['id']} holds {recorded} per-issue "
                    f"block(s) but {INDEX_NAME} groups its {n} sample(s) into {n_issues} "
                    "issue(s) — index and shards are from different renders"
                )
            rows[pos:pos + n] = row
            pos += n
        return rows

    # ───────────────────────────────────────────────────────────── integrity

    def verify_checksums(self) -> None:
        """Re-hash every shard file against the manifest (slow: reads the whole
        store). Used by ``render_shards.py --verify`` and by tests."""
        for shard in self._shards:
            for key in shard_file_keys(self.layout):
                path = self.shard_dir / shard[key]
                want = shard.get(f"sha256_{key}")
                if not want:
                    continue
                got = sha256_file(path)
                if got != want:
                    raise ShardStoreIncomplete(
                        f"{path}: sha256 {got} != manifest {want} — shard is corrupt, "
                        "delete it and re-run tools/render_shards.py"
                    )

    # ─────────────────────────────────────────────────────────────── access

    def _shard_arrays(self, sid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Memmaps for one shard — (per-sample x, y, per-issue block or None),
        LRU-capped so a 400-shard store does not hold 1200 file handles open per
        worker."""
        pid = os.getpid()
        if self._pid != pid:            # forked DataLoader worker: drop inherited maps
            self._open_shards.clear()
            self._pid = pid
        hit = self._open_shards.get(sid)
        if hit is not None:
            self._open_shards.move_to_end(sid)
            return hit
        shard = self._shards[sid]
        pair = (
            np.load(self.shard_dir / shard["x"], mmap_mode="r"),
            np.load(self.shard_dir / shard["y"], mmap_mode="r"),
            (np.load(self.shard_dir / shard["inv"], mmap_mode="r")
             if self.layout == LAYOUT_DEDUP else None),
        )
        self._open_shards[sid] = pair
        while len(self._open_shards) > self._max_open:
            self._open_shards.popitem(last=False)
        return pair

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(f"index {idx} out of range for {self._n} samples")
        xs, ys, inv = self._shard_arrays(int(self._shard_of[idx]))
        off = int(self._offset_of[idx])
        # .astype always copies → a writable, contiguous, owned array (a plain
        # asarray on a float32 memmap slice hands torch a read-only view).
        if inv is None:
            arr = xs[off].astype("float32")
        else:
            # Reassemble in build_input channel order. float16 → float32 is
            # exact, so this is bit-for-bit the flat layout's row; the cost is
            # one (C, H, W) fill per sample instead of one slice copy.
            arr = np.empty((self.n_channels, *self.grid_hw), dtype="float32")
            arr[self._inv_idx] = inv[int(self._inv_row_of[idx])]
            arr[self._var_idx] = xs[off]
        x = torch.from_numpy(arr)
        y = torch.from_numpy(ys[off].astype("float32"))
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y
