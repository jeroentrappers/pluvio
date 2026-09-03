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
Reading a sample here is one memmap slice plus one dtype cast — no zarr, no
normalisation, no datetime arithmetic.

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
both sides. It is *not* bit-equal to the un-quantised float32 sample: normalised
aux (``/255``), the ``lead/120`` plane and the static scalings land on values
float16 cannot represent exactly. That is deliberate and harmless — the store's
own arrays are already float16, and CUDA training runs under
``autocast(float16)`` anyway, so the model never saw more precision than this.
Render with ``--dtype float32`` when exact float32 equality is wanted (2× the
disk).

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


class ShardRecipeMismatch(ValueError):
    """Raised when a shard store's recipe disagrees with what was expected."""


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
    """

    def __init__(
        self,
        shard_dir: str | pathlib.Path,
        *,
        expected_recipe: dict[str, Any] | None = None,
        expected_channels: int | None = None,
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
        self._open_shards: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
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
            raise ShardRecipeMismatch(
                f"{self.shard_dir}: manifest recipe_hash does not match its own recipe "
                "— the manifest was edited or written by a different build"
            )

        self.dtype = str(self.recipe["dtype"])
        self.n_channels = int(self.recipe["n_channels"])
        self.grid_hw: tuple[int, int] = tuple(int(v) for v in self.recipe["grid_hw"])

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
            for key in ("x", "y"):
                path = self.shard_dir / shard[key]
                if not path.exists():
                    raise ShardStoreIncomplete(f"{self.shard_dir}: missing shard file {path.name}")
        if verify_checksums:
            self.verify_checksums()

        # Per-sample metadata (issue_epoch / lead_min) — same order as the
        # samples, so callers can stratify a shard run exactly like a zarr run.
        index_path = self.shard_dir / INDEX_NAME
        self.index = (np.load(index_path) if index_path.exists()
                      else np.zeros(self._n, dtype=INDEX_DTYPE))
        if len(self.index) != self._n:
            raise ShardStoreIncomplete(
                f"{self.shard_dir}: {INDEX_NAME} has {len(self.index)} rows, expected {self._n}"
            )

        LOG.info(
            "shard_dataset: %s — %d samples in %d shard(s), %d channels @ %s, dtype=%s",
            self.shard_dir, self._n, len(self._shards), self.n_channels, self.grid_hw, self.dtype,
        )

    # ───────────────────────────────────────────────────────────── integrity

    def verify_checksums(self) -> None:
        """Re-hash every shard file against the manifest (slow: reads the whole
        store). Used by ``render_shards.py --verify`` and by tests."""
        for shard in self._shards:
            for key in ("x", "y"):
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

    def _shard_arrays(self, sid: int) -> tuple[np.ndarray, np.ndarray]:
        """Memmap pair for one shard, LRU-capped so a 400-shard store does not
        hold 800 file handles open per worker."""
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
        xs, ys = self._shard_arrays(int(self._shard_of[idx]))
        off = int(self._offset_of[idx])
        # .astype always copies → a writable, contiguous, owned array (a plain
        # asarray on a float32 memmap slice hands torch a read-only view).
        x = torch.from_numpy(xs[off].astype("float32"))
        y = torch.from_numpy(ys[off].astype("float32"))
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y
