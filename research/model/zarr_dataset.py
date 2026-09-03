"""PyTorch Dataset that reads the unified ``timeseries.zarr`` (tools/build_zarr.py).

This is the multi-source successor to ``dataset.py`` (which read radar HDF5 +
an old aux cache and only ever saw radar+AWS). It feeds the residual-correction
UNet the full channel set the architecture was designed for:

    radar history (lead-0 analyses, K steps)        K
    operational nowcast at the target lead           1
    lead plane (lead/120)                            1
    time-of-day sin / cos                            2
    aux per-issue 2-D channels (MSG, ALARO, SST,     n_aux
        AWS — auto-detected from the store)
    static channels (elevation, landmask, dist)      n_static
    Lagrangian persistence @ lead (opt-in, 2.3)      0, 1 or 2
  ──────────────────────────────────────────────────────
    in_channels = K + 4 + n_aux + n_static + n_lagrangian   (≈ 33 with the
        current store and the Lagrangian channels off)

The Lagrangian channels are OFF by default (``lagrangian_channels=0``), so an
existing checkpoint's ``in_channels`` and every channel index above stay
exactly what they were; the new planes are APPENDED after the statics. See
``_lagrangian_planes`` for what they carry.

Target: the radar analysis (lead-0) at ``issue_time + lead``.

── Cadence note ──────────────────────────────────────────────────────────
The store is keyed by radar issue-time. With the current 30-min cadence:
  * history steps are 30 min apart (auto-detected from the store), and
  * a target only exists when ``issue+lead`` is itself an issue-time, so the
    trainable leads are the cadence multiples ≤ 120 → {30, 60, 90, 120}.
Re-collecting radar at 5-min cadence (and rebuilding) would unlock 5-min
history and the full 5…120-min lead set; everything here adapts automatically.

Reads are lazy/chunked straight from the zarr (the store is ~12 GB, too big for
RAM); build_zarr writes one chunk per issue-time, so a sample is a handful of
small chunk reads — fine with a few DataLoader workers.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from torch.utils.data import Dataset

from model.motion import (
    BLOCKS,
    block_flow,
    km_per_px_from_bounds,
    max_shift_px,
    upsample_flow,
    warp,
)

LOG = logging.getLogger("pluvio.zarr_dataset")

GRID = (100, 100)
RADAR_HISTORY_STEPS = 6
DEFAULT_LEADS: tuple[int, ...] = (30, 60, 90, 120)
RAIN_THRESHOLD = 0.1  # mm/h, for the optional dry-sample filter
LAGRANGIAN_MAX_CHANNELS = 2   # advected analysis (+ optional flow magnitude)
_LEGACY_KM_PER_PX = 6.0       # KNMI-stereo analysis grid, for stores with no bounds/grid_n

# Channels NOT treated as per-issue aux (handled explicitly or as static).
# "truth" is the TARGET (build_zarr --truth) — auto-detecting it as an aux
# input would leak the label into the features.
_NON_AUX = {"radar", "issue_time", "leads_min", "truth"}

# issue_time must be Unix EPOCH SECONDS (build_store_v3, build_zarr both write
# it that way; a milliseconds mixup silently shifts every issue by ~1000x and
# corrupts every epoch-arithmetic lookup downstream). Sanity range: roughly
# year 2000 to year 2100 as seconds.
_EPOCH_SECONDS_MIN = 946684800   # 2000-01-01T00:00:00Z
_EPOCH_SECONDS_MAX = 4102444800  # 2100-01-01T00:00:00Z


def _assert_epoch_seconds(t: np.ndarray, context: str) -> None:
    """Raise on a units mixup (milliseconds etc. read as seconds lands far in
    the future — every history/lead lookup would silently corrupt). A low
    outlier (e.g. a zero-filled slot from a resize-before-write crash window)
    is not a units bug, so it only gets a WARNING naming the offending count
    — raising here would take down every inference run over one bad slot."""
    if t.size == 0:
        return
    lo, hi = int(t.min()), int(t.max())
    if hi > _EPOCH_SECONDS_MAX:
        raise ValueError(
            f"{context}: issue_time does not look like Unix epoch seconds "
            f"(min={lo}, max={hi}); expected max <= {_EPOCH_SECONDS_MAX} — a "
            "milliseconds/other-unit store would silently corrupt every "
            "history/lead lookup"
        )
    if lo < _EPOCH_SECONDS_MIN:
        n_bad = int(np.count_nonzero(t < _EPOCH_SECONDS_MIN))
        LOG.warning(
            "%s: %d issue_time slot(s) below %d (min=%d) — likely "
            "zero-filled/unwritten rather than a units mixup; those slots "
            "will fail lookups downstream", context, n_bad,
            _EPOCH_SECONDS_MIN, lo,
        )


# Bump whenever _normalise (or build_input's channel order/scaling) changes:
# pre-rendered shards (tools/render_shards.py) record this in their manifest so
# a loader refuses shards baked with different channel semantics.
NORMALISE_VERSION = 1


def advect_with_nan(field: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Advect ``field`` by displacement (dy, dx) — content moves by +D — with
    the store's NaN handled explicitly at both ends.

    ``motion.warp`` bilinearly interpolates, so a single NaN would smear over
    its whole neighbourhood and (via the lead-scaled displacement) drag a
    growing NaN wake across the grid. So: fill before warping, then warp the
    NaN MASK with the same displacement and restore NaN wherever the sampled
    source was NaN. The result is "no observation was advected here" rather
    than a fabricated dry cell — ``build_input``'s own final ``nan_to_num``
    is what turns it into the 0.0 the network sees, the same convention every
    other channel gets.
    """
    field = np.asarray(field, dtype="float32")
    src_nan = ~np.isfinite(field)
    out = warp(np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0), dy, dx)
    out = np.clip(out, 0.0, None)
    if src_nan.any():
        # nearest-ish: any bilinear weight on a NaN source cell poisons the
        # sample, so threshold the warped mask well below 0.5.
        wake = warp(src_nan.astype("float32"), dy, dx)
        out = np.where(wake > 1e-3, np.float32("nan"), out)
    return out.astype("float32")


def _normalise(name: str, arr: np.ndarray) -> np.ndarray:
    """Bring each channel family to ~O(1). aws_* are already normalised in the
    builder; the rendered MSG/ALARO bytes go to [0,1]; SST/static get sensible
    scales; radar stays in mm/h (the model predicts mm/h).

    Any change here changes what every pre-rendered shard means — bump
    NORMALISE_VERSION above so stale shard stores are rejected loudly."""
    if name.startswith("aws_"):
        return arr
    if name == "msg_rdt":
        return arr                      # already a 0..1 coverage fraction
    if name.startswith(("msg_", "alaro_")):
        return arr / 255.0              # rendered grayscale byte → [0,1]
    if name == "sst":
        return (arr - 10.0) / 10.0      # °C → ~O(1)
    if name == "static_elevation_m":
        return arr / 500.0
    if name == "static_distance_km":
        return arr / 100.0
    if name == "static_landmask":
        return arr                      # already 0/1
    return arr


@dataclass(frozen=True)
class _Sample:
    issue_idx: int
    lead_min: int
    lead_idx: int
    history_idx: tuple[int, ...]
    target_idx: int
    issue_epoch: int


class ZarrCorrectionDataset(Dataset):
    def __init__(
        self,
        zarr_path: str | pathlib.Path,
        *,
        time_range: tuple[datetime, datetime] | None = None,
        leads_min: tuple[int, ...] = DEFAULT_LEADS,
        history_steps: int = RADAR_HISTORY_STEPS,
        history_step_min: int | None = None,   # None → auto-detect cadence
        aux_channels: list[str] | None = None,  # None → auto-detect
        include_static: bool = True,
        require_rain_fraction: float | None = None,
        history_tolerance_s: int = 150,
        build_index: bool = True,   # False → inference mode (helpers only, no sample index)
        expected_channels: int | None = None,  # None → also check PLUVIO_EXPECTED_CHANNELS
        lagrangian_channels: int = 0,  # 0 (off, default) / 1 (advected analysis) / 2 (+ flow mag)
    ):
        if lagrangian_channels not in range(LAGRANGIAN_MAX_CHANNELS + 1):
            raise ValueError(
                f"lagrangian_channels must be 0..{LAGRANGIAN_MAX_CHANNELS}, "
                f"got {lagrangian_channels!r}"
            )
        self.lagrangian_channels = int(lagrangian_channels)
        self.zarr_path = str(zarr_path)
        self.time_range = time_range
        self.leads_min = tuple(leads_min)
        self.history_steps = history_steps
        self.require_rain_fraction = require_rain_fraction
        self.history_tolerance_s = history_tolerance_s
        self._store = None  # opened lazily per worker process
        self._pid = None

        root = self._open()
        # The working grid comes from the STORE, not the global geo.GRID: a
        # 192x192 store trained on a box with GRID left at its 100x100 default
        # crashed on buffer broadcast and silently dropped the static channels
        # (shape mismatch in _discover). The store is the source of truth.
        self.grid_hw: tuple[int, int] = tuple(int(x) for x in root["radar"].shape[-2:])
        self._issue_epoch = np.asarray(root["issue_time"][:], dtype="int64")
        _assert_epoch_seconds(self._issue_epoch, f"zarr_dataset: opening {self.zarr_path!r}")
        self._zarr_leads = [int(x) for x in np.asarray(root["leads_min"][:])]
        self._lead_to_idx = {l: i for i, l in enumerate(self._zarr_leads)}

        # Cadence (modal gap) → default history step.
        order = np.argsort(self._issue_epoch)
        self._sorted_epoch = self._issue_epoch[order]
        gaps = np.diff(self._sorted_epoch)
        cadence_s = int(np.median(gaps)) if len(gaps) else 1800
        self.history_step_min = history_step_min or max(1, round(cadence_s / 60))
        self._epoch_to_idx = {int(e): i for i, e in enumerate(self._issue_epoch)}

        # Lagrangian flow: search radius from the store's own grid spacing
        # (same derivation as the benchmark's advection baseline, so the input
        # channel and the baseline it has to beat see the same motion), and a
        # per-issue cache of the 4x4 BLOCK flow — 128 bytes an issue, so the
        # whole split fits and every lead of an issue reuses one estimate.
        attrs = dict(root.attrs)
        spacing = km_per_px_from_bounds(attrs.get("bounds"), attrs.get("grid_n"))
        self.km_per_px = _LEGACY_KM_PER_PX if spacing is None else spacing
        self.lagrangian_max_shift = max_shift_px(self.km_per_px, self.history_step_min)
        self._flow_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Aux + static channels (auto-detect from the store unless given).
        # v2 curriculum: a "truth" array (build_zarr --truth rtcor|qpe) becomes
        # the target; the operational radar stays an input. Auto-detected.
        self._has_truth = "truth" in set(root.array_keys())
        self.aux_channels = (aux_channels if aux_channels is not None
                             else self._discover(root, per_issue=True))
        self.static_channels = (self._discover(root, per_issue=False)
                                if include_static else [])
        self._static_cache: dict[str, np.ndarray] | None = None

        LOG.info("zarr_dataset: resolved channels — aux=%s static=%s lagrangian=%d "
                 "(%d total)", self.aux_channels, self.static_channels,
                 self.lagrangian_channels, self.n_channels)
        if self.lagrangian_channels:
            LOG.info("zarr_dataset: Lagrangian channels ON — grid ~%.2f km/px, flow "
                     "search radius %d px per %d min step",
                     self.km_per_px, self.lagrangian_max_shift, self.history_step_min)
        expected = expected_channels
        if expected is None:
            env_expected = os.environ.get("PLUVIO_EXPECTED_CHANNELS")
            expected = int(env_expected) if env_expected else None
        if expected is not None and self.n_channels != expected:
            raise ValueError(
                f"zarr_dataset: store {self.zarr_path!r} resolves to "
                f"{self.n_channels} channels, expected {expected} "
                f"(aux={self.aux_channels}, static={self.static_channels})"
            )

        self.index: list[_Sample] = []
        if build_index:
            self._build_index(root)

    # ───────────────────────────────────────────────── store / discovery

    def _open(self):
        """Open the zarr group, re-opening in each worker process (zarr handles
        don't survive a fork)."""
        pid = os.getpid()
        if self._store is None or self._pid != pid:
            import zarr
            self._store = zarr.open_group(self.zarr_path, mode="r")
            self._pid = pid
        return self._store

    def _discover(self, root, *, per_issue: bool) -> list[str]:
        """Classify every array in the store as a per-issue aux channel (3-D,
        shape (n_issues, *grid_hw)) or a static channel (2-D, shape
        grid_hw) — anything that doesn't match one of those two shapes is a
        contract violation, not something to silently drop or misfile (1.12:
        `_discover` used to admit any n-length 3-D array as aux and drop any
        mis-shaped static without a word)."""
        n = len(self._issue_epoch)
        out = []
        for name in sorted(root.array_keys()):
            if name in _NON_AUX:
                continue
            shape = tuple(root[name].shape)
            if name.startswith("static_") and len(shape) != 2:
                raise ValueError(
                    f"zarr_dataset: {name!r} looks static (name prefix) but has "
                    f"{len(shape)}-D shape {shape}; expected 2-D {self.grid_hw}"
                )
            if len(shape) == 3:
                if shape != (n, *self.grid_hw):
                    raise ValueError(
                        f"zarr_dataset: aux array {name!r} has shape {shape}, "
                        f"expected ({n}, {self.grid_hw[0]}, {self.grid_hw[1]}) "
                        "for a per-issue channel"
                    )
                if per_issue:
                    out.append(name)
            elif len(shape) == 2:
                if shape != self.grid_hw:
                    raise ValueError(
                        f"zarr_dataset: static array {name!r} has shape {shape}, "
                        f"expected {self.grid_hw}"
                    )
                if not per_issue:
                    out.append(name)
            else:
                raise ValueError(
                    f"zarr_dataset: array {name!r} has unsupported ndim "
                    f"{len(shape)} (shape {shape}); expected a 2-D static "
                    "channel or a 3-D per-issue channel"
                )
        return sorted(out)

    @property
    def n_channels(self) -> int:
        # history (K) + nowcast(1) + lead(1) + sin(1) + cos(1) + aux + static
        # + the opt-in Lagrangian planes, appended last so turning them on
        # never renumbers an existing channel.
        return (self.history_steps + 4 + len(self.aux_channels)
                + len(self.static_channels) + self.lagrangian_channels)

    def __len__(self) -> int:
        return len(self.index)

    # ───────────────────────────────────────────────────────── indexing

    def _lookup(self, epoch: int) -> int | None:
        """Index of the issue-time at ``epoch`` (exact, else nearest within
        history_tolerance_s)."""
        hit = self._epoch_to_idx.get(epoch)
        if hit is not None:
            return hit
        pos = int(np.searchsorted(self._sorted_epoch, epoch))
        for cand in (pos, pos - 1):
            if 0 <= cand < len(self._sorted_epoch):
                if abs(int(self._sorted_epoch[cand]) - epoch) <= self.history_tolerance_s:
                    return self._epoch_to_idx[int(self._sorted_epoch[cand])]
        return None

    def _build_index(self, root) -> None:
        step = self.history_step_min * 60
        radar = root["radar"]
        rng = None
        if self.time_range is not None:
            rng = (int(self.time_range[0].timestamp()),
                   int(self.time_range[1].timestamp()))
        n_missing_hist = n_missing_tgt = n_dry = 0

        for issue_idx in range(len(self._issue_epoch)):
            issue_e = int(self._issue_epoch[issue_idx])
            if rng and not (rng[0] <= issue_e < rng[1]):
                continue
            # history: K lead-0 analyses stepping back, newest = issue itself
            hist = []
            ok = True
            for k in range(self.history_steps - 1, -1, -1):
                hi = self._lookup(issue_e - k * step)
                if hi is None:
                    ok = False
                    break
                hist.append(hi)
            if not ok:
                n_missing_hist += 1
                continue
            for lead in self.leads_min:
                if lead not in self._lead_to_idx:
                    continue
                tgt = self._lookup(issue_e + lead * 60)
                if tgt is None:
                    n_missing_tgt += 1
                    continue
                if self._has_truth:
                    t_arr = np.asarray(root["truth"][tgt])
                    if not np.isfinite(t_arr).any():   # truth source missing here
                        n_missing_tgt += 1
                        continue
                else:
                    t_arr = None
                if self.require_rain_fraction is not None:
                    field = t_arr if t_arr is not None else radar[tgt, 0]
                    frac = float(np.mean(np.nan_to_num(field) >= RAIN_THRESHOLD))
                    if frac < self.require_rain_fraction:
                        n_dry += 1
                        continue
                self.index.append(_Sample(
                    issue_idx=issue_idx, lead_min=lead,
                    lead_idx=self._lead_to_idx[lead],
                    history_idx=tuple(hist), target_idx=tgt, issue_epoch=issue_e))

        LOG.info(
            "indexed %d samples | history step=%dmin leads=%s | dropped "
            "%d no-history, %d no-target, %d too-dry | aux=%d static=%d → %d channels",
            len(self.index), self.history_step_min, self.leads_min,
            n_missing_hist, n_missing_tgt, n_dry,
            len(self.aux_channels), len(self.static_channels), self.n_channels,
        )
        if not self.index:
            raise RuntimeError("empty index — check time_range / leads / cadence.")

    # ───────────────────────────────────────────────── input assembly (shared)

    def build_input(self, issue_idx: int, lead_min: int,
                    history_idx: tuple[int, ...]) -> np.ndarray:
        """Assemble the (n_channels, H, W) model input for one (issue, lead).
        Shared by training (__getitem__) and live inference (infer_latest)."""
        root = self._open()
        radar = root["radar"]
        H = self.history_steps
        lead_idx = self._lead_to_idx[lead_min]
        chans = np.empty((self.n_channels, *self.grid_hw), dtype="float32")

        issue_block = np.asarray(radar[issue_idx])             # (n_lead, H, W)
        for i, hidx in enumerate(history_idx):
            chans[i] = (issue_block[0] if hidx == issue_idx
                        else np.asarray(radar[hidx, 0]))
        chans[H] = issue_block[lead_idx]                       # operational nowcast @ lead
        chans[H + 1] = lead_min / 120.0
        valid = (datetime.fromtimestamp(int(self._issue_epoch[issue_idx]), tz=timezone.utc)
                 + timedelta(minutes=lead_min))
        hour = valid.hour + valid.minute / 60.0
        chans[H + 2] = np.sin(2 * np.pi * hour / 24)
        chans[H + 3] = np.cos(2 * np.pi * hour / 24)

        c = H + 4
        for name in self.aux_channels:
            chans[c] = _normalise(name, np.asarray(root[name][issue_idx]))
            c += 1
        if self.static_channels:
            if self._static_cache is None:
                self._static_cache = {n: _normalise(n, np.asarray(root[n][:]))
                                      for n in self.static_channels}
            for name in self.static_channels:
                chans[c] = self._static_cache[name]
                c += 1
        if self.lagrangian_channels:
            for plane in self._lagrangian_planes(issue_idx, lead_min, history_idx,
                                                 issue_block[0]):
                chans[c] = plane
                c += 1
        np.nan_to_num(chans, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return chans

    # ────────────────────────────────────────────── Lagrangian channels (2.3)

    def issue_block_flow(self, issue_idx: int,
                         history_idx: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Per-block (BLOCKS x BLOCKS) displacement from the two NEWEST history
        frames of this issue, in px per ``history_step_min``.

        Cached per issue: the flow depends only on that frame pair, never on
        the lead, so every lead of an issue must reuse one estimate (the
        estimate is ~150 ms at 192**2 — recomputing it per lead would be the
        dominant cost of a sample). The BLOCK field is what's cached, not the
        upsampled (2, H, W) one: 4x4x2 float32 is 128 bytes an issue, so the
        cache can hold a whole split, while the full field is ~300 kB and
        would force an LRU that misses under a shuffled loader.

        A single-step history (``history_steps == 1``) has no frame pair, so
        the flow is zero — Lagrangian persistence degenerates to persistence,
        which is the honest answer rather than a guess.
        """
        hit = self._flow_cache.get(issue_idx)
        if hit is not None:
            return hit
        root = self._open()
        radar = root["radar"]
        prev_idx = history_idx[-2] if len(history_idx) >= 2 else None
        if prev_idx is None:
            zeros = np.zeros((BLOCKS, BLOCKS), dtype="float32")
            flow = (zeros, zeros)
        else:
            prev = np.nan_to_num(np.asarray(radar[prev_idx, 0], dtype="float32"))
            curr = np.nan_to_num(np.asarray(radar[issue_idx, 0], dtype="float32"))
            # subpixel=False deliberately: this must be the SAME motion the
            # benchmark's advection baseline sees (tools/_advection uses the
            # default), and the parabolic refinement puts up to 0.5 px of
            # spurious offset on an exact match (the score surface either
            # side of a perfect peak is not symmetric), which the lead scaling
            # then multiplies into a visible drift.
            vy, vx, _valid = block_flow(prev, curr, max_shift=self.lagrangian_max_shift)
            flow = (vy, vx)
        self._flow_cache[issue_idx] = flow
        return flow

    def _lagrangian_planes(self, issue_idx: int, lead_min: int,
                           history_idx: tuple[int, ...],
                           latest: np.ndarray) -> list[np.ndarray]:
        """The opt-in Lagrangian input planes for one (issue, lead):

        1. ``lagrangian_rate`` — the latest radar analysis advected to this
           lead by the issue's own flow, scaled linearly in time
           (``lead_min / history_step_min`` steps) and clamped to the grid by
           ``motion.warp``'s coordinate clip. mm/h, same units as the radar
           history planes, so the net can learn a residual on a
           correctly-displaced prior instead of rediscovering advection.
        2. ``lagrangian_flow_mag`` (only with ``lagrangian_channels == 2``) —
           the per-step displacement magnitude in px, divided by the search
           radius so it lands in ~[0, 1]. Deliberately LEAD-INDEPENDENT (a
           property of the issue, not of the lead): it tells the net how far
           this plane's prior was transported, i.e. how much to trust it, and
           marks the blocks where the estimator found no motion at all.
        """
        vy, vx = self.issue_block_flow(issue_idx, history_idx)
        flow = upsample_flow(vy, vx, self.grid_hw)
        scale = lead_min / self.history_step_min if self.history_step_min else 0.0
        planes = [advect_with_nan(latest, scale * flow[0], scale * flow[1])]
        if self.lagrangian_channels >= 2:
            mag = np.hypot(flow[0], flow[1]) / max(self.lagrangian_max_shift, 1)
            planes.append(mag.astype("float32"))
        return planes

    def channel_recipe(self) -> dict:
        """How this dataset assembles ``build_input``, for the checkpoint.

        Saved by train.py so ``infer_latest`` can rebuild the SAME input
        instead of re-deriving it from whatever the store happens to hold
        now: a store that later gains an aux channel would otherwise silently
        shift every channel index under a trained model. ``in_channels`` alone
        catches only the count.
        """
        return {
            "history_steps": int(self.history_steps),
            "history_step_min": int(self.history_step_min),
            "aux_channels": list(self.aux_channels),
            "static_channels": list(self.static_channels),
            "lagrangian_channels": int(self.lagrangian_channels),
            "n_channels": int(self.n_channels),
        }

    def latest_issue_idx(self) -> int:
        """Index of the most recent issue-time (for live inference)."""
        return int(np.argmax(self._issue_epoch))

    def history_for(self, issue_idx: int) -> tuple[int, ...] | None:
        """Compute the radar-history indices for an arbitrary issue, or None if
        a required past frame is missing (same logic as the training indexer)."""
        step = self.history_step_min * 60
        issue_e = int(self._issue_epoch[issue_idx])
        hist: list[int] = []
        for k in range(self.history_steps - 1, -1, -1):
            hi = self._lookup(issue_e - k * step)
            if hi is None:
                return None
            hist.append(hi)
        return tuple(hist)

    # ───────────────────────────────────────────────────────── __getitem__

    def build_target(self, target_idx: int) -> np.ndarray:
        """Assemble the (1, H, W) target for one target issue-index. Shared by
        __getitem__ and the shard renderer (tools/render_shards.py) so both
        sides cannot drift apart."""
        root = self._open()
        y = (np.asarray(root["truth"][target_idx])
             if self._has_truth
             else np.asarray(root["radar"][target_idx, 0]))[None, ...].astype("float32")
        np.nan_to_num(y, copy=False, nan=0.0)
        return y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.index[idx]
        chans = self.build_input(s.issue_idx, s.lead_min, s.history_idx)
        y = self.build_target(s.target_idx)
        return torch.from_numpy(chans), torch.from_numpy(y)


def issue_time_split(zarr_path: str | pathlib.Path, val_frac: float) -> datetime:
    """Time boundary with the most-recent ``val_frac`` of issue-times held out."""
    import zarr
    root = zarr.open_group(str(zarr_path), mode="r")
    epochs = np.sort(np.asarray(root["issue_time"][:], dtype="int64"))
    _assert_epoch_seconds(epochs, f"zarr_dataset: opening {zarr_path!r}")
    cut = epochs[int(len(epochs) * (1.0 - val_frac))]
    return datetime.fromtimestamp(int(cut), tz=timezone.utc)
