"""Dataset for the seamless model — reads the unified seamless zarr.

Successor to `ZarrCorrectionDataset` for the 0 → 240 h seamless net
(docs/seamless_model_plan.md). Differences:
  * **truth = OPERA** (`opera_rate`), not KNMI radar;
  * a **lead-dependent AIFS channel** (the NWP field at the target valid-time)
    so the outlook head can downscale it;
  * **leads to 240 h** with a smooth lead/time conditioning vector (lifts the old
    `lead/120` hard cap);
  * `__getitem__` returns ``(x, cond, y)`` — the cond vector drives FiLM.

Expected store (built by tools/build_seamless_zarr.py), keyed by issue-time:
    issue_time     (n,)              int64 epoch
    leads_min      (n_lead,)         int16   (0,10,…,120, then hourly, 3-hourly…)
    opera_rate     (n, H, W)         mm/h    OPERA analysis at each issue-time (truth)
    aifs_tp        (n, n_lead, H, W) mm/h    AIFS forecast cube: [i, j] = AIFS at issue_i + lead_j
    <obs aux>      (n, H, W)         per-issue 2-D channels (li_flash, gii_*, ctth_*, …)
    static_*       (H, W)            elevation / landmask / dist-to-coast

OPERA is an *analysis* (one field per issue-time), so history = the K most-recent
opera_rate analyses stepping back; target = opera_rate at issue+lead (the
analysis when that time is itself an issue-time — dense with 15-min OPERA). The
per-lead *forecast* anchor the outlook head downscales is AIFS (`aifs_tp[i, j]`).
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset

from model.geo import GRID  # honour PLUVIO_GRID_N (rec #5) — not a hardcoded 100²
from model.seamless import lead_time_encoding

LOG = logging.getLogger("pluvio.seamless_dataset")

RADAR_HISTORY_STEPS = 6
# 0–2 h @ 10 min, to 24 h hourly, to 240 h 3-hourly.
DEFAULT_LEADS: tuple[int, ...] = (
    tuple(range(0, 121, 10)) + tuple(range(180, 1441, 60)) + tuple(range(1440 + 180, 14401, 180))
)
RAIN_THRESHOLD = 0.1
TRUTH_VAR = "opera_rate"
NWP_VAR = "aifs_tp"
_NON_AUX = {TRUTH_VAR, NWP_VAR, "issue_time", "leads_min"}


def _normalise(name: str, arr: np.ndarray) -> np.ndarray:
    """Bring each channel family to ~O(1). Real-valued products (no rendered
    bytes here): scale by physically sensible constants."""
    if name in (TRUTH_VAR, NWP_VAR) or name.startswith(("li_", "opera_", "era5_tp", "om_")):
        return arr  # mm/h — model predicts mm/h (om_* = Open-Meteo NWP precip)
    # ERA5 predictor channels → ~O(1) by physically sensible constants.
    if name == "era5_cape":
        return arr / 1000.0
    if name == "era5_tcwv":
        return arr / 30.0
    if name == "era5_t2m":
        return (arr - 288.0) / 15.0
    if name == "era5_msl":
        return (arr - 101325.0) / 2000.0
    if name in ("era5_u10", "era5_v10"):
        return arr / 10.0
    if name.startswith("gii_"):
        return arr / 10.0
    if name.startswith(("ctth_", "cloud_top_temperature")):
        return (arr - 250.0) / 50.0
    if name.startswith(("oca_", "olr")):
        return arr / 100.0
    if name == "static_elevation_m":
        return arr / 500.0
    if name == "static_distance_km":
        return arr / 100.0
    return arr


@dataclass(frozen=True)
class _Sample:
    issue_idx: int
    lead_min: int
    lead_idx: int
    history_idx: tuple[int, ...]
    target_idx: int


class SeamlessDataset(Dataset):
    def __init__(self, zarr_path, *, time_range=None, leads_min=DEFAULT_LEADS,
                 history_steps=RADAR_HISTORY_STEPS, history_step_min=None,
                 aux_channels=None, include_static=True, require_rain_fraction=None,
                 history_tolerance_s=300, build_index=True, aux_at_valid_time=False):
        self.zarr_path = str(zarr_path)
        self.time_range = time_range
        self.leads_min = tuple(leads_min)
        self.history_steps = history_steps
        self.require_rain_fraction = require_rain_fraction
        # Stage B: read the NWP/aux anchor at the *valid* time (issue+lead) rather
        # than the issue time — i.e. ERA5-at-valid-time as a perfect-forecast
        # proxy the outlook head downscales (AIFS swaps in here at inference).
        self.aux_at_valid_time = aux_at_valid_time
        self.history_tolerance_s = history_tolerance_s
        self._store = None
        self._pid = None

        root = self._open()
        self._issue_epoch = np.asarray(root["issue_time"][:], dtype="int64")
        self._zarr_leads = [int(x) for x in np.asarray(root["leads_min"][:])]
        self._lead_to_idx = {l: i for i, l in enumerate(self._zarr_leads)}
        order = np.argsort(self._issue_epoch)
        self._sorted_epoch = self._issue_epoch[order]
        gaps = np.diff(self._sorted_epoch)
        cadence_s = int(np.median(gaps)) if len(gaps) else 900
        self.history_step_min = history_step_min or max(1, round(cadence_s / 60))
        self._epoch_to_idx = {int(e): i for i, e in enumerate(self._issue_epoch)}

        self.has_aifs = NWP_VAR in list(root.array_keys())  # baseline zarr may omit AIFS
        self.aux_channels = aux_channels if aux_channels is not None else self._discover(root, True)
        self.static_channels = self._discover(root, False) if include_static else []
        self._static_cache = None
        self.index: list[_Sample] = []
        if build_index:
            self._build_index(root)

    def _open(self):
        pid = os.getpid()
        if self._store is None or self._pid != pid:
            import zarr
            self._store = zarr.open_group(self.zarr_path, mode="r")
            self._pid = pid
        return self._store

    def _discover(self, root, per_issue: bool) -> list[str]:
        n = len(self._issue_epoch)
        out = []
        for name in root.array_keys():
            if name in _NON_AUX:
                continue
            shape = root[name].shape
            if per_issue and len(shape) == 3 and shape[0] == n and not name.startswith("static_"):
                out.append(name)
            elif (not per_issue) and len(shape) == 2 and tuple(shape) == GRID:
                out.append(name)
        return sorted(out)

    @property
    def n_channels(self) -> int:
        # history(K) + AIFS forecast anchor @ lead(1, if present) + aux + static
        return (self.history_steps + (1 if self.has_aifs else 0)
                + len(self.aux_channels) + len(self.static_channels))

    def __len__(self) -> int:
        return len(self.index)

    def _lookup(self, epoch: int):
        hit = self._epoch_to_idx.get(epoch)
        if hit is not None:
            return hit
        pos = int(np.searchsorted(self._sorted_epoch, epoch))
        for cand in (pos, pos - 1):
            if 0 <= cand < len(self._sorted_epoch) and abs(int(self._sorted_epoch[cand]) - epoch) <= self.history_tolerance_s:
                return self._epoch_to_idx[int(self._sorted_epoch[cand])]
        return None

    def _build_index(self, root) -> None:
        step = self.history_step_min * 60
        truth = root[TRUTH_VAR]
        rng = (int(self.time_range[0].timestamp()), int(self.time_range[1].timestamp())) if self.time_range else None
        for issue_idx in range(len(self._issue_epoch)):
            issue_e = int(self._issue_epoch[issue_idx])
            if rng and not (rng[0] <= issue_e < rng[1]):
                continue
            hist, ok = [], True
            for k in range(self.history_steps - 1, -1, -1):
                hi = self._lookup(issue_e - k * step)
                if hi is None:
                    ok = False
                    break
                hist.append(hi)
            if not ok:
                continue
            for lead in self.leads_min:
                if lead not in self._lead_to_idx:
                    continue
                tgt = self._lookup(issue_e + lead * 60)
                if tgt is None:
                    continue
                if self.require_rain_fraction is not None:
                    if float(np.mean(truth[tgt] >= RAIN_THRESHOLD)) < self.require_rain_fraction:
                        continue
                self.index.append(_Sample(issue_idx, lead, self._lead_to_idx[lead], tuple(hist), tgt))
        LOG.info("indexed %d samples | leads %d..%d | %d channels",
                 len(self.index), self.leads_min[0], self.leads_min[-1], self.n_channels)
        if not self.index:
            raise RuntimeError("empty index — check time_range / leads / cadence")

    def build_input(self, issue_idx, lead_min, history_idx, target_idx=None):
        """(n_channels, H, W) input — shared by training and live inference."""
        root = self._open()
        truth = root[TRUTH_VAR]          # (n, H, W) OPERA analysis
        H = self.history_steps
        lead_idx = self._lead_to_idx[lead_min]
        # outlook anchor index: the valid-time (target) when aux_at_valid_time,
        # else the issue time (Stage-A downscaling / live-obs aux).
        aux_idx = target_idx if (self.aux_at_valid_time and target_idx is not None) else issue_idx
        chans = np.empty((self.n_channels, *GRID), dtype="float32")
        for i, hidx in enumerate(history_idx):
            chans[i] = np.asarray(truth[hidx])                       # past OPERA analyses
        c = H
        if self.has_aifs:
            chans[c] = _normalise(NWP_VAR, np.asarray(root[NWP_VAR][issue_idx, lead_idx]))  # AIFS forecast @ lead
            c += 1
        for name in self.aux_channels:
            chans[c] = _normalise(name, np.asarray(root[name][aux_idx])); c += 1
        if self.static_channels:
            if self._static_cache is None:
                self._static_cache = {n: _normalise(n, np.asarray(root[n][:])) for n in self.static_channels}
            for name in self.static_channels:
                chans[c] = self._static_cache[name]; c += 1
        np.nan_to_num(chans, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return chans

    def cond_for(self, issue_idx, lead_min) -> np.ndarray:
        valid = datetime.fromtimestamp(int(self._issue_epoch[issue_idx]), tz=timezone.utc)
        valid_min = lead_min
        doy = valid.timetuple().tm_yday
        hour = valid.hour + valid.minute / 60.0 + lead_min / 60.0
        cond = lead_time_encoding(torch.tensor([float(valid_min)]), torch.tensor([hour % 24]),
                                  torch.tensor([float(doy)]))
        return cond.numpy()[0]

    def latest_issue_idx(self) -> int:
        return int(np.argmax(self._issue_epoch))

    def __getitem__(self, idx):
        s = self.index[idx]
        x = self.build_input(s.issue_idx, s.lead_min, s.history_idx, target_idx=s.target_idx)
        cond = self.cond_for(s.issue_idx, s.lead_min)
        y = np.asarray(self._open()[TRUTH_VAR][s.target_idx])[None, ...].astype("float32")
        np.nan_to_num(y, copy=False, nan=0.0)
        return torch.from_numpy(x), torch.from_numpy(cond).float(), torch.from_numpy(y)


def issue_time_split(zarr_path, val_frac: float):
    """Time boundary holding out the most-recent ``val_frac`` of issue-times."""
    import zarr
    from datetime import datetime, timezone
    root = zarr.open_group(str(zarr_path), mode="r")
    epochs = np.sort(np.asarray(root["issue_time"][:], dtype="int64"))
    cut = epochs[int(len(epochs) * (1.0 - val_frac))]
    return datetime.fromtimestamp(int(cut), tz=timezone.utc)
