"""PyTorch Dataset for the residual-correction model.

**Radar-only v1, forecast-file-sourced, RAM-resident.**

Every signal lives inside the KNMI ``radar_forecast`` HDF5 files (the
real-time ``rtcor`` product has an unreliable retention window; the
``_tar`` archive needs a subscription we don't have). A forecast file
issued at T holds 25 frames: ``image1`` is the lead-0 *analysis*
(radar+gauge corrected — verified byte-identical to the rtcor observation),
``image2..25`` are the +5..+120 min forecast.

Performance: parsing HDF5 in ``__getitem__`` made one CPU epoch take >1 h
(each sample touches ~8 files × 25 groups). Instead we parse every file
**once** into a single in-RAM tensor (~860 MB for a 3-day window, cached to
``frames_cache.npz`` so reruns are instant). ``__getitem__`` is then pure
array indexing — the training loop becomes compute-bound, not I/O-bound.

Channel layout (10 channels, 100×100 grid):

    0-5  radar analysis history at issue-25 … issue-0 min (6 frames)
    6    operational nowcast frame at the target lead
    7    lead plane: lead_min / 120, broadcast (lead-conditioning)
    8    sin(2π·hour/24) broadcast
    9    cos(2π·hour/24) broadcast

Target: the analysis at issue_time + lead.
"""

from __future__ import annotations

import functools
import logging
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import _lib as kpi  # noqa: E402

LOG = logging.getLogger("pluvio.dataset")

GRID = kpi.ANALYSIS_GRID  # (100, 100)
RADAR_HISTORY_STEPS = 6
HISTORY_STEP_MIN = 5
TRAIN_LEADS_MIN: tuple[int, ...] = tuple(range(5, 125, 5))  # skip lead 0 (= the analysis)
N_BASE_CHANNELS = RADAR_HISTORY_STEPS + 1 + 1 + 2  # history + nowcast + lead + 2×time = 10


@dataclass(frozen=True)
class ForecastStore:
    """All forecast frames for a window, resident in RAM.

    ``aux`` holds optional auxiliary channels aligned 1:1 with the forecast
    timestamps (same index space as ``frames``), built by ``build_aux.py``:
    e.g. AWS pressure/temp/humidity/wind, normalised to ~O(1).
    """

    frames: np.ndarray  # (n_files, n_leads, H, W) float32, NaNs zeroed
    leads_min: np.ndarray  # (n_leads,) int — typically [0, 5, …, 120]
    ts_to_idx: dict[datetime, int]
    aux: dict[str, np.ndarray]  # name → (n_files, H, W), aligned to frames index
    aux_order: tuple[str, ...]

    def analysis_idx(self, ts: datetime) -> int | None:
        return self.ts_to_idx.get(ts)


def _load_aux(fc_dir: pathlib.Path) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Load aux_cache.npz (AWS etc.) if present, else return empty."""
    aux_path = fc_dir / "aux_cache.npz"
    if not aux_path.exists():
        return {}, ()
    d = np.load(aux_path, allow_pickle=True)
    order = tuple(str(c) for c in d["channel_order"])
    aux = {ch: d[ch] for ch in order}
    LOG.info("loaded aux cache: %d channels %s", len(order), order)
    return aux, order


@functools.lru_cache(maxsize=4)
def load_store(fc_dir_str: str) -> ForecastStore:
    """Parse every forecast file into one RAM tensor, cached to npz."""
    fc_dir = pathlib.Path(fc_dir_str)
    cache = fc_dir.parent / "frames_cache.npz"
    aux, aux_order = _load_aux(fc_dir.parent)
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        ts_list = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in d["ts_epoch"]]
        LOG.info("loaded frame cache: %s (%d files)", cache.name, len(ts_list))
        return ForecastStore(
            frames=d["frames"],
            leads_min=d["leads_min"],
            ts_to_idx={ts: i for i, ts in enumerate(ts_list)},
            aux=aux,
            aux_order=aux_order,
        )

    paths = kpi.discover_runs(fc_dir)
    if not paths:
        raise FileNotFoundError(f"no forecast files under {fc_dir}")
    LOG.info("building frame cache from %d files (one-time)…", len(paths))
    frames: list[np.ndarray] = []
    ts_epoch: list[int] = []
    leads_min: np.ndarray | None = None
    for p in paths:
        run = kpi.load_forecast_h5(p)
        frames.append(np.nan_to_num(run.frames, nan=0.0).astype("float32"))
        ts_epoch.append(int(_ts_from_name(p).timestamp()))
        leads_min = run.leads_min
    stacked = np.stack(frames)  # (N, n_leads, H, W)
    np.savez_compressed(cache, frames=stacked, leads_min=leads_min, ts_epoch=np.array(ts_epoch))
    LOG.info("wrote %s (%.0f MB)", cache.name, stacked.nbytes / 1e6)
    ts_list = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts_epoch]
    return ForecastStore(
        frames=stacked,
        leads_min=leads_min,
        ts_to_idx={ts: i for i, ts in enumerate(ts_list)},
        aux=aux,
        aux_order=aux_order,
    )


@dataclass(frozen=True)
class Sample:
    issue_idx: int
    lead_min: int
    lead_idx: int
    history_idx: tuple[int, ...]
    target_idx: int
    issue_time: datetime


class PluvioCorrectionDataset(Dataset):
    def __init__(
        self,
        data_root: pathlib.Path,
        *,
        time_range: tuple[datetime, datetime] | None = None,
        leads_min: tuple[int, ...] = TRAIN_LEADS_MIN,
        require_rain_fraction: float | None = None,
    ):
        self.data_root = pathlib.Path(data_root)
        self.leads_min = leads_min
        self.time_range = time_range
        self.require_rain_fraction = require_rain_fraction
        self.store = load_store(str(self.data_root / "radar_forecast" / "2.0"))
        self.index: list[Sample] = []
        self._build_index()

    @property
    def n_channels(self) -> int:
        return N_BASE_CHANNELS + len(self.store.aux_order)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.index[idx]
        f = self.store.frames
        chans = np.empty((self.n_channels, *GRID), dtype="float32")
        for i, hidx in enumerate(s.history_idx):
            chans[i] = f[hidx, 0]
        chans[6] = f[s.issue_idx, s.lead_idx]
        chans[7] = s.lead_min / 120.0
        valid = s.issue_time + timedelta(minutes=s.lead_min)
        hour = valid.hour + valid.minute / 60.0
        chans[8] = np.sin(2 * np.pi * hour / 24)
        chans[9] = np.cos(2 * np.pi * hour / 24)
        # Aux channels: surface state at issue time, aligned to issue_idx.
        for j, name in enumerate(self.store.aux_order):
            chans[N_BASE_CHANNELS + j] = self.store.aux[name][s.issue_idx]
        y = f[s.target_idx, 0][None, ...]
        return torch.from_numpy(chans), torch.from_numpy(y.copy())

    # ──────────────────────────────────────────────────────────── indexing

    def _build_index(self) -> None:
        store = self.store
        lead_to_fidx = {int(l): i for i, l in enumerate(store.leads_min)}
        all_ts = sorted(store.ts_to_idx)
        n_considered = n_missing = n_dry = 0

        for issue in all_ts:
            if self.time_range is not None:
                start, end = self.time_range
                if issue < start or issue >= end:
                    continue

            history_idx: list[int] = []
            ok = True
            for k in range(RADAR_HISTORY_STEPS - 1, -1, -1):
                ts = issue - timedelta(minutes=k * HISTORY_STEP_MIN)
                hi = store.analysis_idx(ts)
                if hi is None:
                    ok = False
                    break
                history_idx.append(hi)
            if not ok:
                continue

            issue_idx = store.ts_to_idx[issue]
            for lead in self.leads_min:
                if lead not in lead_to_fidx:
                    continue
                n_considered += 1
                target_idx = store.analysis_idx(issue + timedelta(minutes=lead))
                if target_idx is None:
                    n_missing += 1
                    continue
                if self.require_rain_fraction is not None:
                    if float(np.mean(store.frames[target_idx, 0] >= 0.1)) < self.require_rain_fraction:
                        n_dry += 1
                        continue
                self.index.append(
                    Sample(
                        issue_idx=issue_idx,
                        lead_min=lead,
                        lead_idx=lead_to_fidx[lead],
                        history_idx=tuple(history_idx),
                        target_idx=target_idx,
                        issue_time=issue,
                    )
                )

        LOG.info(
            "indexed %d samples (considered %d lead-pairs, dropped %d missing, %d too-dry)",
            len(self.index), n_considered, n_missing, n_dry,
        )
        if not self.index:
            raise RuntimeError("empty index — window lacks contiguous 5-min forecast files.")


def _ts_from_name(path: pathlib.Path) -> datetime:
    m = re.search(r"(\d{12})", path.name)
    if not m:
        raise ValueError(f"no YYYYMMDDHHMM in {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
