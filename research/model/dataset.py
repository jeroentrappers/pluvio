"""PyTorch Dataset for the residual-correction model.

**Radar-only v1.** This deliberately uses only the inputs that need no
GeoTIFF reprojection — radar observation history + the operational nowcast
frame — so we can train an end-to-end model *today* on the KNMI HDF5 we
already know how to parse. The auxiliary channels (Meteosat, ALARO, AWS)
slot in later by extending ``_assemble_input`` and bumping ``n_channels``;
the rest of the pipeline (train loop, eval) is unchanged.

Channel layout (radar-only): 10 channels on a 100×100 grid

    0-5  radar observation history at issue-25 … issue-0 min (6 frames)
    6    operational nowcast frame at the target lead
    7    lead plane: lead_min / 120, broadcast (lead-conditioning)
    8    sin(2π·hour/24)  broadcast
    9    cos(2π·hour/24)  broadcast

Target: the verifying observation at issue_time + lead, 100×100, mm/h.

One sample = one (issue_time, lead_min) pair. The operational forecast file
holds all 25 lead steps, so we cache it (LRU) to avoid re-reading the HDF5
for every lead of the same issue time.
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
RADAR_HISTORY_STEPS = 6  # 30 min of 5-min observations
HISTORY_STEP_MIN = 5
TRAIN_LEADS_MIN: tuple[int, ...] = tuple(range(5, 125, 5))  # skip lead 0 (degenerate)
N_CHANNELS = RADAR_HISTORY_STEPS + 1 + 1 + 2  # history + nowcast + lead + 2×time


@functools.lru_cache(maxsize=64)
def _cached_forecast(path_str: str) -> kpi.ForecastRun:
    return kpi.load_forecast_h5(pathlib.Path(path_str))


@functools.lru_cache(maxsize=512)
def _cached_observation(path_str: str) -> np.ndarray:
    return kpi.load_observation_h5(pathlib.Path(path_str)).field


@dataclass(frozen=True)
class Sample:
    issue_time: datetime
    lead_min: int
    forecast_path: pathlib.Path
    history_paths: tuple[pathlib.Path, ...]  # oldest → newest, len RADAR_HISTORY_STEPS
    target_path: pathlib.Path


class PluvioCorrectionDataset(Dataset):
    def __init__(
        self,
        data_root: pathlib.Path,
        *,
        time_range: tuple[datetime, datetime] | None = None,
        leads_min: tuple[int, ...] = TRAIN_LEADS_MIN,
        require_rain_fraction: float | None = None,
    ):
        """
        Args:
            data_root: directory holding ``radar_forecast/2.0`` and
                ``nl_rdr_data_rtcor_5m/1.0`` subdirs (the collector layout).
            time_range: optional (start, end) UTC filter on issue times —
                used to make disjoint train / val splits.
            leads_min: which forecast lead steps to emit samples for.
            require_rain_fraction: if set, drop samples whose target has a
                wet-cell fraction below this threshold. Cheap way to focus
                CPU training on the ~5% of timesteps that actually have rain.
        """
        self.data_root = pathlib.Path(data_root)
        self.leads_min = leads_min
        self.time_range = time_range
        self.require_rain_fraction = require_rain_fraction
        self.index: list[Sample] = []
        self._build_index()

    @property
    def n_channels(self) -> int:
        return N_CHANNELS

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.index[idx]
        x = self._assemble_input(s)
        target = _cached_observation(str(s.target_path))
        y = np.nan_to_num(target, nan=0.0).astype("float32")[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y)

    # ──────────────────────────────────────────────────────────── assembly

    def _assemble_input(self, s: Sample) -> np.ndarray:
        channels: list[np.ndarray] = []

        # 0-5: radar history, oldest → newest.
        for p in s.history_paths:
            channels.append(np.nan_to_num(_cached_observation(str(p)), nan=0.0))

        # 6: operational nowcast frame at this lead.
        run = _cached_forecast(str(s.forecast_path))
        lead_idx = int(np.argmin(np.abs(run.leads_min - s.lead_min)))
        channels.append(np.nan_to_num(run.frames[lead_idx], nan=0.0))

        # 7: lead plane (conditioning).
        channels.append(np.full(GRID, s.lead_min / 120.0, dtype="float32"))

        # 8-9: time-of-day encoding.
        valid = s.issue_time + timedelta(minutes=s.lead_min)
        hour = valid.hour + valid.minute / 60.0
        channels.append(np.full(GRID, np.sin(2 * np.pi * hour / 24), dtype="float32"))
        channels.append(np.full(GRID, np.cos(2 * np.pi * hour / 24), dtype="float32"))

        return np.stack(channels, axis=0).astype("float32")

    # ──────────────────────────────────────────────────────────── indexing

    def _build_index(self) -> None:
        fc_dir = self.data_root / "radar_forecast" / "2.0"
        obs_dir = self.data_root / "nl_rdr_data_rtcor_5m" / "1.0"
        if not fc_dir.exists() or not obs_dir.exists():
            raise FileNotFoundError(
                f"expected {fc_dir} and {obs_dir}; run the KNMI collectors first."
            )

        obs_by_ts: dict[datetime, pathlib.Path] = {
            _ts_from_name(p): p for p in kpi.discover_observations(obs_dir)
        }

        forecasts = kpi.discover_runs(fc_dir)
        n_considered = n_missing = n_dry = 0

        for fc_path in forecasts:
            issue = _ts_from_name(fc_path)
            if self.time_range is not None:
                start, end = self.time_range
                if issue < start or issue >= end:
                    continue

            history: list[pathlib.Path] = []
            ok = True
            for k in range(RADAR_HISTORY_STEPS - 1, -1, -1):
                ts = issue - timedelta(minutes=k * HISTORY_STEP_MIN)
                if ts not in obs_by_ts:
                    ok = False
                    break
                history.append(obs_by_ts[ts])
            if not ok:
                continue

            for lead in self.leads_min:
                n_considered += 1
                target_path = obs_by_ts.get(issue + timedelta(minutes=lead))
                if target_path is None:
                    n_missing += 1
                    continue
                if self.require_rain_fraction is not None:
                    field = _cached_observation(str(target_path))
                    if float(np.mean(np.nan_to_num(field) >= 0.1)) < self.require_rain_fraction:
                        n_dry += 1
                        continue
                self.index.append(
                    Sample(
                        issue_time=issue,
                        lead_min=lead,
                        forecast_path=fc_path,
                        history_paths=tuple(history),
                        target_path=target_path,
                    )
                )

        LOG.info(
            "indexed %d samples from %d forecast files "
            "(considered %d lead-pairs, dropped %d missing-obs, %d too-dry)",
            len(self.index), len(forecasts), n_considered, n_missing, n_dry,
        )
        if not self.index:
            raise RuntimeError(
                "empty training index — check the window covers enough contiguous "
                "5-min observations around each forecast issue time."
            )


def _ts_from_name(path: pathlib.Path) -> datetime:
    m = re.search(r"(\d{12})", path.name)
    if not m:
        raise ValueError(f"no YYYYMMDDHHMM in {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
