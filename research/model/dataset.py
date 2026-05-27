"""PyTorch Dataset for the residual-correction model.

**Radar-only v1, forecast-file-sourced.** Every signal we need lives inside
the KNMI ``radar_forecast`` HDF5 files, so this is the *only* product we
depend on (the real-time ``rtcor`` observation product has an unreliable
retention window; the ``_tar`` archive needs a subscription we don't have):

- A forecast file issued at T holds 25 frames: ``image1`` is the lead-0
  *analysis* (radar+gauge corrected — verified byte-identical to the rtcor
  observation), ``image2..25`` are the +5..+120 min forecast.
- So the "observation at time T" is simply ``forecast(T).frames[0]``.
- History, operational nowcast, and verifying truth all come from the
  appropriate forecast files at 5-min cadence.

Channel layout (10 channels, 100×100 grid):

    0-5  radar analysis history at issue-25 … issue-0 min (6 frames, =image1)
    6    operational nowcast frame at the target lead (=image at that lead)
    7    lead plane: lead_min / 120, broadcast (lead-conditioning)
    8    sin(2π·hour/24) broadcast
    9    cos(2π·hour/24) broadcast

Target: the analysis at issue_time + lead (= image1 of that later file).

One sample = one (issue_time, lead_min) pair. Forecast HDF5 reads are
LRU-cached so we don't re-open the same file for every lead.
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
RADAR_HISTORY_STEPS = 6  # 30 min of 5-min analyses
HISTORY_STEP_MIN = 5
TRAIN_LEADS_MIN: tuple[int, ...] = tuple(range(5, 125, 5))  # skip lead 0 (= the analysis)
N_CHANNELS = RADAR_HISTORY_STEPS + 1 + 1 + 2  # history + nowcast + lead + 2×time


@functools.lru_cache(maxsize=128)
def _cached_forecast(path_str: str) -> kpi.ForecastRun:
    return kpi.load_forecast_h5(pathlib.Path(path_str))


def _analysis(path_str: str) -> np.ndarray:
    """The lead-0 analysis (= observation) from a forecast file."""
    return np.nan_to_num(_cached_forecast(path_str).frames[0], nan=0.0)


@dataclass(frozen=True)
class Sample:
    issue_time: datetime
    lead_min: int
    forecast_path: pathlib.Path  # the issue-time file (operational nowcast lives here)
    history_paths: tuple[pathlib.Path, ...]  # forecast files at issue-25..issue-0
    target_path: pathlib.Path  # forecast file at issue+lead (its image1 = truth)


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
            data_root: dir holding ``radar_forecast/2.0`` (the collector layout).
            time_range: optional (start, end) UTC filter on issue times for
                disjoint train / val splits.
            leads_min: which forecast lead steps to emit samples for.
            require_rain_fraction: if set, drop samples whose target wet-cell
                fraction is below this — focuses CPU epochs on rainy timesteps.
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
        y = _analysis(str(s.target_path)).astype("float32")[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y)

    # ──────────────────────────────────────────────────────────── assembly

    def _assemble_input(self, s: Sample) -> np.ndarray:
        channels: list[np.ndarray] = []

        # 0-5: radar analysis history, oldest → newest.
        for p in s.history_paths:
            channels.append(_analysis(str(p)))

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
        if not fc_dir.exists():
            raise FileNotFoundError(f"expected {fc_dir}; run the KNMI collector first.")

        fc_by_ts: dict[datetime, pathlib.Path] = {
            _ts_from_name(p): p for p in kpi.discover_runs(fc_dir)
        }
        all_ts = sorted(fc_by_ts)
        n_considered = n_missing = n_dry = 0

        for issue in all_ts:
            if self.time_range is not None:
                start, end = self.time_range
                if issue < start or issue >= end:
                    continue

            # History: issue-25 … issue-0, all must be present.
            history: list[pathlib.Path] = []
            ok = True
            for k in range(RADAR_HISTORY_STEPS - 1, -1, -1):
                ts = issue - timedelta(minutes=k * HISTORY_STEP_MIN)
                if ts not in fc_by_ts:
                    ok = False
                    break
                history.append(fc_by_ts[ts])
            if not ok:
                continue

            for lead in self.leads_min:
                n_considered += 1
                target_path = fc_by_ts.get(issue + timedelta(minutes=lead))
                if target_path is None:
                    n_missing += 1
                    continue
                if self.require_rain_fraction is not None:
                    if float(np.mean(_analysis(str(target_path)) >= 0.1)) < self.require_rain_fraction:
                        n_dry += 1
                        continue
                self.index.append(
                    Sample(
                        issue_time=issue,
                        lead_min=lead,
                        forecast_path=fc_by_ts[issue],
                        history_paths=tuple(history),
                        target_path=target_path,
                    )
                )

        LOG.info(
            "indexed %d samples from %d forecast files "
            "(considered %d lead-pairs, dropped %d missing, %d too-dry)",
            len(self.index), len(fc_by_ts), n_considered, n_missing, n_dry,
        )
        if not self.index:
            raise RuntimeError(
                "empty index — the window needs contiguous 5-min forecast files "
                "around each issue time (for history and the +lead target)."
            )


def _ts_from_name(path: pathlib.Path) -> datetime:
    m = re.search(r"(\d{12})", path.name)
    if not m:
        raise ValueError(f"no YYYYMMDDHHMM in {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
