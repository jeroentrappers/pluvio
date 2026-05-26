"""PyTorch Dataset that joins the four data sources onto a common grid.

For each training sample, we need a 100×100 input tensor with all the
feature channels described in ``docs/model_architecture.md`` plus a target
(the verified observation at the requested lead time).

This module focuses on the *assembly* step. The collectors land raw files
on disk; we read them lazily here, regrid them to the common grid, and
yield ``(input_tensor, target_tensor)`` pairs.

It's deliberately conservative about caching: every transformation is done
on-the-fly so the Dataset stays under ~200 lines and is easy to audit. For
production training you'd pre-bake into a zarr store via ``build_zarr.py``
(see the TODO at the bottom of this file).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from torch.utils.data import Dataset

# These imports come from the verification notebook helpers. Keep the
# Dataset module dependency-light — we re-use the existing HDF5 parsers
# rather than duplicate them.
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import _lib as kpi  # noqa: E402  KNMI parser library

GRID = kpi.ANALYSIS_GRID  # 100 × 100, the common analysis grid


@dataclass(frozen=True)
class SamplePaths:
    """File paths needed to assemble one training sample."""

    radar_history: list[pathlib.Path]  # 6 most recent observations before issue_time
    operational_forecast: pathlib.Path  # the radar_forecast HDF5 we're correcting
    observation_at_lead: pathlib.Path  # the verifying observation file
    msg_ir108: pathlib.Path | None  # nearest Meteosat IR 10.8 GeoTIFF
    msg_wv062: pathlib.Path | None
    msg_kindex: pathlib.Path | None
    msg_li: pathlib.Path | None
    msg_cth: pathlib.Path | None
    msg_rdt: pathlib.Path | None
    aws_pressure_grid: np.ndarray | None  # interpolated AWS field, 100×100
    alaro_tp: pathlib.Path | None  # ALARO total precipitation
    alaro_cc: pathlib.Path | None  # cloud cover
    alaro_wind: pathlib.Path | None  # 10-m wind speed


class PluvioCorrectionDataset(Dataset):
    """One sample = one (issue_time, lead_min) pair across all sources.

    Construction is two-phase:
      1. ``index_paths()`` scans the data directories and builds a list of
         valid ``(issue_time, lead_min, SamplePaths)`` tuples. Slow but
         only done once per epoch.
      2. ``__getitem__`` reads the actual arrays and stacks them into a
         (29, 100, 100) input tensor and a (1, 100, 100) target.
    """

    def __init__(
        self,
        data_root: pathlib.Path,
        leads_min: tuple[int, ...] = tuple(range(5, 125, 5)),
        radar_history_steps: int = 6,
    ):
        self.data_root = pathlib.Path(data_root)
        self.leads_min = leads_min
        self.radar_history_steps = radar_history_steps
        self.index: list[tuple[datetime, int, SamplePaths]] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        issue_time, lead_min, paths = self.index[idx]
        channels: list[np.ndarray] = []

        # 1. Radar history (6 channels)
        for p in paths.radar_history:
            channels.append(kpi.load_observation_h5(p).field)
        while len(channels) < self.radar_history_steps:
            channels.append(np.zeros(GRID, dtype="float32"))

        # 2. Operational forecast at this lead (1 channel)
        run = kpi.load_forecast_h5(paths.operational_forecast)
        lead_idx = int(np.argmin(np.abs(run.leads_min - lead_min)))
        channels.append(run.frames[lead_idx])

        # 3. Meteosat channels (6 channels — IR108, ΔIR108, WV062, K, LI, CTH, RDT)
        ir108 = _load_geotiff(paths.msg_ir108, GRID)
        ir108_prev = _load_geotiff_offset(paths.msg_ir108, GRID, minutes=-30)
        channels.append(ir108)
        channels.append(ir108 - ir108_prev)  # cooling trend
        channels.append(_load_geotiff(paths.msg_wv062, GRID))
        channels.append(_load_geotiff(paths.msg_kindex, GRID))
        channels.append(_load_geotiff(paths.msg_li, GRID))
        channels.append(_load_geotiff(paths.msg_cth, GRID))
        channels.append(_load_geotiff(paths.msg_rdt, GRID))

        # 4. AWS interpolated fields (5 channels)
        if paths.aws_pressure_grid is not None:
            channels.append(paths.aws_pressure_grid)
        else:
            channels.append(np.zeros(GRID, dtype="float32"))
        # TODO: same for pressure_tendency_3h, temp, humidity, wind_speed
        for _ in range(4):
            channels.append(np.zeros(GRID, dtype="float32"))

        # 5. ALARO context (4 channels) at the lead's valid time
        channels.append(_load_geotiff(paths.alaro_tp, GRID))
        channels.append(_load_geotiff(paths.alaro_cc, GRID))
        channels.append(_load_geotiff(paths.alaro_wind, GRID))
        channels.append(np.zeros(GRID, dtype="float32"))  # placeholder for ALARO humidity

        # 6. Static features (3 channels) — loaded from packaged assets
        terrain = _static_field("terrain", GRID)
        landmask = _static_field("landmask", GRID)
        coastdist = _static_field("coast_distance", GRID)
        channels.extend([terrain, landmask, coastdist])

        # 7. Time encoding (4 channels)
        valid = issue_time + timedelta(minutes=lead_min)
        hour = valid.hour + valid.minute / 60
        doy = valid.timetuple().tm_yday
        channels.append(np.full(GRID, np.sin(2 * np.pi * hour / 24), dtype="float32"))
        channels.append(np.full(GRID, np.cos(2 * np.pi * hour / 24), dtype="float32"))
        channels.append(np.full(GRID, np.sin(2 * np.pi * doy / 365.25), dtype="float32"))
        channels.append(np.full(GRID, np.cos(2 * np.pi * doy / 365.25), dtype="float32"))

        x = np.stack([_ensure(c) for c in channels], axis=0).astype("float32")
        y = _ensure(kpi.load_observation_h5(paths.observation_at_lead).field).astype("float32")

        meta = {"issue_time": issue_time.isoformat(), "lead_min": int(lead_min)}
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0), meta

    # ------------------------------------------------------------------ private

    def _build_index(self) -> None:
        # Look at every available forecast file and every lead step we care
        # about. Validate that the matching observation exists. Resolve the
        # nearest Meteosat / ALARO / AWS files (15-min and 1-h cadences).
        # TODO(pluvio): implement against the actual on-disk layout once
        # the collectors have run for at least a week.
        raise NotImplementedError(
            "PluvioCorrectionDataset._build_index — implement once the "
            "collectors have produced a contiguous training window. "
            "See docs/model_architecture.md for the source layout."
        )


# ---------------------------------------------------------------- helpers


def _load_geotiff(path: pathlib.Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(shape, dtype="float32")
    # TODO(pluvio): use rasterio to decode + reproject to the KNMI grid.
    # For now, return zeros so the rest of the pipeline can be exercised
    # without a rasterio dependency.
    return np.zeros(shape, dtype="float32")


def _load_geotiff_offset(
    path: pathlib.Path | None, shape: tuple[int, int], minutes: int
) -> np.ndarray:
    # Sister to _load_geotiff but resolves the file `minutes` ago/ahead.
    return _load_geotiff(path, shape)


def _static_field(kind: str, shape: tuple[int, int]) -> np.ndarray:
    # TODO(pluvio): ship pre-computed terrain / landmask / coast-distance
    # numpy arrays under `assets/static/` and load them here.
    return np.zeros(shape, dtype="float32")


def _ensure(x: np.ndarray) -> np.ndarray:
    """Coerce NaN → 0 and guarantee dtype float32 + correct shape."""
    x = np.asarray(x, dtype="float32")
    if x.shape != GRID:
        # Crude resize; the collectors should already deliver the right shape.
        from PIL import Image

        x = np.asarray(
            Image.fromarray(x).resize(GRID[::-1], Image.BILINEAR), dtype="float32"
        )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x
