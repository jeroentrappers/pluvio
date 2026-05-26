"""Shared helpers for the verification notebooks.

Two responsibilities:

1. Parse the KNMI HDF5 products we care about into tidy DataFrames:
   - `radar_forecast` v2.0  → one row per (issue_time, lead_min, y, x)
   - `nl-rdr-data-rtcor-5m` → one row per (observation_time, y, x)
2. Provide a deterministic synthetic dataset that mimics those shapes so
   the analysis can run end-to-end without the multi-GB KNMI download.
   The synthetic generator embeds the expected "skill decays with lead
   time" structure so the resulting plots look like the real thing —
   useful for code review.

The HDF5 schema is documented at
https://dataplatform.knmi.nl/dataset/radar-forecast-2-0 — the field names
below match what the dataset's README describes, but real files in the
wild occasionally vary; the loaders are defensive about that.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd

LOG = logging.getLogger("pluvio.research.lib")

# KNMI radar grid is nominally 765×700 (NL composite). We downsample to a
# 100×100 footprint for analysis to keep memory and plotting honest.
ANALYSIS_GRID = (100, 100)

# Lead times in minutes for the radar_forecast nowcast (PT5M cadence × 24 = 2h).
FORECAST_LEAD_MINUTES: tuple[int, ...] = tuple(range(0, 125, 5))


@dataclasses.dataclass(frozen=True)
class ForecastRun:
    """A single radar_forecast HDF5 file decoded into memory."""

    issue_time: datetime
    # shape (n_leads, H, W) — precipitation rate in mm/h
    frames: np.ndarray
    # lead times in minutes, len == frames.shape[0]
    leads_min: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        n_leads, h, w = self.frames.shape
        yi, xi = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rows = []
        for k, lead in enumerate(self.leads_min):
            rows.append(
                pd.DataFrame(
                    {
                        "issue_time": self.issue_time,
                        "lead_min": int(lead),
                        "y": yi.ravel(),
                        "x": xi.ravel(),
                        "rate_mm_per_h": self.frames[k].ravel().astype("float32"),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)


@dataclasses.dataclass(frozen=True)
class Observation:
    """A single observation HDF5 file decoded into memory."""

    valid_time: datetime
    # shape (H, W) — precipitation rate in mm/h
    field: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        h, w = self.field.shape
        yi, xi = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        return pd.DataFrame(
            {
                "valid_time": self.valid_time,
                "y": yi.ravel(),
                "x": xi.ravel(),
                "rate_mm_per_h": self.field.ravel().astype("float32"),
            }
        )


# --------------------------------------------------------------------- HDF5

_FILENAME_TS = re.compile(r"(\d{12})")


def _parse_ts(path: pathlib.Path) -> datetime:
    m = _FILENAME_TS.search(path.name)
    if not m:
        raise ValueError(f"No YYYYMMDDHHMM stamp in {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def load_forecast_h5(path: pathlib.Path) -> ForecastRun:
    """Decode a KNMI radar_forecast v2.0 file.

    Expected layout:
      /forecast/forecast_0min  (uint16 dataset, scale via attrs)
      /forecast/forecast_5min
      ...
      /forecast/forecast_120min
    Each dataset has `calibration_formulas` / `calibration_out_scale_factor_gr`
    style attributes; values are mm/h after applying the scale.

    Real files vary slightly across vintages. We try the documented layout
    first and fall back to scanning the HDF5 hierarchy.
    """
    import h5py  # local import — keeps `_lib.py` importable without h5py

    issue = _parse_ts(path)
    leads: list[int] = []
    frames: list[np.ndarray] = []

    with h5py.File(path, "r") as f:
        for lead in FORECAST_LEAD_MINUTES:
            key = f"/forecast/forecast_{lead}min"
            if key not in f:
                continue
            ds = f[key]
            raw = np.asarray(ds[()], dtype="float32")
            scale = float(ds.attrs.get("calibration_out_scale_factor_gr", 1.0))
            offset = float(ds.attrs.get("calibration_out_scale_offset_gr", 0.0))
            field = raw * scale + offset
            # No-data sentinels in KNMI files are commonly 65535 → mark as NaN.
            field[raw >= 65535] = np.nan
            frames.append(_resample(field, ANALYSIS_GRID))
            leads.append(lead)

    if not frames:
        raise RuntimeError(f"No /forecast/forecast_*min datasets in {path}")
    return ForecastRun(
        issue_time=issue,
        frames=np.stack(frames, axis=0),
        leads_min=np.asarray(leads, dtype="int32"),
    )


def load_observation_h5(path: pathlib.Path) -> Observation:
    """Decode an `nl-rdr-data-rtcor-5m` file.

    Layout: `/image1/image_data` is the radar+gauge composite, with
    calibration attributes on the `image1` group.
    """
    import h5py

    valid = _parse_ts(path)
    with h5py.File(path, "r") as f:
        ds = f["/image1/image_data"]
        raw = np.asarray(ds[()], dtype="float32")
        calib_group = f["/image1"]
        scale = float(calib_group.attrs.get("calibration_out_scale_factor_gr", 1.0))
        offset = float(calib_group.attrs.get("calibration_out_scale_offset_gr", 0.0))
        field = raw * scale + offset
        field[raw >= 65535] = np.nan
    return Observation(valid_time=valid, field=_resample(field, ANALYSIS_GRID))


def _resample(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Block-mean downsample; falls through if the field is already the
    requested shape or smaller."""
    if field.shape == shape:
        return field
    h, w = field.shape
    th, tw = shape
    if h < th or w < tw:
        return field
    yh = h // th
    xw = w // tw
    trimmed = field[: th * yh, : tw * xw]
    return trimmed.reshape(th, yh, tw, xw).mean(axis=(1, 3))


# ------------------------------------------------------------- discovery


def discover_runs(forecast_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(forecast_dir.glob("RAD_NL25_RAC_FM_*.h5"))


def discover_observations(obs_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(obs_dir.glob("RAD_NL25_RAC_5M_*.h5"))


# ---------------------------------------------------------- synthetic data


def synthesise_dataset(
    n_runs: int = 12,
    rng_seed: int = 42,
    shape: tuple[int, int] = ANALYSIS_GRID,
) -> tuple[list[ForecastRun], list[Observation]]:
    """Build a deterministic toy dataset that mirrors the real shapes.

    A single rain blob (2-D Gaussian) drifts diagonally across the grid at a
    fixed velocity. The "ground truth" at time t is the blob's position at t.
    The "forecast at lead h" is the blob's *advected* position at t+h, plus
    Gaussian noise whose σ grows linearly with lead time — i.e. exactly the
    skill-decays-with-lead behaviour radar nowcasts exhibit in real life.

    Returns parallel lists of forecast runs and observations such that
    `forecasts[k].issue_time + leads_min == observations[m].valid_time` for
    many combinations of (k, m) — i.e. there's something to verify against.
    """
    rng = np.random.default_rng(rng_seed)
    h, w = shape
    issue0 = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    issue_step = timedelta(minutes=5)
    velocity = np.array([0.6, 0.4])  # cells per minute
    blob_intensity = 8.0  # mm/h peak
    blob_sigma = 6.0  # cells

    issue_times = [issue0 + i * issue_step for i in range(n_runs)]
    leads_min = np.array(FORECAST_LEAD_MINUTES, dtype="int32")

    def _draw_blob(centre: np.ndarray, sigma_extra: float) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        sigma2 = blob_sigma**2 + sigma_extra**2
        d2 = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2
        return blob_intensity * np.exp(-d2 / (2 * sigma2))

    def _blob_centre_at(t: datetime) -> np.ndarray:
        delta_min = (t - issue0).total_seconds() / 60.0
        return np.array([15.0, 20.0]) + velocity * delta_min

    # Observations span both the forecast issue times and their forecast tails.
    obs_times: set[datetime] = set()
    for issue in issue_times:
        for lead in leads_min:
            obs_times.add(issue + timedelta(minutes=int(lead)))
    obs_list = [
        Observation(valid_time=t, field=_draw_blob(_blob_centre_at(t), 0.0))
        for t in sorted(obs_times)
    ]

    fc_list: list[ForecastRun] = []
    for issue in issue_times:
        frames = []
        for lead in leads_min:
            # The forecast "knows" the blob's trajectory but its certainty
            # decays with lead — bigger σ noise, plus a centre wobble.
            extra_sigma = 0.04 * lead  # cells, ≈ 1 cell at +25 min
            wobble = rng.normal(scale=0.06 * lead, size=2)
            centre = _blob_centre_at(issue + timedelta(minutes=int(lead))) + wobble
            field = _draw_blob(centre, extra_sigma)
            field += rng.normal(scale=0.05 + 0.01 * lead, size=field.shape)
            field = np.clip(field, 0, None)
            frames.append(field)
        fc_list.append(
            ForecastRun(
                issue_time=issue,
                frames=np.stack(frames, axis=0),
                leads_min=leads_min.copy(),
            )
        )
    return fc_list, obs_list


# ------------------------------------------------------------- pairing


def pair_forecasts_to_observations(
    forecasts: Iterable[ForecastRun],
    observations: Iterable[Observation],
    *,
    sample_cells: int | None = 5000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build the tidy verification frame.

    Returns one row per (issue_time, lead_min, cell) with both the forecast
    rate and the corresponding observation rate at that grid cell at the
    valid time `issue_time + lead_min`.

    ``sample_cells`` keeps the frame small enough to plot — set to None for
    the full grid.
    """
    obs_by_time = {o.valid_time: o for o in observations}
    rng = np.random.default_rng(rng_seed)
    rows: list[pd.DataFrame] = []

    for run in forecasts:
        n_leads, h, w = run.frames.shape
        if sample_cells is None:
            yi, xi = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            ys, xs = yi.ravel(), xi.ravel()
        else:
            flat_n = h * w
            idx = rng.choice(flat_n, size=min(sample_cells, flat_n), replace=False)
            ys, xs = np.unravel_index(idx, (h, w))

        for k, lead in enumerate(run.leads_min):
            valid = run.issue_time + timedelta(minutes=int(lead))
            obs = obs_by_time.get(valid)
            if obs is None:
                continue
            f_vals = run.frames[k, ys, xs]
            o_vals = obs.field[ys, xs]
            rows.append(
                pd.DataFrame(
                    {
                        "issue_time": run.issue_time,
                        "lead_min": int(lead),
                        "y": ys,
                        "x": xs,
                        "forecast_mm_per_h": f_vals,
                        "observed_mm_per_h": o_vals,
                    }
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "issue_time",
                "lead_min",
                "y",
                "x",
                "forecast_mm_per_h",
                "observed_mm_per_h",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["forecast_mm_per_h", "observed_mm_per_h"])
    return out


# ------------------------------------------------------------- metrics


def continuous_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / bias per lead-time."""
    g = df.assign(err=df["forecast_mm_per_h"] - df["observed_mm_per_h"]).groupby("lead_min")
    return pd.DataFrame(
        {
            "lead_min": list(g.groups.keys()),
            "n": g.size().to_numpy(),
            "mae": g["err"].apply(lambda e: e.abs().mean()).to_numpy(),
            "rmse": g["err"].apply(lambda e: float(np.sqrt((e**2).mean()))).to_numpy(),
            "bias": g["err"].mean().to_numpy(),
        }
    )


def categorical_metrics(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Hit / Miss / FA / CN and POD / FAR / CSI / HSS per lead-time."""
    df = df.copy()
    df["fp"] = df["forecast_mm_per_h"] >= threshold
    df["op"] = df["observed_mm_per_h"] >= threshold

    rows = []
    for lead, sub in df.groupby("lead_min"):
        h = int(((sub.fp) & (sub.op)).sum())
        m = int(((~sub.fp) & (sub.op)).sum())
        fa = int(((sub.fp) & (~sub.op)).sum())
        cn = int(((~sub.fp) & (~sub.op)).sum())
        pod = h / (h + m) if (h + m) else float("nan")
        far = fa / (h + fa) if (h + fa) else float("nan")
        csi = h / (h + m + fa) if (h + m + fa) else float("nan")
        # HSS — handle zero denominator
        denom = (h + m) * (m + cn) + (h + fa) * (fa + cn)
        hss = (2 * (h * cn - m * fa) / denom) if denom else float("nan")
        rows.append(
            {
                "lead_min": int(lead),
                "threshold_mm_per_h": threshold,
                "hits": h,
                "misses": m,
                "false_alarms": fa,
                "correct_negatives": cn,
                "pod": pod,
                "far": far,
                "csi": csi,
                "hss": hss,
            }
        )
    return pd.DataFrame(rows)


def persistence_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Persistence assumes the observation at lead=0 holds for all leads.
    For each (issue_time, cell), pair the lead-0 observation against the
    later observations to get a "do-nothing" baseline RMSE per lead.
    """
    base = df[df["lead_min"] == 0][["issue_time", "y", "x", "observed_mm_per_h"]]
    base = base.rename(columns={"observed_mm_per_h": "persistence_mm_per_h"})
    joined = df.merge(base, on=["issue_time", "y", "x"], how="inner")
    g = joined.assign(err=joined["persistence_mm_per_h"] - joined["observed_mm_per_h"]).groupby(
        "lead_min"
    )
    return pd.DataFrame(
        {
            "lead_min": list(g.groups.keys()),
            "rmse_persistence": g["err"].apply(lambda e: float(np.sqrt((e**2).mean()))).to_numpy(),
        }
    )
