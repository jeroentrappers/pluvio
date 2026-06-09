"""Shared helpers for the verification notebooks.

Two responsibilities:

1. Parse the KNMI HDF5 products we care about into tidy DataFrames:
   - `radar_forecast` v2.0 ........ one file per *issue* time. Contains 25
     prediction steps (image1 → image25, +0 → +120 min at 5-min cadence).
     Each `imageN/image_data` is a uint16 grid (765×700) of *5-minute
     accumulation in mm*, calibrated via `GEO = scale·PV + offset` on the
     attribute `image_geo_parameter == "PRECIP_[MM]"`. Multiply by 12 to
     get mm/h.
   - `nl_rdr_data_rtcor_5m` v1.0 .. one file per *observation* time.
     `image1` is the precipitation field (same calibration / units as the
     forecast file), `image2` is quality, `image3` is the gauge-adjustment
     factor — we ignore the latter two for now.
2. Provide a deterministic synthetic dataset with the same shapes so the
   notebook can run with zero network.

The HDF5 schema notes above are pinned from a live response on 2026-05-26;
KNMI documents the calibration on `imageN/calibration/calibration_formulas`,
the no-data sentinel as 65534 and the out-of-image sentinel as 65535.
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

# Native KNMI grid is 765×700. We block-mean down to 100×100 for analysis
# so paired DataFrames stay small enough to plot without losing structure.
ANALYSIS_GRID = (100, 100)

# 5-min accumulation → hourly rate factor.
_MM_5MIN_TO_MM_PER_H = 12.0


@dataclasses.dataclass(frozen=True)
class ForecastRun:
    """A single radar_forecast HDF5 file decoded into memory."""

    issue_time: datetime
    # shape (n_leads, H, W) — precipitation **rate** in mm/h
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
_KNMI_VALID_TS = re.compile(r"^(\d{2})-([A-Z]{3})-(\d{4});(\d{2}):(\d{2}):(\d{2})")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _parse_ts_from_filename(path: pathlib.Path) -> datetime:
    m = _FILENAME_TS.search(path.name)
    if not m:
        raise ValueError(f"No YYYYMMDDHHMM stamp in {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def _parse_knmi_valid_ts(raw: bytes | str) -> datetime:
    """Decode KNMI's `dd-MON-yyyy;HH:MM:SS.fff` valid-time string."""
    s = raw.decode() if isinstance(raw, bytes) else raw
    m = _KNMI_VALID_TS.match(s)
    if not m:
        raise ValueError(f"Unrecognised valid time {s!r}")
    day, mon, year, hh, mm, ss = m.groups()
    return datetime(
        int(year), _MONTHS[mon], int(day), int(hh), int(mm), int(ss), tzinfo=timezone.utc
    )


def _decode_precip_grid(image_group, image_ds, fallback_cal=None) -> np.ndarray:
    """Apply the calibration on a KNMI ``imageN`` group and return mm/h.

    In the radar_forecast v2.0 archive only image1 carries a `calibration`
    subgroup; image2..25 share it implicitly. Callers pass image1's
    calibration as `fallback_cal` so the same loader works on both layouts.
    """
    raw = np.asarray(image_ds[()], dtype="float32")
    # Calibration formula is on the *child* group `imageN/calibration`:
    #   GEO = scale*PV + offset
    cal = image_group["calibration"] if "calibration" in image_group else fallback_cal
    if cal is None:
        raise KeyError("no calibration on this image group and no fallback supplied")
    formula = cal.attrs.get("calibration_formulas", b"GEO=0.01*PV+0.0")
    if isinstance(formula, bytes):
        formula = formula.decode()
    m = re.match(r"GEO\s*=\s*([0-9.+-]+)\s*\*\s*PV\s*\+\s*([0-9.+-]+)", formula)
    if not m:
        raise ValueError(f"Unparseable calibration {formula!r}")
    scale, offset = float(m.group(1)), float(m.group(2))
    mm_5min = raw * scale + offset
    missing = int(cal.attrs.get("calibration_missing_data", [65534])[0])
    out_of_img = int(cal.attrs.get("calibration_out_of_image", [65535])[0])
    mm_5min[raw == missing] = np.nan
    mm_5min[raw == out_of_img] = np.nan
    return mm_5min * _MM_5MIN_TO_MM_PER_H


def load_forecast_h5(path: pathlib.Path) -> ForecastRun:
    """Decode a KNMI radar_forecast v2.0 file.

    Layout (verified against a live 2026-05-26 file):
      image1 / image2 / … / image25 — each a group with:
        @image_datetime_valid   "dd-MON-yyyy;HH:MM:SS.fff"
        image_data              uint16 (765×700), raw pixel value
        calibration             group with @calibration_formulas
    image1 is the +0 forecast (= issue time); image25 is +120 min.
    """
    import h5py

    issue_from_name = _parse_ts_from_filename(path)
    leads: list[int] = []
    frames: list[np.ndarray] = []

    with h5py.File(path, "r") as f:
        # Sort image-group names by their numeric suffix so we get +0, +5, … in order.
        groups = sorted(
            (n for n in f if n.startswith("image")),
            key=lambda n: int(n.removeprefix("image")),
        )
        if not groups:
            raise RuntimeError(f"No imageN groups in {path}")

        issue_time = None
        # Newer schema only carries the calibration on image1; share it.
        fallback_cal = f["image1"].get("calibration") if "image1" in f else None
        for name in groups:
            grp = f[name]
            valid_raw = grp.attrs.get("image_datetime_valid")
            if valid_raw is None:
                continue
            valid = _parse_knmi_valid_ts(valid_raw)
            if issue_time is None:
                issue_time = valid
            lead_min = int(round((valid - issue_time).total_seconds() / 60))
            field = _decode_precip_grid(grp, grp["image_data"], fallback_cal=fallback_cal)
            frames.append(_resample(field, ANALYSIS_GRID))
            leads.append(lead_min)

    if not frames:
        raise RuntimeError(f"No usable image groups in {path}")
    return ForecastRun(
        issue_time=issue_time or issue_from_name,
        frames=np.stack(frames, axis=0),
        leads_min=np.asarray(leads, dtype="int32"),
    )


def load_observation_h5(path: pathlib.Path) -> Observation:
    """Decode an `nl_rdr_data_rtcor_5m` file.

    The file holds three image groups; we only need image1 (precipitation).
    The filename timestamp is the *end* of the 5-min accumulation window.
    """
    import h5py

    valid = _parse_ts_from_filename(path)
    with h5py.File(path, "r") as f:
        grp = f["image1"]
        field = _decode_precip_grid(grp, grp["image_data"])
    return Observation(valid_time=valid, field=_resample(field, ANALYSIS_GRID))


def _resample(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Block-mean downsample; pass through if already at or below shape."""
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
    """Glob for forecast files in either the old (5M suffix) or new (FM)
    naming. The current product is `RAD_NL25_RAC_FM_*.h5`."""
    return sorted(forecast_dir.glob("RAD_NL25_RAC_FM_*.h5"))


def discover_observations(obs_dir: pathlib.Path) -> list[pathlib.Path]:
    """Observation files are `RAD_NL25_RAC_RT_*.h5` (RT = real-time)."""
    return sorted(obs_dir.glob("RAD_NL25_RAC_RT_*.h5"))


# ---------------------------------------------------------- synthetic data


def synthesise_dataset(
    n_runs: int = 12,
    rng_seed: int = 42,
    shape: tuple[int, int] = ANALYSIS_GRID,
) -> tuple[list[ForecastRun], list[Observation]]:
    """Build a deterministic toy dataset that mirrors the real shapes.

    A single rain blob (2-D Gaussian) drifts diagonally across the grid. The
    "ground truth" at time t is the blob's true position at t. The "forecast
    at lead h" advects the blob forward and then adds noise whose σ grows
    linearly with lead time — exactly the skill-decays-with-lead behaviour
    radar nowcasts show in practice.
    """
    rng = np.random.default_rng(rng_seed)
    h, w = shape
    issue0 = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    issue_step = timedelta(minutes=5)
    velocity = np.array([0.6, 0.4])  # cells per minute
    blob_intensity = 8.0
    blob_sigma = 6.0
    leads_min = np.arange(0, 125, 5, dtype="int32")

    def _draw_blob(centre: np.ndarray, sigma_extra: float) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        sigma2 = blob_sigma**2 + sigma_extra**2
        d2 = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2
        return blob_intensity * np.exp(-d2 / (2 * sigma2))

    def _blob_centre_at(t: datetime) -> np.ndarray:
        delta_min = (t - issue0).total_seconds() / 60.0
        return np.array([15.0, 20.0]) + velocity * delta_min

    issue_times = [issue0 + i * issue_step for i in range(n_runs)]

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
            extra_sigma = 0.04 * lead
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
    """Tidy verification frame.

    One row per (issue_time, lead_min, cell) with both forecast and truth.
    ``sample_cells`` keeps the frame small enough to plot — set to None to
    use the full grid.
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
    """Persistence assumes the lead-0 observation holds at every lead.

    For each (issue_time, cell), pair the lead-0 observation against later
    observations. RMSE of that baseline lets us compute a skill score
    that rewards beating "do nothing".
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
