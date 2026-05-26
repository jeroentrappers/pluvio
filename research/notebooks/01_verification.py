# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Nowcast verification
#
# Pair the radar nowcast against the eventual observation, then compute the
# canonical meteorological verification metrics by lead-time and by
# precipitation intensity. The notebook is data-source-agnostic:
#
# - If `--data <dir>` points at a directory holding KNMI HDF5 files
#   (`RAD_NL25_RAC_FM_*.h5` forecasts and `RAD_NL25_RAC_5M_*.h5`
#   observations), it parses them.
# - Otherwise it generates a deterministic synthetic dataset that mimics the
#   structure (an advected Gaussian "rain blob" with lead-dependent forecast
#   noise) so you can review the analysis flow before the multi-GB KNMI
#   download finishes.
#
# Output: `research/output/` holds the plots and a Markdown summary.

# %%
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `_lib` importable whether we run from `research/` or `research/notebooks/`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _lib  # noqa: E402  (after sys.path tweak)

# %% [markdown]
# ## Setup

# %%

LOG = logging.getLogger("verification")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory holding KNMI HDF5 files (forecast + observation). "
            "When omitted, the synthetic generator is used."
        ),
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output"),
        help="Where plots / summary land. Default: ./output",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Rain/no-rain threshold (mm/h) for the categorical metrics.",
    )
    p.add_argument("--verbose", action="store_true")
    # parse_known_args lets the file run inside a notebook (where sys.argv has noise)
    args, _ = p.parse_known_args(argv)
    return args


args = _parse_args()
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
args.output.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load data
#
# Real KNMI files if a `--data` directory was given, otherwise the synthetic
# dataset. Either path produces the same in-memory shapes.

# %%

if args.data and args.data.exists():
    LOG.info("Reading KNMI HDF5 from %s", args.data)
    fc_paths = _lib.discover_runs(args.data / "radar_forecast" / "2.0")
    obs_paths = _lib.discover_observations(args.data / "nl_rdr_data_rtcor_5m" / "1.0")
    if not fc_paths or not obs_paths:
        LOG.warning("No HDF5 files found under %s — falling back to synthetic.", args.data)
        forecasts, observations = _lib.synthesise_dataset()
    else:
        forecasts = [_lib.load_forecast_h5(p) for p in fc_paths]
        observations = [_lib.load_observation_h5(p) for p in obs_paths]
else:
    LOG.info("No --data given; using synthetic dataset.")
    forecasts, observations = _lib.synthesise_dataset()

LOG.info(
    "Loaded %d forecast runs (issue range %s → %s), %d observations.",
    len(forecasts),
    forecasts[0].issue_time if forecasts else "—",
    forecasts[-1].issue_time if forecasts else "—",
    len(observations),
)

# %% [markdown]
# ## Pair forecasts ↔ observations
#
# For each forecast run, every lead-time frame is matched to the
# observation file whose `valid_time == issue_time + lead`. The result is a
# tidy DataFrame: one row per (run, lead, grid cell) with both forecast and
# truth.

# %%

paired = _lib.pair_forecasts_to_observations(forecasts, observations, sample_cells=5000)
LOG.info("Paired %d (run, lead, cell) samples across %d lead bins.",
         len(paired), paired["lead_min"].nunique())
paired.head()

# %% [markdown]
# ## Continuous metrics (MAE / RMSE / bias)

# %%

continuous = _lib.continuous_metrics(paired)
continuous

# %%

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True)
for ax, col, ylabel in zip(
    axes,
    ["mae", "rmse", "bias"],
    ["MAE (mm/h)", "RMSE (mm/h)", "Bias (mm/h, fcst − obs)"],
):
    ax.plot(continuous["lead_min"], continuous[col], marker="o")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if col == "bias":
        ax.axhline(0, color="grey", linewidth=0.8)
fig.suptitle("Continuous skill by lead time")
fig.tight_layout()
fig.savefig(args.output / "continuous_skill.png", dpi=120)
plt.show()

# %% [markdown]
# ## Skill vs. persistence baseline
#
# Persistence ("it stays as it is right now") is the cheapest possible
# baseline. A skill score > 0 means the nowcast beats it; in real radar
# nowcasts, persistence is hard to beat in the first 30 min and the skill
# typically crosses zero somewhere between 60–90 min.

# %%

persistence = _lib.persistence_baseline(paired)
skill = continuous.merge(persistence, on="lead_min")
skill["skill_score_vs_persistence"] = 1 - (skill["rmse"] ** 2) / (skill["rmse_persistence"] ** 2)
skill[["lead_min", "rmse", "rmse_persistence", "skill_score_vs_persistence"]]

# %%

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(skill["lead_min"], skill["skill_score_vs_persistence"], marker="o")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Lead time (min)")
ax.set_ylabel("Skill score vs. persistence")
ax.set_title("RMSE-skill vs. persistence baseline\n(positive = nowcast beats 'no change')")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(args.output / "skill_vs_persistence.png", dpi=120)
plt.show()

# %% [markdown]
# ## Categorical metrics (POD / FAR / CSI / HSS)
#
# Treat rain/no-rain as a binary problem at a threshold `τ`. For each lead
# bin, contingency table → POD / FAR / CSI / HSS. Repeat at several
# thresholds to expose intensity-dependent skill.

# %%

cat_tables = {
    f"τ = {tau} mm/h": _lib.categorical_metrics(paired, threshold=tau)
    for tau in (0.1, 1.0, 4.0)
}
list(cat_tables.values())[0]

# %%

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharex=True)
for ax, metric, ylabel in zip(
    axes,
    ["pod", "far", "csi", "hss"],
    ["POD (↑)", "FAR (↓)", "CSI (↑)", "HSS (↑)"],
):
    for label, table in cat_tables.items():
        ax.plot(table["lead_min"], table[metric], marker="o", label=label)
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
axes[-1].legend(loc="best", fontsize=8)
fig.suptitle("Categorical skill by lead time, at three rain thresholds")
fig.tight_layout()
fig.savefig(args.output / "categorical_skill.png", dpi=120)
plt.show()

# %% [markdown]
# ## Intensity-stratified error
#
# Does the nowcast nail drizzle but miss storms? Bin the observed rate, then
# look at the mean forecast error (forecast − observed) inside each bin.

# %%

bins = [0, 0.1, 0.5, 1.0, 2.5, 7.5, 50.0, np.inf]
labels = ["dry", "trace", "very light", "light", "moderate", "heavy", "violent"]
paired = paired.assign(intensity_band=pd.cut(paired["observed_mm_per_h"], bins=bins, labels=labels))
strata = (
    paired.assign(err=paired["forecast_mm_per_h"] - paired["observed_mm_per_h"])
    .groupby(["lead_min", "intensity_band"], observed=True)["err"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "bias_mm_per_h", "count": "n"})
)
strata.head(10)

# %%

pivot = strata.pivot(index="lead_min", columns="intensity_band", values="bias_mm_per_h")
fig, ax = plt.subplots(figsize=(8, 4))
pivot.plot(ax=ax, marker="o")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Lead time (min)")
ax.set_ylabel("Mean signed error (mm/h)")
ax.set_title("Bias by observed intensity band")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(args.output / "intensity_stratified_bias.png", dpi=120)
plt.show()

# %% [markdown]
# ## Summary report
#
# Pluvio will eventually surface a "this forecast is right N% of the time at
# horizon H" indicator next to the timeline; the numbers come from a table
# very close to this one. Write a Markdown report alongside the plots.

# %%

cat_main = cat_tables[f"τ = {args.threshold} mm/h"]
summary_md_lines = [
    "# Pluvio nowcast verification — automated summary",
    "",
    f"- Dataset source: **{'KNMI HDF5' if args.data else 'synthetic (deterministic)'}**",
    f"- Forecast runs included: **{len(forecasts)}**",
    f"- Paired samples: **{len(paired):,}** across **{paired['lead_min'].nunique()}** lead bins",
    f"- Rain threshold τ: **{args.threshold} mm/h**",
    "",
    "## Headline numbers",
    "",
    "| Lead (min) | MAE | RMSE | Bias | CSI | HSS | n |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for _, row in continuous.merge(cat_main, on="lead_min").iterrows():
    summary_md_lines.append(
        f"| {row['lead_min']} | {row['mae']:.2f} | {row['rmse']:.2f} | "
        f"{row['bias']:+.2f} | {row['csi']:.2f} | {row['hss']:.2f} | "
        f"{int(row['n']):,} |"
    )
summary_md_lines += [
    "",
    "## What 'good' looks like",
    "",
    "Public benchmarks for INCA-class extrapolation nowcasts (KNMI pysteps, ",
    "MeteoSwiss INCA, NCAR DGMR papers) at τ = 0.1 mm/h:",
    "",
    "| Lead | CSI typical |",
    "|---|---|",
    "| +10 min | 0.65–0.75 |",
    "| +30 min | 0.45–0.55 |",
    "| +60 min | 0.30–0.40 |",
    "| +120 min | 0.12–0.20 |",
    "",
    "Numbers far below those usually mean the pairing is wrong (timezone, ",
    "off-by-one on lead). Far above usually means an unintentional ",
    "train/test leak.",
]
summary_path = args.output / "summary.md"
summary_path.write_text("\n".join(summary_md_lines) + "\n", encoding="utf-8")
LOG.info("Wrote %s", summary_path)
print("\n".join(summary_md_lines[:14]))
