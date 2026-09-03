"""Threshold config for the QC checks — YAML/JSON, env-overridable.

Defaults reflect the store's OBSERVED conventions as of the first production
run (2026-09-02/03), not idealised physical units. Several channels are
stored normalised or in an unexplained integer range pending 1.6, and the
first hourly run threw false alarms hard-coding guessed physical ranges:

  alaro_precip  observed [0, 255] in the store. Whether that is mm/h, an
                encoded/scaled integer, or something else is an OPEN
                question (TODO 1.6) — this range is a sanity band on the
                stored representation, not a claim about units.
  msg_ir108     observed [4.26, 239] — NOT Kelvin brightness temperature,
                stored normalised/rescaled.
  aws_*         observed roughly [-0.87, 2.51] — normalised anomalies, not
                raw physical units. The range applies to any aws_-prefixed
                channel (aws_temp, aws_wind, ...).
  radar / truth mm/h, physically bounded [0, 400].

The per-channel NaN-fraction limit is 0.9 for every channel, including sst.
sst going 100% NaN over the newest 48 issues (found the first production
night) is a real dead-feed signal, not a calibration artefact — it is
deliberately NOT relaxed or special-cased away here.

Override the whole set with a YAML or JSON file, either passed explicitly
to `load_thresholds(path=...)` or via the PLUVIO_QC_THRESHOLDS env var. A
file only needs to set the keys it wants to change; anything absent falls
back to the defaults below.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field, replace

# channel name -> (min, max). Matched exactly first, then by "<prefix>_"
# so "aws" covers aws_temp, aws_wind, etc.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "radar": (0.0, 400.0),
    "truth": (0.0, 400.0),
    "alaro_precip": (0.0, 255.0),
    "msg_ir108": (0.0, 260.0),
    "aws": (-3.0, 3.0),
}

DEFAULT_NAN_LIMIT = 0.9

DEFAULT_REG_OFFSET_WARN_DEG = 0.07
DEFAULT_REG_CORR_WARN = 0.25
DEFAULT_AUX_CORR_WARN = 0.05
DEFAULT_STALE_WARN_MIN = 75.0

DEFAULT_CHURN_SCAN_WARN = 55.0
DEFAULT_INTERP_RATIO_WARN = 1.3
DEFAULT_PARITY_WARN = -0.45
DEFAULT_FREEZE_WARN = 0.34
DEFAULT_STALE_WARN_S = 2100.0
DEFAULT_GAUGE_BIAS_WARN = 5.0

_SCALAR_FIELDS = (
    "nan_limit",
    "reg_offset_warn_deg",
    "reg_corr_warn",
    "aux_corr_warn",
    "stale_warn_min",
    "churn_scan_warn",
    "interp_ratio_warn",
    "parity_warn",
    "freeze_warn",
    "stale_warn_s",
    "gauge_bias_warn",
)


@dataclass
class Thresholds:
    ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_RANGES)
    )
    nan_limit: float = DEFAULT_NAN_LIMIT

    # qc_inputs
    reg_offset_warn_deg: float = DEFAULT_REG_OFFSET_WARN_DEG
    reg_corr_warn: float = DEFAULT_REG_CORR_WARN
    aux_corr_warn: float = DEFAULT_AUX_CORR_WARN
    stale_warn_min: float = DEFAULT_STALE_WARN_MIN

    # qc_watchdog
    churn_scan_warn: float = DEFAULT_CHURN_SCAN_WARN
    interp_ratio_warn: float = DEFAULT_INTERP_RATIO_WARN
    parity_warn: float = DEFAULT_PARITY_WARN
    freeze_warn: float = DEFAULT_FREEZE_WARN
    stale_warn_s: float = DEFAULT_STALE_WARN_S
    gauge_bias_warn: float = DEFAULT_GAUGE_BIAS_WARN

    def range_for(self, channel: str) -> tuple[float, float] | None:
        """Look up the plausible-value range for a channel by exact name,
        falling back to a prefix match (e.g. "aws" for "aws_temp")."""
        if channel in self.ranges:
            return self.ranges[channel]
        for prefix, rng in self.ranges.items():
            if channel.startswith(prefix + "_"):
                return rng
        return None


def _apply_mapping(th: Thresholds, data: dict) -> Thresholds:
    th = replace(th)
    if "ranges" in data:
        th.ranges = {**th.ranges, **{k: tuple(v) for k, v in data["ranges"].items()}}
    for key in _SCALAR_FIELDS:
        if key in data:
            setattr(th, key, float(data[key]))
    return th


def load_thresholds(path: str | pathlib.Path | None = None) -> Thresholds:
    """Load thresholds. Resolution order: explicit `path` argument, then the
    PLUVIO_QC_THRESHOLDS env var, then the built-in defaults above."""
    resolved = path or os.environ.get("PLUVIO_QC_THRESHOLDS")
    th = Thresholds()
    if not resolved:
        return th
    p = pathlib.Path(resolved)
    text = p.read_text()
    if p.suffix.lower() in (".yml", ".yaml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text) if text.strip() else {}
    return _apply_mapping(th, data)
