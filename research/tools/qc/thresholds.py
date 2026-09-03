"""Threshold config for the QC checks — YAML/JSON, env-overridable.

Defaults reflect the store's OBSERVED conventions as of the first production
run (2026-09-02/03), not idealised physical units. Several channels are
stored normalised or in an unexplained integer range pending 1.6, and the
first hourly run threw false alarms hard-coding guessed physical ranges:

  alaro_precip  observed [0, 255] in the store. Whether that is mm/h, an
                encoded/scaled integer, or something else is an OPEN
                question (TODO 1.6) — this range is a sanity band on the
                stored representation, not a claim about units.
  msg_ir108     observed [4.26, 239] — NOT Kelvin brightness temperature.
                It is band-1 luminance (0-255) of the rendered GeoTIFF, not
                a normalised physical quantity.
  aws_*         each AWS channel is normalised independently in
                research/model/build_aux.py (AWS_CHANNELS: column ->
                (centre, scale), applied as (value - centre) / scale), so a
                single +-3 band across all of them false-alarms on ordinary
                weather — see per-channel derivations below.
  radar / truth mm/h, physically bounded [0, 400].

Range checks compare the channel's 0.1st/99.9th percentile over the window
to the band, not the hard min/max — a single noisy IDW cell or one bad AWS
report over 48 issues x grid should not page anyone; a real unit/feed
regression moves the bulk of the distribution, which the percentile catches.

Per-channel AWS bands (centre, scale) from build_aux.py's AWS_CHANNELS,
envelope = (physical_min - centre) / scale .. (physical_max - centre) / scale:

  aws_pressure  (1013, 20) hPa. Envelope 933-1063 hPa (deep depression to
                strong anticyclone at sea level) -> (-4.0, 2.5).
  aws_temp      (10, 10) degC. Envelope -35..45 degC (Benelux/NW-Europe
                extremes with margin) -> (-4.5, 3.5).
  aws_wind      (4, 4) m/s. Envelope 0-36 m/s (calm to storm-force gust)
                -> (-1.0, 8.0).
  aws_humidity  (70, 30) % RH. Envelope -2..100 %RH (0% plus IDW/normalis-
                ation slack) -> (-2.4, 1.0).

Only channels stored aligned to `humidity_rel_shelter_avg` etc. use these;
an aws_* channel not in this table has no default range (nan/limit checks
still apply) until it is added here.

The per-channel NaN-fraction limit is 0.9 for every channel, including sst.
sst going 100% NaN over the newest 48 issues (found the first production
night) is a real dead-feed signal, not a calibration artefact — it is
deliberately NOT relaxed or special-cased away here.

Override the whole set with a YAML or JSON file, either passed explicitly
to `load_thresholds(path=...)` or via the PLUVIO_QC_THRESHOLDS env var. A
file only needs to set the keys it wants to change; anything absent falls
back to the defaults below. The top level must be a mapping, and every
top-level key must be a recognised one ("ranges" or a scalar threshold
field) — an unrecognised key (a typo like "nan_limi") raises ValueError
rather than being silently ignored.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field, replace

# channel name -> (min, max), checked against the window's 0.1/99.9
# percentile (see module docstring). Matched exactly first, then by prefix
# for any key ending in "_" (e.g. a future "aws_" catch-all) — a bare key
# like "radar" is NEVER treated as a prefix, so it can't accidentally catch
# an unrelated "radar_dbz"/"radar_quality" channel later.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "radar": (0.0, 400.0),
    "truth": (0.0, 400.0),
    "alaro_precip": (0.0, 255.0),
    "msg_ir108": (0.0, 255.0),
    "aws_pressure": (-4.0, 2.5),
    "aws_temp": (-4.5, 3.5),
    "aws_wind": (-1.0, 8.0),
    "aws_humidity": (-2.4, 1.0),
}

DEFAULT_NAN_LIMIT = 0.9
DEFAULT_RANGE_PERCENTILE = 99.9  # compare p(100-P)/p(P), not hard min/max

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
    "range_percentile",
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
_TOP_LEVEL_KEYS = frozenset({"ranges", *_SCALAR_FIELDS})


@dataclass
class Thresholds:
    ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_RANGES)
    )
    nan_limit: float = DEFAULT_NAN_LIMIT
    range_percentile: float = DEFAULT_RANGE_PERCENTILE

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
        falling back to a prefix match ONLY for keys that end in "_" (so a
        bare "radar" entry never matches "radar_dbz")."""
        if channel in self.ranges:
            return self.ranges[channel]
        for prefix, rng in self.ranges.items():
            if prefix.endswith("_") and channel.startswith(prefix):
                return rng
        return None


def _apply_mapping(th: Thresholds, data: dict) -> Thresholds:
    if not isinstance(data, dict):
        raise ValueError(
            f"thresholds file must contain a mapping at the top level, got {type(data).__name__}"
        )
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"unknown threshold key(s): {sorted(unknown)}; valid keys are "
            f"{sorted(_TOP_LEVEL_KEYS)}"
        )
    th = replace(th)
    if "ranges" in data:
        th.ranges = {**th.ranges, **{k: tuple(v) for k, v in data["ranges"].items()}}
    for key in _SCALAR_FIELDS:
        if key in data:
            setattr(th, key, float(data[key]))
    return th


def load_thresholds(path: str | pathlib.Path | None = None) -> Thresholds:
    """Load thresholds. Resolution order: explicit `path` argument, then the
    PLUVIO_QC_THRESHOLDS env var, then the built-in defaults above.

    Raises ValueError if the file's top level isn't a mapping, or contains
    an unrecognised key (a typo like "nan_limi" fails loudly instead of
    being silently ignored).
    """
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
