"""Colour ramp shared with the Flutter app.

Mirrors `lib/features/radar/presentation/widgets/precipitation_legend.dart`
so the overlay PNGs the API serves look identical to the in-app legend.

Thresholds (mm/h) follow the WMO 1985 classification used everywhere else
in the project: 0 / 2.5 / 7.5 / 50.
"""

from __future__ import annotations

import numpy as np

# (lower-bound mm/h, RGB tuple) — matches PrecipitationPalette in Dart.
STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (220, 220, 220)),    # "none" — light grey
    (0.001, (158, 202, 225)),  # light
    (2.5, (49, 130, 189)),     # moderate
    (7.5, (253, 141, 60)),     # heavy
    (50.0, (227, 26, 28)),     # violent
]


def rgba_for_array(mm_per_h: np.ndarray, max_alpha: int = 220) -> np.ndarray:
    """Map a (H, W) precipitation array to an (H, W, 4) uint8 RGBA array.

    Pixels with rate ≤ 0 are fully transparent so the underlying base map
    shows through. Above zero we interpolate linearly between the stops,
    and use a smooth alpha ramp so light drizzle fades in rather than
    snapping on.
    """
    if mm_per_h.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {mm_per_h.shape}")

    rate = np.clip(np.nan_to_num(mm_per_h, nan=0.0), 0.0, None).astype("float32")
    h, w = rate.shape

    stop_values = np.array([s[0] for s in STOPS], dtype="float32")
    stop_colors = np.array([s[1] for s in STOPS], dtype="float32")  # (n_stops, 3)

    # Find for each pixel the upper stop index, then interpolate.
    idx = np.searchsorted(stop_values, rate, side="right")
    idx = np.clip(idx, 1, len(STOPS) - 1)
    lower = stop_values[idx - 1]
    upper = stop_values[idx]
    denom = np.where(upper > lower, upper - lower, 1.0)
    t = np.clip((rate - lower) / denom, 0.0, 1.0)

    c_lo = stop_colors[idx - 1]
    c_hi = stop_colors[idx]
    rgb = c_lo + (c_hi - c_lo) * t[..., None]

    # Alpha: zero rain → fully transparent. Light drizzle ramps in over
    # the first 0.3 mm/h. Saturate at ``max_alpha`` so a base map remains
    # visible underneath the heaviest cells.
    alpha = np.clip(rate / 0.3, 0.0, 1.0) * max_alpha
    rgba = np.empty((h, w, 4), dtype="uint8")
    rgba[..., :3] = rgb.astype("uint8")
    rgba[..., 3] = alpha.astype("uint8")
    rgba[rate <= 0] = (0, 0, 0, 0)
    return rgba
