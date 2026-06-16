"""Colour ramp shared with the legend and the Flutter app.

The overlay paints each pixel the **discrete WMO band colour** the legend shows
(`web/src/components/PrecipitationLegend.tsx` / `domain/precip.ts`): light,
moderate, heavy, violent. We deliberately do *not* interpolate between bands or
ramp alpha with rate — an earlier continuous ramp rendered light rain as a faint
wash that didn't match any legend swatch, so what you saw on the map didn't
correspond to the key. Banding makes "this pixel is light blue" mean exactly
"light rain (0.1–2.5 mm/h)", the same as the legend.

Thresholds (mm/h) follow the WMO 1985 classification: 0.1 / 2.5 / 7.5 / 50.
"""

from __future__ import annotations

import numpy as np

# Below this rate it's a trace (model noise), shown as nothing — kept in sync
# with the web client's RAIN_THRESHOLD_MM_H (web/src/domain/precip.ts) so the
# map overlay doesn't tint where the chart/headline say "dry".
RAIN_THRESHOLD_MM_H = 0.1

# (lower-bound mm/h, RGB) per WMO band — colours match PRECIP_COLOR in
# web/src/domain/precip.ts and PrecipitationPalette in Dart, exactly.
BANDS: list[tuple[float, tuple[int, int, int]]] = [
    (0.1, (158, 202, 225)),   # light    #9ecae1  → [0.1, 2.5)
    (2.5, (49, 130, 189)),    # moderate #3182bd  → [2.5, 7.5)
    (7.5, (253, 141, 60)),    # heavy    #fd8d3c  → [7.5, 50)
    (50.0, (227, 26, 28)),    # violent  #e31a1c  → [50, ∞)
]


def rgba_for_array(mm_per_h: np.ndarray, alpha: int = 220) -> np.ndarray:
    """Map a (H, W) precipitation array to an (H, W, 4) uint8 RGBA array.

    Each pixel gets the solid colour of its WMO band (the legend swatch) at a
    constant ``alpha`` so the base map shows through; pixels below the rain
    threshold are fully transparent. No cross-band interpolation — the map is a
    1:1 read of the legend.
    """
    if mm_per_h.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {mm_per_h.shape}")

    rate = np.clip(np.nan_to_num(mm_per_h, nan=0.0), 0.0, None).astype("float32")
    h, w = rate.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")  # transparent by default

    # Assign each band its solid colour. Iterating low→high lets each higher
    # band overwrite the previous where the rate qualifies.
    for lo, color in BANDS:
        mask = rate >= lo
        rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = color
        rgba[mask, 3] = alpha

    # Trace/no rain → fully transparent (matches the web client's dry cutoff).
    rgba[rate < RAIN_THRESHOLD_MM_H] = (0, 0, 0, 0)
    return rgba
