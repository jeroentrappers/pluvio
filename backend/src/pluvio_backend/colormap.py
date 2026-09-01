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

# (lower-bound mm/h, RGB) per band. The scale follows the Met Office rainfall
# key (deep blue < 0.5 through dark red > 32): eight steps instead of the four
# WMO bands, because 0.1-2.5 mm/h as ONE flat colour hid most of the structure
# in a field — the same rain looked flatter and weaker than on reference maps.
# Colours must match MAP_RAMP in web/src/domain/precip.ts exactly.
BANDS: list[tuple[float, tuple[int, int, int]]] = [
    (0.1,  (18, 25, 200)),    # deep blue   #1219c8  → [0.1, 0.5)
    (0.5,  (60, 110, 230)),   # royal blue  #3c6ee6  → [0.5, 1)
    (1.0,  (105, 200, 240)),  # sky blue    #69c8f0  → [1, 2)
    (2.0,  (60, 180, 60)),    # green       #3cb43c  → [2, 4)
    (4.0,  (240, 215, 70)),   # yellow      #f0d746  → [4, 8)
    (8.0,  (240, 160, 60)),   # orange      #f0a03c  → [8, 16)
    (16.0, (230, 60, 55)),    # red         #e63c37  → [16, 32)
    (32.0, (200, 35, 35)),    # dark red    #c82323  → [32, ∞)
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

    # Perceptual anti-flicker: within the light band, fade opacity in with rate
    # (90 → alpha across 0.1–0.5 mm/h). Radar trace echo sits right at the wet
    # threshold and crosses it scan to scan — as a solid swatch that reads as
    # cells blinking in and out of existence; as a faint wash it reads as what it
    # is, marginal drizzle. Band COLOURS are unchanged, so the legend stays true;
    # rates at 0.5 mm/h and above keep full opacity.
    light = (rate >= RAIN_THRESHOLD_MM_H) & (rate < 0.5)
    frac = (rate[light] - RAIN_THRESHOLD_MM_H) / (0.5 - RAIN_THRESHOLD_MM_H)
    rgba[light, 3] = (140 + frac * (alpha - 140)).astype("uint8")

    # Trace/no rain → fully transparent (matches the web client's dry cutoff).
    rgba[rate < RAIN_THRESHOLD_MM_H] = (0, 0, 0, 0)
    return rgba


# Diverging palette for forecast-minus-observed: red = over-forecast, blue =
# under-forecast, transparent near zero. Scaled to +/-8 mm/h — beyond that the
# sign is what matters, not the magnitude.
def diff_rgba(diff_mm_h: np.ndarray, alpha: int = 220) -> np.ndarray:
    d = np.nan_to_num(diff_mm_h, nan=0.0).astype("float32")
    h, w = d.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    mag = np.clip(np.abs(d) / 8.0, 0.0, 1.0)
    over = d > 0.15
    under = d < -0.15
    rgba[over, 0] = 220
    rgba[over, 1] = (120 * (1 - mag[over])).astype("uint8")
    rgba[over, 2] = (120 * (1 - mag[over])).astype("uint8")
    rgba[under, 2] = 220
    rgba[under, 0] = (120 * (1 - mag[under])).astype("uint8")
    rgba[under, 1] = (120 * (1 - mag[under])).astype("uint8")
    visible = over | under
    rgba[visible, 3] = (80 + 140 * mag[visible]).astype("uint8")
    return rgba
