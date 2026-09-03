"""Render precipitation arrays to PNG overlays.

The Flutter app pulls these via `OverlayImage` in `radar_map.dart`. We pre-
render one PNG per (refresh, lead_min) so the request path is a single
static file read.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
from PIL import Image

from .colormap import draw_fiducials, rgba_for_array, upsample_field


def render_overlay(
    mm_per_h: np.ndarray,
    target_hw: tuple[int, int] | None = None,
    fiducials: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Render a single precipitation field as PNG bytes."""
    if target_hw is not None:
        mm_per_h = upsample_field(mm_per_h, target_hw)
    rgba = rgba_for_array(mm_per_h)
    if fiducials is not None:
        draw_fiducials(rgba, fiducials)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_overlay_to_path(
    mm_per_h: np.ndarray,
    path: pathlib.Path,
    target_hw: tuple[int, int] | None = None,
    fiducials: tuple[float, float, float, float] | None = None,
) -> pathlib.Path:
    """Convenience wrapper: render and write to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_overlay(mm_per_h, target_hw, fiducials=fiducials))
    return path


def render_sprite(
    fields: list[np.ndarray],
    cols: int = 12,
    target_hw: tuple[int, int] | None = None,
    fiducials: tuple[float, float, float, float] | None = None,
) -> tuple[bytes, int, int]:
    """Tile many precipitation fields into one RGBA sprite-sheet PNG.

    The web client downloads this single image per prediction and scrubs by
    cropping the tile for the current lead — so animating the whole horizon costs
    one request instead of one-per-frame. Tiles are laid out row-major in the
    given order. Returns (png_bytes, rows, cols). Mostly-transparent precip
    fields compress to a tiny PNG.
    """
    if not fields:
        raise ValueError("no fields to tile")
    if target_hw is not None:
        fields = [upsample_field(f, target_hw) for f in fields]
    h, w = fields[0].shape
    rows = -(-len(fields) // cols)  # ceil
    sheet = np.zeros((rows * h, cols * w, 4), dtype="uint8")
    for i, f in enumerate(fields):
        r, c = divmod(i, cols)
        tile = rgba_for_array(f)
        if fiducials is not None:
            draw_fiducials(tile, fiducials)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    buf = io.BytesIO()
    Image.fromarray(sheet, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue(), rows, cols
