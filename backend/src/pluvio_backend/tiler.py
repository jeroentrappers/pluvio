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

from .colormap import rgba_for_array


def render_overlay(mm_per_h: np.ndarray) -> bytes:
    """Render a single precipitation field as PNG bytes."""
    rgba = rgba_for_array(mm_per_h)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_overlay_to_path(mm_per_h: np.ndarray, path: pathlib.Path) -> pathlib.Path:
    """Convenience wrapper: render and write to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_overlay(mm_per_h))
    return path
