"""Serve observed rainfall (radar QPE) for the app's history mode.

A hetz1 producer (research/model/produce_observed.py) maintains `observed.npz`:
the last ~3 h of gauge-validated radar composites on the Belgium serving bounds
at ~1 km. This module reads it and mirrors the forecast serving shape — per-point
rates plus one sprite sheet — so the web client can reuse its whole animation
pipeline with negative lead times.

The sprite is rendered lazily on first request per npz mtime and cached on disk,
so the producer stays a plain numpy writer and the request path stays a file read.
"""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
import threading
from datetime import UTC, datetime

import numpy as np

from .colormap import rgba_for_array
from .tiler import render_sprite

LOG = logging.getLogger("pluvio.history")

OBSERVED_NPZ = pathlib.Path(
    os.environ.get("PLUVIO_OBSERVED_NPZ", "/opt/pluvio/serve/observed.npz")
)
MAX_AGE_S = int(os.environ.get("PLUVIO_OBSERVED_MAX_AGE_S", "3600"))
_SPRITE_DIR = pathlib.Path(os.environ.get("PLUVIO_HISTORY_SPRITE_DIR", "/tmp/pluvio_history"))
_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "data": None}


def _load():
    """The observed cube, cached per file mtime. None when missing/stale/unreadable."""
    try:
        mtime = OBSERVED_NPZ.stat().st_mtime
    except OSError:
        return None
    with _LOCK:
        if _CACHE["mtime"] == mtime and _CACHE["data"] is not None:
            return _CACHE["data"]
        try:
            d = np.load(OBSERVED_NPZ, allow_pickle=False)
            times = d["times"].astype("int64")
            rates = d["rates"].astype("float32")
            bounds = [float(x) for x in d["bounds"]]  # W, S, E, N
        except Exception as exc:
            LOG.warning("observed cube unreadable (%s)", exc)
            return None
        age = datetime.now(UTC).timestamp() - float(times[-1])
        if age > MAX_AGE_S:
            LOG.warning("observed cube stale (%.0f s)", age)
            return None
        data = {"mtime": mtime, "times": times, "rates": rates,
                "bounds": {"west": bounds[0], "south": bounds[1],
                           "east": bounds[2], "north": bounds[3]}}
        _CACHE.update(mtime=mtime, data=data)
        return data


def available() -> bool:
    return _load() is not None


def point_frames(lat: float, lon: float, span_min: int):
    """[(epoch, rate)] at a location for the trailing span, oldest→newest."""
    data = _load()
    if data is None:
        return None
    b = data["bounds"]
    if not (b["west"] <= lon <= b["east"] and b["south"] <= lat <= b["north"]):
        raise ValueError(f"location ({lat}, {lon}) outside observed bounds")
    n, h, w = data["rates"].shape
    row = min(h - 1, max(0, int((b["north"] - lat) / (b["north"] - b["south"]) * h)))
    col = min(w - 1, max(0, int((lon - b["west"]) / (b["east"] - b["west"]) * w)))
    newest = int(data["times"][-1])
    out = []
    for i in range(n):
        t = int(data["times"][i])
        if newest - t > span_min * 60:
            continue
        r0, r1 = max(0, row - 1), row + 2
        c0, c1 = max(0, col - 1), col + 2
        blk = data["rates"][i, r0:r1, c0:c1]
        blk = blk[np.isfinite(blk)]
        out.append((t, float(blk.max()) if blk.size else 0.0))
    return out


def sprite_info():
    """Layout of the current sprite ({tile_w, tile_h, cols, rows, index}), or None.

    index maps epoch→tile, matching the order of `times`.
    """
    data = _load()
    if data is None:
        return None
    n, h, w = data["rates"].shape
    cols = 6
    rows = -(-n // cols)
    return {"tile_w": w, "tile_h": h, "cols": cols, "rows": rows,
            "count": n, "index": {int(t): i for i, t in enumerate(data["times"])},
            "mtime": int(data["mtime"])}


def sprite_png_path() -> pathlib.Path | None:
    """Render (or reuse) the sprite sheet for the current cube."""
    data = _load()
    if data is None:
        return None
    _SPRITE_DIR.mkdir(parents=True, exist_ok=True)
    path = _SPRITE_DIR / f"sprite_{int(data['mtime'])}.png"
    if path.exists():
        return path
    fields = [data["rates"][i] for i in range(data["rates"].shape[0])]
    png, _rows, _cols = render_sprite(fields, cols=6)
    with tempfile.NamedTemporaryFile(dir=_SPRITE_DIR, suffix=".png", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    tmp.write_bytes(png)
    tmp.replace(path)
    for old in _SPRITE_DIR.glob("sprite_*.png"):      # keep only the current one
        if old != path:
            old.unlink(missing_ok=True)
    return path


def overlay_png(epoch: int) -> bytes | None:
    """One frame as PNG (fallback for clients that don't use the sprite)."""
    from PIL import Image
    import io

    data = _load()
    if data is None:
        return None
    idx = {int(t): i for i, t in enumerate(data["times"])}
    if epoch not in idx:
        return None
    rgba = rgba_for_array(data["rates"][idx[epoch]])
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
