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

from .colormap import draw_fiducials, rgba_for_array
from .tiler import render_sprite

LOG = logging.getLogger("pluvio.history")

OBSERVED_NPZ = pathlib.Path(
    os.environ.get("PLUVIO_OBSERVED_NPZ", "/opt/pluvio/serve/observed.npz")
)
# Full-resolution cube as a raw .npy + .json sidecar (see produce_observed): at 1 km
# the continental cube is ~1 GB, so it is memory-MAPPED and sliced per tile/point —
# never loaded whole. The npz above becomes the low-zoom OVERVIEW.
OBSERVED_HI = pathlib.Path(
    os.environ.get("PLUVIO_OBSERVED_HI", "/opt/pluvio/serve/observed_hi.npy")
)
TILE_PX = int(os.environ.get("PLUVIO_HISTORY_TILE_PX", "256"))
_HI_CACHE: dict = {"mtime": None, "data": None}
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


def _load_hi():
    """Memmap view of the hi-res cube, cached per sidecar mtime. None if absent/stale."""
    import json
    meta_path = OBSERVED_HI.with_suffix(".json")
    try:
        mtime = meta_path.stat().st_mtime
    except OSError:
        return None
    with _LOCK:
        if _HI_CACHE["mtime"] == mtime and _HI_CACHE["data"] is not None:
            return _HI_CACHE["data"]
        try:
            meta = json.loads(meta_path.read_text())
            rates = np.load(OBSERVED_HI, mmap_mode="r")
            times = np.asarray(meta["times"], dtype="int64")
            if list(rates.shape) != list(meta["shape"]) or rates.shape[0] != len(times):
                raise ValueError("hi cube / sidecar mismatch")
        except Exception as exc:
            LOG.warning("hi cube unreadable (%s)", exc)
            return None
        if datetime.now(UTC).timestamp() - float(times[-1]) > MAX_AGE_S:
            LOG.warning("hi cube stale")
            return None
        w, s_, e, n = meta["bounds"]
        data = {"mtime": mtime, "times": times, "rates": rates,
                "bounds": {"west": w, "south": s_, "east": e, "north": n}}
        _HI_CACHE.update(mtime=mtime, data=data)
        return data


def tiles_info():
    """Manifest of the hi-res tile pyramid level, or None when no hi cube exists.

    The grid splits into fixed TILE_PX-square tiles (edge tiles smaller). The client
    computes each tile's geographic bounds from the linear lat/lon split, downloads
    only the sprites intersecting its viewport, and keeps the npz overview for low
    zoom — that is what makes 1-km serving affordable: full resolution on screen,
    bandwidth proportional to the viewport.
    """
    data = _load_hi()
    if data is None:
        return None
    n, h, w = data["rates"].shape
    return {"tile_px": TILE_PX,
            "nx": -(-w // TILE_PX), "ny": -(-h // TILE_PX),
            "grid_h": h, "grid_w": w, "count": n,
            "bounds": data["bounds"],
            "index": {int(t): i for i, t in enumerate(data["times"])},
            "mtime": int(data["mtime"]), "cols": 6}


def tile_sprite_png_path(tx: int, ty: int) -> pathlib.Path | None:
    """Render (or reuse) the sprite sheet for ONE tile of the hi-res cube."""
    data = _load_hi()
    if data is None:
        return None
    n, h, w = data["rates"].shape
    if not (0 <= tx < -(-w // TILE_PX) and 0 <= ty < -(-h // TILE_PX)):
        return None
    _SPRITE_DIR.mkdir(parents=True, exist_ok=True)
    path = _SPRITE_DIR / f"tile_{int(data['mtime'])}_{tx}_{ty}.png"
    if path.exists():
        return path
    r0, r1 = ty * TILE_PX, min((ty + 1) * TILE_PX, h)
    c0, c1 = tx * TILE_PX, min((tx + 1) * TILE_PX, w)
    fields = [np.asarray(data["rates"][i, r0:r1, c0:c1], dtype="float32")
              for i in range(n)]
    png, _rows, _cols = render_sprite(fields, cols=6)
    with tempfile.NamedTemporaryFile(dir=_SPRITE_DIR, suffix=".png", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    tmp.write_bytes(png)
    tmp.replace(path)
    stamp = f"tile_{int(data['mtime'])}_"
    for old in _SPRITE_DIR.glob("tile_*.png"):        # drop tiles of older cubes
        if not old.name.startswith(stamp):
            old.unlink(missing_ok=True)
    return path


def available() -> bool:
    return _load() is not None


def point_frames(lat: float, lon: float, span_min: int):
    """[(epoch, rate)] at a location for the trailing span, oldest→newest.

    Reads the hi-res cube when present (a 3x3 memmap slice per frame is cheap and
    the 1-km answer is the honest one); the overview only serves as fallback."""
    data = _load_hi() or _load()
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
    import os as _os
    if _os.environ.get("PLUVIO_DEBUG_FIDUCIALS") == "1":
        b = data["bounds"]  # dict(west, south, east, north) — see _load_hi
        draw_fiducials(rgba, (float(b["west"]), float(b["south"]),
                              float(b["east"]), float(b["north"])))
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
