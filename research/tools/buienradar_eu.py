"""Archive Buienradar's Europe rain-radar composite and every forecast run.

We validate our own nowcasts against Buienradar's, so we need their product as
it was published, not as it can be reconstructed later: their site keeps only
3 hours of composite history and overwrites the forecast run in place. This
collector turns that ephemeral feed into (a) a continuous linear history of the
European radar composite and (b) one directory per forecast run, plus a
``runs.jsonl`` ledger that records the run cadence.

Collection happens with Buienradar's written permission (validation research).
Politeness is therefore still a hard requirement, not a nicety:

  * every request carries ``USER_AGENT``, which names pluvio and the repo, so
    their operators can throttle or block us by name rather than by IP;
  * a frame is fetched exactly once (sqlite index + on-disk presence), except
    for the newest ``--revision-window`` composite frames, which are re-fetched
    to detect after-the-fact revisions;
  * fetches are spaced ``--sleep`` seconds apart and retried with exponential
    backoff, and one failing frame never aborts the tick.

Endpoints (verified live 2026-09-03)
------------------------------------
The page https://www.buienradar.be/wereldwijd/europa/buienradar/eu3uurs carries
``window.apiKey = '<uuid>'``; its JS builds

    https://image-lite.buienradar.nl/3.0/metadata/<imagetype>
        ?size=full&forecast=<N>&history=<M>&ak=<apiKey>

``RadarMapRain5mEU`` with ``forecast=36&history=0`` returns the current forecast
run (30 frames observed, 5-min steps, +35 min .. +180 min from the run time).
``RadarMapRain15mEU`` with ``forecast=0&history=12`` returns the composite
history: 12 frames at 15-min steps, i.e. only 3 hours deep, and ``history`` is
capped at 12 — hence a >= 15-minute collection cadence is mandatory and we run
every 5 minutes because it costs nothing when there is nothing new.

Timestamps are UTC
------------------
The metadata carries ``timeOffset`` (hours) alongside naive ISO timestamps.
``timeOffset`` is the *display* offset the site adds, not an offset already
baked into the values: buienradar.min.js parses the timestamp with
``new Date(y, m-1, d, H, M, S)`` and then does
``d.setTime(d.getTime() + 60 * timeOffset * 60 * 1e3)`` before formatting it
for the user. Confirmed three ways on 2026-09-03:

  * the page's own rendered label for the frame stamped ``19:20:00`` read
    ``21:20`` with ``timeOffset = 2.0`` (CEST) -- so the stored value is UTC;
  * the newest composite frame was ``18:45`` while it was 19:20 UTC (a 35-min
    publication lag) and its blob's ``Last-Modified`` was 19:10:33 GMT -- a
    frame cannot be published 25 minutes after a valid time that is still two
    hours in the future;
  * the first forecast frame equalled the wall clock in UTC, not in local time.

The compact ids in the image URLs (``.../runs/webm/<run>/<valid>.png`` and
``.../current/webm/<valid>.png``) are the same UTC instants, so archive paths
and the ledger are UTC throughout and ``Z``-suffixed to say so.
:func:`check_run_id` re-verifies that identity on every tick and warns if
Buienradar ever changes it.

Usage
-----
    python -m tools.buienradar_eu collect --root /mnt/storagebox/buienradar_eu
    python -m tools.buienradar_eu collect --root ... --dry-run
    python -m tools.buienradar_eu cadence --root ... [--days 7]
    python -m tools.buienradar_eu verify --root ...
    python -m tools.buienradar_eu decode --png frame.png [--out rate.npy]

Exit status: 0 also on partial success (missing frames are warnings), non-zero
only when a metadata document itself could not be fetched or parsed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import itertools
import json
import logging
import math
import os
import pathlib
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

LOG = logging.getLogger("pluvio.buienradar_eu")

PAGE_URL = "https://www.buienradar.be/wereldwijd/europa/buienradar/eu3uurs"
METADATA_URL = (
    "https://image-lite.buienradar.nl/3.0/metadata/{imagetype}"
    "?size={size}&forecast={forecast}&history={history}&ak={key}"
)

# Identify ourselves by name and give a contact route (the repo) in every
# request, per the collection agreement.
USER_AGENT = "pluvio-buienradar-eu/1.0 (+https://github.com/jeroentrappers/pluvio)"

# Last key seen in the page; only a seed. The live key is scraped and cached
# under <root>/apikey.txt and re-scraped whenever metadata returns 400/401/403.
DEFAULT_API_KEY = "3c4a3037-85e6-4d1e-ad6c-f3f6e4b75f2f"
API_KEY_RE = re.compile(r"window\.apiKey\s*=\s*['\"]([0-9a-fA-F-]{8,})['\"]")

# size=full is the largest the endpoint honours: medium -> 550x468, full ->
# 766x652, and large/xl fall back to a 400-wide image.
IMAGE_SIZE = "full"

#: The two documents we poll. ``kind`` is also the on-disk directory name.
FEEDS = {
    "composite": {"imagetype": "RadarMapRain15mEU", "forecast": 0, "history": 12},
    "forecast": {"imagetype": "RadarMapRain5mEU", "forecast": 36, "history": 0},
}

WIDTH, HEIGHT = 766, 652

# ---------------------------------------------------------------------------
# Georeference
#
# The site's Leaflet config declares
#   euCombined: {bounds: [[61, -13.5], [34, 35]], isWebMercator: true}
# i.e. the PNG is an EPSG:3857 image whose corners are those lat/lon pairs.
# Verified: in Web Mercator that box is 5_398_995 x 4_597_021 m, an aspect of
# 0.85146 against the image's 652/766 = 0.85117 (0.03% off) and square ~7.05 km
# pixels. The equirectangular reading would need an aspect of 0.557, so the
# isWebMercator flag is real and must not be ignored.
# ---------------------------------------------------------------------------
NORTH, WEST, SOUTH, EAST = 61.0, -13.5, 34.0, 35.0
EARTH_RADIUS_M = 6378137.0  # EPSG:3857 sphere
CRS = "EPSG:3857"


def mercator_x(lon_deg: float) -> float:
    return EARTH_RADIUS_M * math.radians(lon_deg)


def mercator_y(lat_deg: float) -> float:
    return EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0))


def mercator_bounds() -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of the image in EPSG:3857 metres."""
    return (mercator_x(WEST), mercator_y(SOUTH), mercator_x(EAST), mercator_y(NORTH))


def affine_transform(width: int = WIDTH, height: int = HEIGHT) -> tuple[float, ...]:
    """GDAL/rasterio affine coefficients ``(a, b, c, d, e, f)`` for the frame.

    Maps pixel centres/corners to EPSG:3857 metres: ``x = a*col + b*row + c``,
    ``y = d*col + e*row + f`` with row 0 = north (so ``e`` is negative). Feed
    it straight to ``rasterio.transform.Affine(*affine_transform())``.
    """
    left, bottom, right, top = mercator_bounds()
    return ((right - left) / width, 0.0, left, 0.0, -(top - bottom) / height, top)


def rasterio_transform(width: int = WIDTH, height: int = HEIGHT):
    """The same transform as a ``rasterio.transform.Affine`` (import is lazy)."""
    from rasterio.transform import Affine

    return Affine(*affine_transform(width, height))


def georeference() -> dict:
    """Sidecar payload describing the frame grid — written once per archive."""
    left, bottom, right, top = mercator_bounds()
    a, b, c, d, e, f = affine_transform()
    return {
        "crs": CRS,
        "width": WIDTH,
        "height": HEIGHT,
        "row_0": "north",
        "latlon_corner_bounds": {"north": NORTH, "west": WEST, "south": SOUTH, "east": EAST},
        "bounds_3857": {"left": left, "bottom": bottom, "right": right, "top": top},
        "transform": [a, b, c, d, e, f],
        "pixel_size_m": [a, -e],
        "source": "buienradar.nl Leaflet config euCombined (isWebMercator=true)",
        "note": (
            "Corner lat/lon are the EPSG:3857 image corners, not an "
            "equirectangular box; reproject with the transform above."
        ),
    }


def world_file() -> str:
    """ESRI world-file (.pgw) text for the frame, for plain GIS consumers."""
    a, b, c, d, e, f = affine_transform()
    # World files reference pixel *centres*, the affine above pixel corners.
    return "\n".join(
        f"{v!r}" for v in (a, d, b, e, c + a / 2.0, f + e / 2.0)
    ) + "\n"


# ---------------------------------------------------------------------------
# Palette
#
# The published legend (page HTML `div.legend.precipitation` + the swatch
# colours in buienradar.min.css) has five classes in mm/h:
#
#     #e4e5ff  0-2    "zeer lichte neerslag"
#     #4d5dff  2-5    "lichte neerslag"
#     #000770  5-10   "matige neerslag"
#     #fe1600  10-100 "zware neerslag"
#     #c01cc4  100+   "extreme neerslag"
#
# The PNGs themselves are 8-bit palette images whose palette is re-quantised
# per frame (indices are NOT an intensity scale), and they carry a much finer
# continuous colour ramp than the five published classes. Chaining the ~124
# distinct colours observed across five live frames by nearest neighbour
# recovers that ramp exactly as a 4-segment polyline in RGB:
#
#     (235,236,254) -> (106,117,255) -> (5,28,209) -> (253,23,2) -> (192,28,196)
#     near-white blue    pure blue      deep blue      red           magenta
#
# so intensity is decoded by projecting a pixel colour onto that polyline and
# reading off the arc-length position t in [0, 1].
#
# PROVISIONAL: Buienradar publishes no finer legend than the five classes
# above and no dBZ/mm-h table for this ramp, so t -> mm/h below is fitted by
# projecting the four legend swatches that lie on the ramp onto it and
# interpolating log10(rate) between them. It reproduces the published class
# boundaries by construction but the values *within* a class are an
# assumption, and the 5 mm/h anchor is the weakest: the #000770 swatch sits
# ~80 RGB units off the ramp, so its position is inferred rather than
# measured. Treat rates as ordinal until validated against our own composite
# (TODO 3.7); :func:`png_to_class` needs no such assumption and is preferred
# whenever a class is enough.
# ---------------------------------------------------------------------------
#: (hex, low mm/h, high mm/h or None, label) exactly as the site publishes it.
PALETTE = (
    ("#e4e5ff", 0.0, 2.0, "very light"),
    ("#4d5dff", 2.0, 5.0, "light"),
    ("#000770", 5.0, 10.0, "moderate"),
    ("#fe1600", 10.0, 100.0, "severe"),
    ("#c01cc4", 100.0, None, "extreme"),
)

#: Colour ramp control points (RGB), recovered from live frames — see above.
RAMP = (
    (235, 236, 254),
    (106, 117, 255),
    (5, 28, 209),
    (253, 23, 2),
    (192, 28, 196),
)

#: (t, mm/h) anchors on the ramp. t values are the arc-length positions of the
#: PALETTE swatches; the 0.0117 anchor is the bottom of the "0-2" class and is
#: given the dry floor rather than 0 because the ramp's lightest colour still
#: means "some echo".
RATE_ANCHORS = (
    (0.0117, 0.1),
    (0.2501, 2.0),
    (0.4458, 5.0),
    (0.7591, 10.0),
    (1.0000, 100.0),
)

#: A colour further than this (RGB euclidean) from the ramp is not precipitation.
MAX_RAMP_DISTANCE = 40.0


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


def _ramp_geometry():
    import numpy as np

    pts = np.asarray(RAMP, dtype="float64")
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return pts, seg, cum / cum[-1]


def ramp_position(rgb):
    """Project RGB colours onto the ramp polyline.

    ``rgb`` is an array-like of shape (..., 3). Returns ``(t, distance)``:
    arc-length position in [0, 1] and the RGB distance to the ramp.
    """
    import numpy as np

    pts, seg, t_nodes = _ramp_geometry()
    c = np.asarray(rgb, dtype="float32")
    flat = c.reshape(-1, 3)
    best_d = np.full(flat.shape[0], np.inf, dtype="float32")
    best_t = np.zeros(flat.shape[0], dtype="float32")
    for i in range(len(seg)):
        a = pts[i]
        d = seg[i]
        denom = float(d @ d)
        u = np.clip(((flat - a) @ d) / denom, 0.0, 1.0)
        proj = a + u[:, None] * d
        dist = np.linalg.norm(flat - proj, axis=1).astype("float32")
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_t = np.where(better, t_nodes[i] + u * (t_nodes[i + 1] - t_nodes[i]), best_t)
    return best_t.reshape(c.shape[:-1]), best_d.reshape(c.shape[:-1])


def ramp_colour(t: float) -> tuple[int, int, int]:
    """The ramp colour at arc-length position ``t`` — the inverse of
    :func:`ramp_position`, useful for legends and for testing the decoder."""
    import numpy as np

    pts, _seg, t_nodes = _ramp_geometry()
    channels = [np.interp(t, t_nodes, pts[:, channel]) for channel in range(3)]
    return tuple(int(v) for v in np.rint(channels))


def rate_from_position(t):
    """Provisional t -> mm/h, log-linear between :data:`RATE_ANCHORS`."""
    import numpy as np

    ts = np.asarray([a[0] for a in RATE_ANCHORS], dtype="float64")
    ls = np.log10([a[1] for a in RATE_ANCHORS])
    t = np.asarray(t, dtype="float64")
    return (10.0 ** np.interp(t, ts, ls)).astype("float32")


def _rgba(path: pathlib.Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"))
    return arr


def png_to_rate(path):
    """Decode an archived frame to a float32 mm/h array (NaN where no data).

    NaN means "no echo / outside coverage": the frames are fully transparent
    where Buienradar has nothing to draw, and any colour too far from the ramp
    is rejected rather than guessed at. Rates are PROVISIONAL — see the palette
    notes above.
    """
    import numpy as np

    arr = _rgba(pathlib.Path(path))
    t, dist = ramp_position(arr[..., :3])
    rate = rate_from_position(t)
    nodata = (arr[..., 3] == 0) | (dist > MAX_RAMP_DISTANCE)
    out = np.where(nodata, np.nan, rate).astype("float32")
    return out


def png_to_class(path):
    """Decode an archived frame to the published legend class index.

    0..4 index :data:`PALETTE`, -1 is no data. Unlike :func:`png_to_rate` this
    needs no assumption beyond "intensity increases along the ramp": the class
    edges are the swatch positions Buienradar publishes.
    """
    import numpy as np

    arr = _rgba(pathlib.Path(path))
    t, dist = ramp_position(arr[..., :3])
    edges = np.asarray([a[0] for a in RATE_ANCHORS[1:]], dtype="float32")
    cls = np.digitize(t, edges).astype("int8")
    nodata = (arr[..., 3] == 0) | (dist > MAX_RAMP_DISTANCE)
    cls[nodata] = -1
    return cls


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)


class HttpError(Exception):
    """A non-retryable HTTP status (e.g. a rotated API key -> 401/403)."""

    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


def _open_url(url: str, timeout: float) -> bytes:
    """Single GET. Tests monkeypatch this; nothing else touches urllib."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read()


def http_get(
    url: str,
    *,
    timeout: float = 20.0,
    retries: int = 3,
    backoff: float = 1.0,
    sleep=time.sleep,
) -> bytes:
    """GET with exponential backoff. Raises :class:`HttpError` for a 4xx that
    retrying cannot fix, the last network error otherwise."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _open_url(url, timeout)
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise HttpError(exc.code, url) from exc
            last = exc
        except NETWORK_ERRORS as exc:
            last = exc
        if attempt < retries - 1:
            delay = backoff * (2**attempt)
            LOG.warning("GET %s failed (%s); retrying in %.1fs", url, last, delay)
            sleep(delay)
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
def scrape_api_key(**kw) -> str | None:
    """Re-read ``window.apiKey`` from the public page."""
    try:
        html = http_get(PAGE_URL, **kw).decode("utf-8", errors="replace")
    except (HttpError, *NETWORK_ERRORS) as exc:
        LOG.warning("could not fetch %s for the api key (%s)", PAGE_URL, exc)
        return None
    m = API_KEY_RE.search(html)
    if not m:
        LOG.warning("no window.apiKey in %s", PAGE_URL)
        return None
    return m.group(1)


def api_key_path(root: pathlib.Path) -> pathlib.Path:
    return root / "apikey.txt"


def load_api_key(root: pathlib.Path) -> str:
    path = api_key_path(root)
    if path.exists():
        cached = path.read_text().strip()
        if cached:
            return cached
    return DEFAULT_API_KEY


def store_api_key(root: pathlib.Path, key: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    api_key_path(root).write_text(key + "\n")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
COMPACT = "%Y%m%d%H%M"
RUN_URL_RE = re.compile(r"/runs/webm/(\d{12})/(\d{12})\.png")
CURRENT_URL_RE = re.compile(r"/current/webm/(\d{12})\.png")


class Frame:
    """One archived image: a valid time (UTC), its URL and its run (if any)."""

    __slots__ = ("run", "url", "valid")

    def __init__(self, valid: dt.datetime, url: str, run: dt.datetime | None):
        self.valid = valid
        self.url = url
        self.run = run

    @property
    def valid_id(self) -> str:
        return self.valid.strftime(COMPACT)

    @property
    def run_id(self) -> str | None:
        return self.run.strftime(COMPACT) if self.run else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Frame({self.valid_id}Z, run={self.run_id}, {self.url})"


class Metadata:
    """Parsed metadata document."""

    def __init__(
        self,
        imagetype: str,
        time_offset_h: float,
        width: int,
        height: int,
        timestamp: dt.datetime,
        frames: list[Frame],
        raw: dict,
    ):
        self.imagetype = imagetype
        self.time_offset_h = time_offset_h
        self.width = width
        self.height = height
        self.timestamp = timestamp
        self.frames = frames
        self.raw = raw

    @property
    def run(self) -> dt.datetime | None:
        """The forecast run this document describes, if it is a forecast."""
        runs = {f.run for f in self.frames if f.run is not None}
        if not runs:
            return None
        if len(runs) > 1:
            LOG.warning("%s mixes %d runs: %s", self.imagetype, len(runs), sorted(runs))
        return min(runs)


def parse_timestamp(value: str) -> dt.datetime:
    """Parse a metadata timestamp. They are UTC — see the module docstring."""
    return dt.datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.UTC)


def local_from_utc(when: dt.datetime, time_offset_h: float) -> dt.datetime:
    """Apply ``timeOffset`` to get the clock time the site shows its users."""
    return when + dt.timedelta(hours=time_offset_h)


def amsterdam_offset_hours(when: dt.datetime) -> float:
    """Europe/Amsterdam UTC offset at ``when`` (UTC), in hours."""
    from zoneinfo import ZoneInfo

    return when.astimezone(ZoneInfo("Europe/Amsterdam")).utcoffset().total_seconds() / 3600.0


def check_time_offset(meta: Metadata) -> bool:
    """Warn if ``timeOffset`` is not the Amsterdam offset for the run time.

    A silent change here (or a Buienradar switch to local timestamps) would
    shift the whole archive by an hour or two, so it is checked every tick
    instead of being assumed. 1.0 in winter, 2.0 in summer.
    """
    expected = amsterdam_offset_hours(meta.timestamp)
    if abs(meta.time_offset_h - expected) > 1e-6:
        LOG.warning(
            "timeOffset %.1f != Europe/Amsterdam offset %.1f at %s — "
            "timestamp interpretation may have changed",
            meta.time_offset_h,
            expected,
            meta.timestamp.isoformat(),
        )
        return False
    return True


def check_run_id(meta: Metadata) -> bool:
    """Warn if the compact run id in the URLs is not the UTC ``timestamp``.

    This is the assertion the whole UTC naming scheme rests on.
    """
    run = meta.run
    if run is None:
        return True
    if run != meta.timestamp:
        LOG.warning(
            "forecast run id %s != document timestamp %s (delta %+.0f min) — "
            "verify the run-id timezone before trusting archive paths",
            run.strftime(COMPACT),
            meta.timestamp.strftime(COMPACT),
            (run - meta.timestamp).total_seconds() / 60.0,
        )
        return False
    return True


def parse_metadata(doc: dict) -> Metadata:
    """Turn a metadata document into :class:`Metadata`, raising on nonsense."""
    try:
        imagetype = str(doc["imagetype"])
        time_offset = float(doc["timeOffset"])
        width = int(doc["width"])
        height = int(doc["height"])
        timestamp = parse_timestamp(str(doc["timestamp"]))
        times = doc["times"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed metadata document: {exc}") from exc

    frames: list[Frame] = []
    for entry in times:
        try:
            url = str(entry["url"])
            valid = parse_timestamp(str(entry["timestamp"]))
        except (KeyError, TypeError, ValueError) as exc:
            LOG.warning("skipping malformed times[] entry %r (%s)", entry, exc)
            continue
        run = None
        m = RUN_URL_RE.search(url)
        if m:
            run = dt.datetime.strptime(m.group(1), COMPACT).replace(tzinfo=dt.UTC)
            url_valid = dt.datetime.strptime(m.group(2), COMPACT).replace(tzinfo=dt.UTC)
        else:
            m = CURRENT_URL_RE.search(url)
            url_valid = (
                dt.datetime.strptime(m.group(1), COMPACT).replace(tzinfo=dt.UTC) if m else None
            )
        if url_valid is not None and url_valid != valid:
            LOG.warning(
                "frame url id %s != times[].timestamp %s; trusting the url",
                url_valid.strftime(COMPACT),
                valid.strftime(COMPACT),
            )
            valid = url_valid
        frames.append(Frame(valid, url, run))

    if not frames:
        raise ValueError("metadata document carries no usable frames")
    frames.sort(key=lambda f: f.valid)
    return Metadata(imagetype, time_offset, width, height, timestamp, frames, doc)


def metadata_url(kind: str, key: str) -> str:
    feed = FEEDS[kind]
    return METADATA_URL.format(
        imagetype=feed["imagetype"],
        size=IMAGE_SIZE,
        forecast=feed["forecast"],
        history=feed["history"],
        key=key,
    )


def fetch_metadata(kind: str, root: pathlib.Path, **kw) -> tuple[dict, bytes, str]:
    """Fetch one metadata document, re-scraping a rotated API key once.

    Returns ``(document, raw_bytes, key_used)``.
    """
    key = load_api_key(root)
    try:
        payload = http_get(metadata_url(kind, key), **kw)
    except HttpError as exc:
        if exc.status not in (400, 401, 403):
            raise
        LOG.warning("metadata returned %d — re-scraping window.apiKey", exc.status)
        fresh = scrape_api_key(**kw)
        if not fresh:
            raise
        if fresh != key:
            store_api_key(root, fresh)
        key = fresh
        payload = http_get(metadata_url(kind, key), **kw)
    return json.loads(payload.decode("utf-8")), payload, key


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    path       TEXT PRIMARY KEY,   -- relative to the archive root
    kind       TEXT NOT NULL,
    run_id     TEXT,               -- NULL for composite frames
    valid_id   TEXT NOT NULL,
    url        TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    revision   INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_kind_valid ON frames (kind, valid_id);
CREATE INDEX IF NOT EXISTS frames_run ON frames (run_id);
CREATE TABLE IF NOT EXISTS meta (
    path       TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    first_valid TEXT NOT NULL,
    last_valid  TEXT NOT NULL,
    frames      INTEGER NOT NULL,
    first_seen  TEXT NOT NULL
);
"""


def open_index(root: pathlib.Path, *, create: bool = True) -> sqlite3.Connection:
    """Open (and migrate) the frame index.

    ``create=False`` is the --dry-run path: an existing index is still read, so
    the dry run reports what a real tick would actually download, but nothing
    is created on disk when the archive does not exist yet.
    """
    # The archive root is usually a CIFS mount (the storage box), where SQLite
    # cannot take file locks ("database is locked" on an empty file). The index
    # can therefore live elsewhere — a local disk — via PLUVIO_BUIENRADAR_EU_INDEX
    # (or --index); frames and metadata stay under ``root``.
    override = os.environ.get("PLUVIO_BUIENRADAR_EU_INDEX")
    target = pathlib.Path(override) if override else root / "index.sqlite"
    if not create and not target.exists():
        target = ":memory:"
    else:
        root.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def latest_meta_sha(conn: sqlite3.Connection, kind: str) -> str | None:
    row = conn.execute(
        "SELECT sha256 FROM meta WHERE kind = ? ORDER BY fetched_at DESC, path DESC LIMIT 1",
        (kind,),
    ).fetchone()
    return row["sha256"] if row else None


def frame_rows(conn: sqlite3.Connection, kind: str, valid_id: str, run_id: str | None):
    if run_id is None:
        return conn.execute(
            "SELECT * FROM frames WHERE kind = ? AND valid_id = ? AND run_id IS NULL"
            " ORDER BY revision",
            (kind, valid_id),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM frames WHERE kind = ? AND valid_id = ? AND run_id = ? ORDER BY revision",
        (kind, valid_id, run_id),
    ).fetchall()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def composite_relpath(valid: dt.datetime, revision: int = 0) -> str:
    suffix = "" if revision == 0 else f".r{revision}"
    return f"composite/{valid:%Y/%m/%d}/{valid:{COMPACT}}Z{suffix}.png"


def forecast_relpath(run: dt.datetime, valid: dt.datetime, revision: int = 0) -> str:
    suffix = "" if revision == 0 else f".r{revision}"
    return f"forecast/{run:%Y/%m/%d}/{run:{COMPACT}}Z/{valid:{COMPACT}}Z{suffix}.png"


def frame_relpath(kind: str, frame: Frame, revision: int = 0) -> str:
    if kind == "forecast":
        if frame.run is None:
            raise ValueError(f"forecast frame without a run id: {frame.url}")
        return forecast_relpath(frame.run, frame.valid, revision)
    return composite_relpath(frame.valid, revision)


def meta_relpath(kind: str, fetched: dt.datetime) -> str:
    return f"meta/{kind}/{fetched:%Y/%m/%d}/{fetched:%Y%m%d%H%M%S}Z.json"


def runs_ledger(root: pathlib.Path) -> pathlib.Path:
    return root / "forecast" / "runs.jsonl"


def write_atomic(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self) -> None:
        self.downloaded = 0
        self.skipped = 0
        self.revised = 0
        self.restored = 0
        self.failed = 0
        self.meta_stored = 0
        self.runs_new = 0

    def as_dict(self) -> dict:
        return {
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "revised": self.revised,
            "restored": self.restored,
            "failed": self.failed,
            "meta_stored": self.meta_stored,
            "runs_new": self.runs_new,
        }


def _store_metadata(
    conn: sqlite3.Connection,
    root: pathlib.Path,
    kind: str,
    payload: bytes,
    fetched: dt.datetime,
    stats: Stats,
    dry_run: bool,
) -> None:
    """Archive a metadata document, but only when it differs from the last one.

    Deduping by hash is what keeps a 5-minute cadence from writing 288
    identical documents a day while still capturing every change — including
    the ones that matter, like a run appearing or a frame list shrinking.
    """
    digest = sha256(payload)
    if latest_meta_sha(conn, kind) == digest:
        return
    rel = meta_relpath(kind, fetched)
    stats.meta_stored += 1
    if dry_run:
        LOG.info("[dry-run] would store metadata %s", rel)
        return
    write_atomic(root / rel, payload)
    conn.execute(
        "INSERT OR REPLACE INTO meta (path, kind, sha256, fetched_at) VALUES (?, ?, ?, ?)",
        (rel, kind, digest, fetched.isoformat()),
    )
    conn.commit()


def _record_run(
    conn: sqlite3.Connection,
    root: pathlib.Path,
    meta: Metadata,
    fetched: dt.datetime,
    stats: Stats,
    dry_run: bool,
) -> None:
    """Append a run to ``forecast/runs.jsonl`` the first time we see it.

    ``first_seen`` is the fetch instant, so differences between consecutive
    ``first_seen`` values are the *observed* publication cadence — that is the
    whole point of the ledger, and why the line is written on first sight and
    never rewritten. A run whose frame list later changes gets a second line
    with ``"update": true`` so the history stays append-only.
    """
    run = meta.run
    if run is None:
        return
    run_id = run.strftime(COMPACT)
    first_valid = meta.frames[0].valid_id
    last_valid = meta.frames[-1].valid_id
    count = len(meta.frames)

    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is not None and (row["frames"], row["last_valid"]) == (count, last_valid):
        return
    update = row is not None
    entry = {
        "run": run_id,
        "run_utc": run.isoformat(),
        "first_valid": first_valid,
        "last_valid": last_valid,
        "frames": count,
        "first_seen": (row["first_seen"] if update else fetched.isoformat()),
        "lead_min": [
            int((meta.frames[0].valid - run).total_seconds() // 60),
            int((meta.frames[-1].valid - run).total_seconds() // 60),
        ],
        "time_offset_h": meta.time_offset_h,
    }
    if update:
        entry["update"] = True
        entry["seen_at"] = fetched.isoformat()
    else:
        stats.runs_new += 1
    if dry_run:
        LOG.info("[dry-run] would append run %s to runs.jsonl", run_id)
        return
    ledger = runs_ledger(root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, first_valid, last_valid, frames, first_seen)"
        " VALUES (?, ?, ?, ?, ?)",
        (run_id, first_valid, last_valid, count, entry["first_seen"]),
    )
    conn.commit()
    if not update:
        LOG.info("new forecast run %sZ (%d frames, first seen %s)", run_id, count, fetched.isoformat())


def collect_frames(
    conn: sqlite3.Connection,
    root: pathlib.Path,
    kind: str,
    meta: Metadata,
    fetched: dt.datetime,
    stats: Stats,
    *,
    dry_run: bool = False,
    revision_window: int = 2,
    sleep_between: float = 0.2,
    sleep=time.sleep,
    http_kw: dict | None = None,
) -> None:
    """Download every frame we do not have yet; re-check the newest few.

    ``revision_window`` frames at the tail are re-fetched even when archived,
    because a composite frame is published ~25 min after its valid time and
    could be re-issued afterwards. A differing sha256 is archived alongside the
    original as ``<id>Z.rN.png`` — we never overwrite what we already
    published a hash for.
    """
    http_kw = http_kw or {}
    revision_ids = {f.valid_id for f in meta.frames[len(meta.frames) - revision_window :]}
    first = True

    for frame in meta.frames:
        rows = frame_rows(conn, kind, frame.valid_id, frame.run_id)
        recheck = frame.valid_id in revision_ids
        # An indexed frame whose file has gone missing counts as not archived,
        # so a deleted or half-written file heals on the next tick instead of
        # being skipped forever on the strength of its index row.
        present = [r for r in rows if (root / r["path"]).exists()]
        if present and not recheck:
            stats.skipped += 1
            continue
        rel = frame_relpath(kind, frame, 0)
        if rows and not present:
            LOG.warning("index has %s but the file is missing; refetching", rows[0]["path"])
        if dry_run:
            if not present:
                LOG.info("[dry-run] would download %s -> %s", frame.url, rel)
                stats.downloaded += 1
            else:
                stats.skipped += 1
            continue

        if not first and sleep_between:
            sleep(sleep_between)
        first = False
        try:
            payload = http_get(frame.url, **http_kw)
        except Exception as exc:  # one bad frame must not end the tick
            stats.failed += 1
            LOG.warning("frame %s failed (%s)", frame.url, exc)
            continue
        digest = sha256(payload)
        match = next((r for r in rows if r["sha256"] == digest), None)
        if match is not None:
            if (root / match["path"]).exists():
                stats.skipped += 1
                continue
            # Same bytes as a row we already have, but the file is gone:
            # restore it in place rather than minting a bogus revision.
            write_atomic(root / match["path"], payload)
            stats.restored += 1
            LOG.info("restored missing %s", match["path"])
            continue
        revision = (max((r["revision"] for r in rows), default=-1) + 1) if rows else 0
        rel = frame_relpath(kind, frame, revision)
        write_atomic(root / rel, payload)
        conn.execute(
            "INSERT OR REPLACE INTO frames"
            " (path, kind, run_id, valid_id, url, sha256, bytes, revision, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rel,
                kind,
                frame.run_id,
                frame.valid_id,
                frame.url,
                digest,
                len(payload),
                revision,
                fetched.isoformat(),
            ),
        )
        conn.commit()
        if revision:
            stats.revised += 1
            LOG.info("REVISED %s %sZ -> %s", kind, frame.valid_id, rel)
        else:
            stats.downloaded += 1


def collect(
    root: pathlib.Path,
    *,
    dry_run: bool = False,
    revision_window: int = 2,
    sleep_between: float = 0.2,
    now: dt.datetime | None = None,
    sleep=time.sleep,
    http_kw: dict | None = None,
) -> tuple[Stats, list[str]]:
    """One collection tick. Returns ``(stats, fatal_errors)``."""
    root = pathlib.Path(root)
    fetched = now or dt.datetime.now(dt.UTC).replace(microsecond=0)
    http_kw = http_kw or {}
    stats = Stats()
    fatal: list[str] = []

    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        write_atomic(
            root / "georeference.json",
            (json.dumps(georeference(), indent=2, sort_keys=True) + "\n").encode(),
        )
        write_atomic(root / "frame.pgw", world_file().encode())

    conn = open_index(root, create=not dry_run)
    try:
        for kind in FEEDS:
            try:
                doc, payload, _ = fetch_metadata(kind, root, **http_kw)
                meta = parse_metadata(doc)
            # Anything at all here (transport, HTTP status, JSON, schema) means
            # this feed is unusable this tick; report it and keep the other one.
            except Exception as exc:
                fatal.append(f"{kind}: {exc}")
                LOG.error("metadata for %s unavailable (%s)", kind, exc)
                continue

            if (meta.width, meta.height) != (WIDTH, HEIGHT):
                LOG.warning(
                    "%s image size %dx%d != expected %dx%d — georeference may be stale",
                    kind,
                    meta.width,
                    meta.height,
                    WIDTH,
                    HEIGHT,
                )
            check_time_offset(meta)
            check_run_id(meta)

            _store_metadata(conn, root, kind, payload, fetched, stats, dry_run)
            if kind == "forecast":
                _record_run(conn, root, meta, fetched, stats, dry_run)
            collect_frames(
                conn,
                root,
                kind,
                meta,
                fetched,
                stats,
                dry_run=dry_run,
                revision_window=revision_window,
                sleep_between=sleep_between,
                sleep=sleep,
                http_kw=http_kw,
            )
    finally:
        conn.close()
    return stats, fatal


# ---------------------------------------------------------------------------
# Cadence + integrity reporting
# ---------------------------------------------------------------------------
def read_runs(root: pathlib.Path) -> list[dict]:
    """Every first-sight run line from the ledger, oldest first."""
    ledger = runs_ledger(pathlib.Path(root))
    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            LOG.warning("skipping unparseable runs.jsonl line: %.80s", line)
            continue
        if entry.get("update"):
            continue
        out.append(entry)
    out.sort(key=lambda e: e.get("run", ""))
    return out


def cadence_summary(root: pathlib.Path, days: int | None = None) -> dict:
    """Summarise the observed forecast-run cadence from the ledger."""
    runs = read_runs(root)
    if days:
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
        runs = [r for r in runs if dt.datetime.fromisoformat(r["run_utc"]) >= cutoff]
    if not runs:
        return {"runs": 0}

    run_times = [dt.datetime.fromisoformat(r["run_utc"]) for r in runs]
    gaps = [(b - a).total_seconds() / 60.0 for a, b in itertools.pairwise(run_times)]
    lags = []
    for r, t in zip(runs, run_times, strict=False):
        seen = r.get("first_seen")
        if seen:
            lags.append((dt.datetime.fromisoformat(seen) - t).total_seconds() / 60.0)
    counts = sorted({r["frames"] for r in runs})

    def stat(values: list[float]) -> dict | None:
        if not values:
            return None
        s = sorted(values)
        return {
            "min": s[0],
            "median": s[len(s) // 2],
            "max": s[-1],
            "mean": sum(s) / len(s),
        }

    return {
        "runs": len(runs),
        "first_run": runs[0]["run"],
        "last_run": runs[-1]["run"],
        "gap_min": stat(gaps),
        # first_seen - run time. For runs we caught on their first tick this is
        # Buienradar's publication lag; for a run already on the site when the
        # collector started it is only an upper bound, so read it together with
        # gap_min rather than on its own.
        "first_seen_lag_min": stat(lags),
        "frames_per_run": counts,
        "lead_min": sorted({tuple(r["lead_min"]) for r in runs if "lead_min" in r}),
    }


def index_check(root: pathlib.Path) -> dict:
    """Cross-check the sqlite index against the files on disk."""
    root = pathlib.Path(root)
    conn = open_index(root)
    try:
        rows = conn.execute("SELECT * FROM frames").fetchall()
        missing, corrupt = [], []
        for row in rows:
            path = root / row["path"]
            if not path.exists():
                missing.append(row["path"])
                continue
            if sha256(path.read_bytes()) != row["sha256"]:
                corrupt.append(row["path"])
        indexed = {row["path"] for row in rows}
        on_disk = {
            str(p.relative_to(root))
            for kind in ("composite", "forecast")
            for p in (root / kind).rglob("*.png")
        }
        return {
            "frames_indexed": len(rows),
            "frames_on_disk": len(on_disk),
            "missing_files": sorted(missing),
            "sha_mismatch": sorted(corrupt),
            "unindexed_files": sorted(on_disk - indexed),
            "revisions": conn.execute(
                "SELECT COUNT(*) AS n FROM frames WHERE revision > 0"
            ).fetchone()["n"],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="buienradar_eu", description=__doc__.split("\n")[0])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--index",
        default=None,
        help="sqlite index path (default <root>/index.sqlite; put it on local disk when "
        "the root is a CIFS mount — same as PLUVIO_BUIENRADAR_EU_INDEX)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="fetch metadata and archive any new frames")
    c.add_argument("--root", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--sleep", type=float, default=0.2, help="seconds between image fetches")
    c.add_argument("--timeout", type=float, default=20.0)
    c.add_argument("--retries", type=int, default=3)
    c.add_argument(
        "--revision-window",
        type=int,
        default=2,
        help="re-fetch this many newest frames per feed to detect revisions",
    )

    d = sub.add_parser("cadence", help="summarise the forecast run cadence")
    d.add_argument("--root", required=True)
    d.add_argument("--days", type=int, default=None)

    v = sub.add_parser("verify", help="cross-check the index against the files on disk")
    v.add_argument("--root", required=True)

    p = sub.add_parser("decode", help="decode one archived PNG to mm/h (provisional)")
    p.add_argument("--png", required=True)
    p.add_argument("--out", default=None, help="write a .npy of the float32 mm/h array")

    args = parser.parse_args(argv)
    if args.index:
        os.environ["PLUVIO_BUIENRADAR_EU_INDEX"] = args.index
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    if args.cmd == "collect":
        stats, fatal = collect(
            pathlib.Path(args.root),
            dry_run=args.dry_run,
            revision_window=args.revision_window,
            sleep_between=args.sleep,
            http_kw={"timeout": args.timeout, "retries": args.retries},
        )
        LOG.info("tick %s", json.dumps(stats.as_dict(), sort_keys=True))
        if fatal:
            for msg in fatal:
                LOG.error("metadata failure — %s", msg)
            return 1
        if stats.failed:
            LOG.warning("%d frame(s) failed; they will be retried next tick", stats.failed)
        return 0

    if args.cmd == "cadence":
        print(json.dumps(cadence_summary(pathlib.Path(args.root), args.days), indent=2))
        return 0

    if args.cmd == "verify":
        report = index_check(pathlib.Path(args.root))
        print(json.dumps(report, indent=2))
        bad = report["missing_files"] or report["sha_mismatch"]
        return 1 if bad else 0

    if args.cmd == "decode":
        import numpy as np

        rate = png_to_rate(args.png)
        wet = np.isfinite(rate)
        print(
            json.dumps(
                {
                    "shape": list(rate.shape),
                    "wet_px": int(wet.sum()),
                    "max_mm_h": float(np.nanmax(rate)) if wet.any() else None,
                    "mean_wet_mm_h": float(np.nanmean(rate)) if wet.any() else None,
                    "note": "rates are PROVISIONAL — see tools/buienradar_eu.py palette notes",
                },
                indent=2,
            )
        )
        if args.out:
            np.save(args.out, rate)
        return 0

    return 2  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":
    sys.exit(main())
