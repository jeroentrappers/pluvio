"""KMI ALARO numerical-weather-prediction output, 24h+ horizon.

The opendata.meteo.be WMS exposes ALARO with a `time` dimension that runs
roughly 60 hours forward of the latest run at 1-hour cadence. We pull
`Total_precipitation` (kg m⁻² accumulated per step) as GeoTIFF — that's the
input to the 24h extension story.

Output: one GeoTIFF per requested timestep, named `alaro_TP_<YYYYmmddTHHMMZ>.tif`.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import sys
from datetime import datetime, timedelta, timezone

import httpx

LOG = logging.getLogger("pluvio.fetch_alaro")

WMS_URL = "https://opendata.meteo.be/service/alaro/wms"
USER_AGENT = "pluvio-research/0.1 (+https://github.com/appmire/pluvio)"


def fetch_time_dimension(client: httpx.Client, layer: str) -> list[datetime]:
    r = client.get(
        WMS_URL,
        params={"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"},
        timeout=30,
    )
    r.raise_for_status()
    xml = r.text
    block = _extract_layer_block(xml, layer)
    if block is None:
        raise RuntimeError(f"Layer {layer!r} not found in WMS capabilities")
    m = re.search(r'<Dimension[^>]*name="time"[^>]*>([\s\S]*?)</Dimension>', block)
    if not m:
        raise RuntimeError(f"No <Dimension name=time> on layer {layer!r}")
    return _expand_time_dim(m.group(1).strip())


def fetch_geotiff(
    client: httpx.Client,
    layer: str,
    when: datetime,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    out_path: pathlib.Path,
) -> None:
    minx, miny, maxx, maxy = bbox
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "format": "image/geotiff",
        "transparent": "true",
        "crs": "EPSG:4326",
        "bbox": f"{miny},{minx},{maxy},{maxx}",  # WMS 1.3 lat,lon order for EPSG:4326
        "width": str(size[0]),
        "height": str(size[1]),
        "time": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    r = client.get(WMS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="Total_precipitation")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="How many hours of forecast to download from the latest run.",
    )
    parser.add_argument("--out", default="data/alaro")
    parser.add_argument(
        "--bbox",
        default="0.0,48.5,11.0,56.0",
        help=(
            "minx,miny,maxx,maxy in EPSG:4326. Default matches the model's "
            "analysis grid (model/geo.py:bbox()) so reprojection onto the "
            "100x100 KNMI grid has full coverage."
        ),
    )
    parser.add_argument("--size", default="512x384")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    bbox = tuple(float(x) for x in args.bbox.split(","))
    if len(bbox) != 4:
        LOG.error("--bbox must be minx,miny,maxx,maxy")
        return 2
    width, height = (int(x) for x in args.size.lower().split("x"))

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(http2=True) as client:
        timesteps = fetch_time_dimension(client, args.layer)
        if not timesteps:
            LOG.error("No timesteps published for %s", args.layer)
            return 1
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=args.hours)
        wanted = [t for t in timesteps if now - timedelta(hours=1) <= t <= horizon]
        LOG.info(
            "ALARO %s — %d timesteps available, %d within +%dh window.",
            args.layer,
            len(timesteps),
            len(wanted),
            args.hours,
        )
        for when in wanted:
            stamp = when.strftime("%Y%m%dT%H%M%SZ")
            target = out_dir / f"alaro_{args.layer}_{stamp}.tif"
            if target.exists():
                continue
            try:
                fetch_geotiff(client, args.layer, when, bbox, (width, height), target)
                LOG.info("Wrote %s", target.name)
            except httpx.HTTPError as exc:
                LOG.warning("Failed %s: %s", stamp, exc)
    return 0


def _extract_layer_block(xml: str, layer_name: str) -> str | None:
    """Return the <Layer>…</Layer> block whose <Name> equals ``layer_name``."""
    pattern = re.compile(r"<Layer[^>]*>([\s\S]*?</Layer>)", re.M)
    for m in pattern.finditer(xml):
        block = m.group(1)
        name_m = re.search(r"<Name>([^<]+)</Name>", block)
        if name_m and name_m.group(1) == layer_name:
            return block
    return None


def _expand_time_dim(raw: str) -> list[datetime]:
    out: list[datetime] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "/" in token:
            start_s, end_s, period_s = token.split("/")
            start = _parse_iso(start_s)
            end = _parse_iso(end_s)
            step = _parse_iso8601_duration(period_s)
            cur = start
            while cur <= end:
                out.append(cur)
                cur = cur + step
        else:
            try:
                out.append(_parse_iso(token))
            except ValueError:
                pass
    return sorted(out)


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _parse_iso8601_duration(s: str) -> timedelta:
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", s)
    if not m:
        raise ValueError(f"Cannot parse ISO 8601 duration {s!r}")
    days, hours, minutes, seconds = (int(g or 0) for g in m.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


if __name__ == "__main__":
    sys.exit(main())
