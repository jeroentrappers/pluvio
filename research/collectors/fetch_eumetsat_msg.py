"""Meteosat SEVIRI imagery + derived products via EUMETView WMS.

Endpoint:    https://view.eumetsat.int/geoserver/wms
Cadence:     15 minutes
Coverage:    Full disk; we crop to a Benelux bounding box

Layers we care about (others are listed at the end of this docstring):
  msg_fes:ir108        — Brightness temperature, IR 10.8 µm  (cloud-top temp)
  msg_fes:wv062        — Brightness temperature, WV 6.2 µm   (mid-troposphere moisture)
  msg_fes:gii_kindex   — K-index (thunderstorm potential, °C)
  msg_fes:gii_liftedindex — Lifted index (atmospheric instability)
  msg_fes:cth          — Cloud-top height (m)
  msg_fes:rdt          — Rapid Developing Thunderstorms (polygon mask)
  msg_fes:clm          — Cloud mask (binary)

`fes` = Full Earth Scan from MSG / Meteosat-11 over Europe + Africa.
Each GetMap returns a PNG / GeoTIFF for a single timestamp; we store
GeoTIFF so the data array is preserved (PNG colour-maps the values).

Licence: EUMETSAT data is free for non-commercial use. Cite "EUMETSAT".
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import sys
from datetime import datetime, timedelta, timezone

import httpx

LOG = logging.getLogger("pluvio.fetch_eumetsat_msg")

WMS_URL = "https://view.eumetsat.int/geoserver/wms"
USER_AGENT = "pluvio-research/0.1 (+https://github.com/appmire/pluvio)"

LAYERS_DEFAULT = (
    "msg_fes:ir108",
    "msg_fes:wv062",
    "msg_fes:gii_kindex",
    "msg_fes:gii_liftedindex",
    "msg_fes:cth",
    "msg_fes:rdt",
)


def fetch_time_dimension(client: httpx.Client, layer: str) -> list[datetime]:
    r = client.get(
        WMS_URL,
        params={"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"},
        timeout=60,
    )
    r.raise_for_status()
    xml = r.text
    block = _extract_layer_block(xml, layer)
    if block is None:
        raise RuntimeError(f"Layer {layer!r} not present in EUMETView WMS capabilities")
    m = re.search(r'<Dimension[^>]*name="time"[^>]*>([\s\S]*?)</Dimension>', block)
    if not m:
        raise RuntimeError(f"No <Dimension name=time> for {layer!r}")
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
        "bbox": f"{miny},{minx},{maxy},{maxx}",
        "width": str(size[0]),
        "height": str(size[1]),
        "time": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    r = client.get(WMS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default=",".join(LAYERS_DEFAULT),
                        help="Comma-separated EUMETView layer names.")
    parser.add_argument("--hours", type=int, default=6,
                        help="How many hours back from now to pull.")
    parser.add_argument("--start", default=None,
                        help="ISO UTC start; if given, overrides --hours.")
    parser.add_argument("--end", default=None, help="ISO UTC end.")
    parser.add_argument("--bbox", default="2.0,49.0,7.5,52.0",
                        help="minx,miny,maxx,maxy in EPSG:4326 (Benelux default).")
    parser.add_argument("--size", default="512x384")
    parser.add_argument("--out", default="data/msg")
    parser.add_argument(
        "--cadence-minutes",
        type=int,
        default=15,
        help=(
            "Subsample to one timestep every N minutes (default 15 = full WMS). "
            "Use 30 or 60 to cut volume when the full Meteosat cadence isn't needed."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    bbox = tuple(float(x) for x in args.bbox.split(","))
    if len(bbox) != 4:
        LOG.error("--bbox must be minx,miny,maxx,maxy")
        return 2
    width, height = (int(x) for x in args.size.lower().split("x"))
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    out_root = pathlib.Path(args.out)

    now = datetime.now(timezone.utc)
    start = _parse_iso(args.start) if args.start else now - timedelta(hours=args.hours)
    end = _parse_iso(args.end) if args.end else now

    with httpx.Client() as client:
        for layer in layers:
            LOG.info("Layer %s", layer)
            try:
                timesteps = fetch_time_dimension(client, layer)
            except Exception as exc:
                LOG.warning("Skip %s: %s", layer, exc)
                continue
            wanted = [t for t in timesteps if start <= t <= end]
            if args.cadence_minutes > 15:
                # Filter to N-minute boundaries (15 is the WMS native rate).
                wanted = [t for t in wanted if t.minute % args.cadence_minutes == 0]
            LOG.info(
                "  %d timesteps in window (%d total in WMS, cadence %d min)",
                len(wanted), len(timesteps), args.cadence_minutes,
            )

            layer_dir = out_root / layer.replace(":", "_")
            layer_dir.mkdir(parents=True, exist_ok=True)
            for when in wanted:
                stamp = when.strftime("%Y%m%dT%H%M%SZ")
                target = layer_dir / f"{layer.replace(':','_')}_{stamp}.tif"
                if target.exists():
                    continue
                try:
                    fetch_geotiff(client, layer, when, bbox, (width, height), target)
                except httpx.HTTPError as exc:
                    LOG.warning("  %s: %s", stamp, exc)
    return 0


def _extract_layer_block(xml: str, layer_name: str) -> str | None:
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
        raise ValueError(f"Unparseable ISO 8601 duration {s!r}")
    days, hours, minutes, seconds = (int(g or 0) for g in m.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


if __name__ == "__main__":
    sys.exit(main())
