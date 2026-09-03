"""Third-party point nowcasts, sampled and archived so the benchmark scoreboard
can put Buienradar (and later others) on the same ruler as our own models.

Buienradar publishes a free, unauthenticated point forecast at

    https://gpsgadget.buienradar.nl/data/raintext?lat=LAT&lon=LON

as plain text, one line per 5-minute step for roughly the next two hours:

    VVV|HH:MM

``VVV`` is a byte 0-255 on a log scale, ``HH:MM`` is Europe/Amsterdam local
clock time with no date. The conversion to a rain rate is

    mm/h = 10 ** ((VVV - 109) / 32)

so 109 -> 1.0 mm/h and 0 -> ~0.0004 mm/h (never exactly zero -- the feed has
no explicit "no rain" sentinel, it is just very small numbers). The byte
range formally extends to 255, which the formula maps to an enormous,
never-actually-observed rate; that is a property of the encoding, not a bug
in this reader.

Everything here is read-only and best-effort: a fetch failure returns
``None`` rather than raising, because a missing station should degrade
scoreboard coverage, not take down the sampler.

Usage:
    python -m tools.external_baselines sample --archive /path/to/archive
    python -m tools.external_baselines show --archive /path/to/archive --day 2026-09-03
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Protocol

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback, not expected here
    from backports.zoneinfo import ZoneInfo  # type: ignore

LOG = logging.getLogger("pluvio.external_baselines")

AMSTERDAM = ZoneInfo("Europe/Amsterdam")

BUIENRADAR_URL = "https://gpsgadget.buienradar.nl/data/raintext?lat={lat}&lon={lon}"

# ---------------------------------------------------------------------------
# Stations: ~20 well-spread points across Belgium and the Netherlands.
#
# Chosen to cover:
#   - the major population centres (so most users' "does this match what I
#     see out the window" checks land on a sampled point)
#   - a spread across the whole BE/NL landmass, not just the Randstad/Brussels
#     core, so convective cells over the edges of the domain are represented
#   - a few deliberately rural/coastal spots (Ostend, Vlissingen, Arlon,
#     Leeuwarden) since coastal squall lines and inland showers behave
#     differently and both need to show up in the scoreboard
# ---------------------------------------------------------------------------
STATIONS: list[tuple[str, float, float]] = [
    # Belgium
    ("Brussels", 50.8503, 4.3517),
    ("Antwerp", 51.2194, 4.4025),
    ("Ghent", 51.0543, 3.7174),
    ("Liege", 50.6326, 5.5797),
    ("Ostend", 51.2154, 2.9286),       # coastal
    ("Bruges", 51.2093, 3.2247),
    ("Charleroi", 50.4108, 4.4446),
    ("Namur", 50.4674, 4.8718),
    ("Hasselt", 50.9307, 5.3378),
    ("Arlon", 49.6833, 5.8167),        # rural Ardennes, southern tip
    # Netherlands
    ("Amsterdam", 52.3676, 4.9041),
    ("Rotterdam", 51.9244, 4.4777),
    ("The Hague", 52.0705, 4.3007),
    ("Utrecht", 52.0907, 5.1214),
    ("Eindhoven", 51.4416, 5.4697),
    ("Groningen", 53.2194, 6.5665),
    ("Maastricht", 50.8514, 5.6910),
    ("Leeuwarden", 53.2012, 5.7999),   # rural north
    ("Vlissingen", 51.4426, 3.5736),   # coastal Zeeland
    ("Enschede", 52.2215, 6.8937),     # eastern border
]


def value_to_mm_per_h(value: float) -> float:
    """Buienradar's 0-255 log-scale byte -> mm/h."""
    return 10.0 ** ((value - 109.0) / 32.0)


def _fetch_text(url: str, timeout: float = 15.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        LOG.warning("fetch failed for %s (%s)", url, exc)
        return None


def parse_raintext(text: str, issue_time_utc: dt.datetime) -> list[tuple[int, float]]:
    """Parse a raintext payload into (valid_epoch_utc, mm_per_h) pairs.

    ``issue_time_utc`` anchors the (dateless) local clock times onto real
    calendar dates: the first parsed line takes the issue time's local date,
    and every time a line's clock time is *earlier* than the previous line's
    the local date rolls forward by one day (handles the feed crossing local
    midnight, e.g. an issue near 23:50 local with lines running into 00:xx).

    Malformed or blank lines are skipped rather than raising.
    """
    if issue_time_utc.tzinfo is None:
        issue_time_utc = issue_time_utc.replace(tzinfo=dt.timezone.utc)

    local_date = issue_time_utc.astimezone(AMSTERDAM).date()
    prev_tod: dt.time | None = None
    out: list[tuple[int, float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        value_str, _, time_str = line.partition("|")
        try:
            value = float(value_str)
            hh, mm = time_str.split(":")
            tod = dt.time(int(hh), int(mm))
        except (ValueError, IndexError):
            LOG.debug("skipping malformed raintext line: %r", raw_line)
            continue

        if prev_tod is not None and tod < prev_tod:
            local_date += dt.timedelta(days=1)
        prev_tod = tod

        local_dt = dt.datetime.combine(local_date, tod, tzinfo=AMSTERDAM)
        valid_epoch = int(local_dt.astimezone(dt.timezone.utc).timestamp())
        out.append((valid_epoch, value_to_mm_per_h(value)))

    return out


class Source(Protocol):
    """Design hook for future third-party sources (UKMO, OPERA, ...)."""

    name: str

    def fetch_point(
        self, lat: float, lon: float, issue_time_utc: dt.datetime
    ) -> list[tuple[int, float]] | None:
        """Return [(valid_epoch_utc, mm_per_h), ...] or None on failure."""
        ...


class BuienradarSource:
    name = "buienradar"

    def fetch_point(
        self, lat: float, lon: float, issue_time_utc: dt.datetime
    ) -> list[tuple[int, float]] | None:
        url = BUIENRADAR_URL.format(lat=lat, lon=lon)
        text = _fetch_text(url)
        if text is None:
            return None
        rows = parse_raintext(text, issue_time_utc)
        return rows if rows else None


def sample_all(
    issue_time: dt.datetime | None = None,
    stations: list[tuple[str, float, float]] | None = None,
    source: "Source | None" = None,
    delay_s: float = 0.5,
) -> list[dict]:
    """Fetch every station and flatten to scoreboard-ready rows."""
    if issue_time is None:
        issue_time = dt.datetime.now(dt.timezone.utc)
    elif issue_time.tzinfo is None:
        issue_time = issue_time.replace(tzinfo=dt.timezone.utc)
    stations = STATIONS if stations is None else stations
    source = source or BuienradarSource()
    issue_epoch = int(issue_time.timestamp())

    rows: list[dict] = []
    for i, (name, lat, lon) in enumerate(stations):
        points = source.fetch_point(lat, lon, issue_time)
        if points:
            for valid_epoch, mm_per_h in points:
                rows.append(
                    {
                        "source": source.name,
                        "station": name,
                        "lat": lat,
                        "lon": lon,
                        "issue_epoch": issue_epoch,
                        "valid_epoch": valid_epoch,
                        "lead_min": round((valid_epoch - issue_epoch) / 60.0),
                        "mm_per_h": mm_per_h,
                    }
                )
        else:
            LOG.warning("no data for station %s", name)
        if delay_s and i < len(stations) - 1:
            time.sleep(delay_s)
    return rows


# ---------------------------------------------------------------------------
# Archive: one JSON-lines file per UTC day, keyed idempotently.
# ---------------------------------------------------------------------------

def _archive_path(root: pathlib.Path, source_name: str, day: dt.date) -> pathlib.Path:
    return pathlib.Path(root) / source_name / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl"


def append_archive(rows: list[dict], root: str | pathlib.Path) -> int:
    """Append rows to per-day JSONL files, skipping ones already recorded.

    Idempotency key is (station, issue_epoch, valid_epoch) within a day's
    file. Rows are grouped by the day (UTC) of their valid_epoch, since that
    is what a later "what actually fell" verification will join against.

    Returns the number of rows actually written.
    """
    root = pathlib.Path(root)
    by_day: dict[tuple[str, dt.date], list[dict]] = {}
    for row in rows:
        source_name = row.get("source", "unknown")
        day = dt.datetime.fromtimestamp(row["valid_epoch"], dt.timezone.utc).date()
        by_day.setdefault((source_name, day), []).append(row)

    written = 0
    for (source_name, day), day_rows in by_day.items():
        path = _archive_path(root, source_name, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_keys: set[tuple[str, int, int]] = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    existing_keys.add(
                        (rec.get("station"), rec.get("issue_epoch"), rec.get("valid_epoch"))
                    )

        new_lines = []
        for row in day_rows:
            key = (row.get("station"), row.get("issue_epoch"), row.get("valid_epoch"))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_lines.append(json.dumps(row, sort_keys=True))

        if new_lines:
            with path.open("a", encoding="utf-8") as fh:
                for line in new_lines:
                    fh.write(line + "\n")
            written += len(new_lines)

    return written


def load_archive(root: str | pathlib.Path, day: dt.date, source_name: str = "buienradar") -> list[dict]:
    """Read back one day's archive rows (empty list if the file doesn't exist)."""
    path = _archive_path(pathlib.Path(root), source_name, day)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = (0.1, 0.5, 1.0)


def score_against_truth(
    rows: list[dict],
    truth_lookup,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[int, dict]:
    """Per-lead verification stats.

    ``truth_lookup(lat, lon, valid_epoch) -> mm_per_h | None`` supplies the
    observed rate (typically from the composite) for each row; rows with no
    truth are skipped. Returns {lead_min: {n, bias, rmse, csi_<thr>: ...}}.
    """
    by_lead: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        truth = truth_lookup(row["lat"], row["lon"], row["valid_epoch"])
        if truth is None:
            continue
        by_lead.setdefault(row["lead_min"], []).append((row["mm_per_h"], truth))

    out: dict[int, dict] = {}
    for lead, pairs in by_lead.items():
        n = len(pairs)
        if n == 0:
            continue
        errs = [pred - obs for pred, obs in pairs]
        bias = sum(errs) / n
        rmse = (sum(e * e for e in errs) / n) ** 0.5

        stats = {"n": n, "bias": bias, "rmse": rmse}
        for thr in thresholds:
            hits = misses = false_alarms = 0
            for pred, obs in pairs:
                pred_yes = pred >= thr
                obs_yes = obs >= thr
                if pred_yes and obs_yes:
                    hits += 1
                elif pred_yes and not obs_yes:
                    false_alarms += 1
                elif not pred_yes and obs_yes:
                    misses += 1
            denom = hits + misses + false_alarms
            stats[f"csi_{thr}"] = (hits / denom) if denom else None
        out[lead] = stats
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_stations_arg(arg: str) -> list[tuple[str, float, float]]:
    if arg == "all":
        return STATIONS
    wanted = {s.strip() for s in arg.split(",") if s.strip()}
    by_name = {name: (name, lat, lon) for name, lat, lon in STATIONS}
    missing = wanted - set(by_name)
    if missing:
        raise SystemExit(f"unknown station(s): {sorted(missing)}")
    return [by_name[name] for name in wanted]


def _cmd_sample(args: argparse.Namespace) -> None:
    stations = _parse_stations_arg(args.stations)
    rows = sample_all(stations=stations, delay_s=args.delay)
    if args.dry_run:
        print(f"sampled {len(rows)} rows from {len(stations)} station(s), not written (--dry-run)")
        return
    written = append_archive(rows, args.archive)
    print(f"sampled {len(rows)} rows, wrote {written} new rows to {args.archive}")


def _cmd_show(args: argparse.Namespace) -> None:
    day = dt.date.fromisoformat(args.day)
    rows = load_archive(args.archive, day, source_name=args.source)
    print(f"{len(rows)} rows for {args.source} on {day}")
    for row in rows[: args.limit]:
        print(json.dumps(row, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="external_baselines")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="fetch every station and archive")
    p_sample.add_argument("--archive", required=True)
    p_sample.add_argument("--stations", default="all")
    p_sample.add_argument("--delay", type=float, default=0.5)
    p_sample.add_argument("--dry-run", action="store_true")
    p_sample.set_defaults(func=_cmd_sample)

    p_show = sub.add_parser("show", help="print an archived day")
    p_show.add_argument("--archive", required=True)
    p_show.add_argument("--day", required=True, help="YYYY-MM-DD")
    p_show.add_argument("--source", default="buienradar")
    p_show.add_argument("--limit", type=int, default=20)
    p_show.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
