"""Third-party point nowcasts, sampled and archived so the benchmark scoreboard
can put Buienradar (and later others) on the same ruler as our own models.

Buienradar publishes a free, unauthenticated point forecast at

    https://gpsgadget.buienradar.nl/data/raintext?lat=LAT&lon=LON

as plain text, one line per 5-minute step for roughly the next two hours:

    VVV|HH:MM

``VVV`` is a byte 0-255 on a log scale, ``HH:MM`` is Europe/Amsterdam local
clock time with no date. The conversion to a rain rate is

    mm/h = 10 ** ((VVV - 109) / 32)

so 109 -> 1.0 mm/h. Below ``DRY_FLOOR_MM_H`` (0.01 mm/h, value ~45 and
below) the result is snapped to exactly 0.0: the log scale never truly
reaches zero, and treating everything under that floor as "dry" keeps this
source comparable to whatever floor the other baselines and our own model
output use, rather than scoring sub-drizzle noise as a wet forecast. Bytes
outside 0-255 are not valid encodings and are rejected as malformed.

Because the feed carries no date, lines are anchored to real calendar
instants by walking forward from the first line and, at each step, picking
whichever (date, DST-fold) candidate lands closest to "previous line's
instant + 5 minutes". This is what makes local-midnight rollover and both
DST transitions (the repeated hour in October, the skipped hour in March)
resolve correctly and monotonically, instead of a "clock went backwards ->
new day" heuristic that misfires on fall-back nights.

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
import http.client
import json
import logging
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Protocol

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback, not expected here
    from backports.zoneinfo import ZoneInfo  # type: ignore

LOG = logging.getLogger("pluvio.external_baselines")

AMSTERDAM = ZoneInfo("Europe/Amsterdam")

BUIENRADAR_URL = "https://gpsgadget.buienradar.nl/data/raintext?lat={lat}&lon={lon}"

# Identify ourselves in the request so Buienradar's operators can throttle or
# block us by name instead of by IP if this sampler is ever a problem for
# them.
USER_AGENT = "pluvio-external-baselines/1.0 (+https://github.com/jeroentrappers/pluvio)"

DRY_FLOOR_MM_H = 0.01

# Sanity bounds on lead time, in minutes, used to catch anchoring bugs
# rather than to model anything physical about the feed.
LEAD_MIN_MIN = -5
LEAD_MIN_MAX = 130
# How far (minutes) the feed's first line is allowed to sit from the fetch
# instant before we treat the date anchoring as suspect and warn.
FIRST_LINE_SLACK_MIN = 10

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
    """Buienradar's 0-255 log-scale byte -> mm/h, floored below DRY_FLOOR_MM_H."""
    raw = 10.0 ** ((value - 109.0) / 32.0)
    return 0.0 if raw < DRY_FLOOR_MM_H else raw


def _fetch_text(url: str, timeout: float = 15.0) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        LOG.warning("fetch failed for %s (%s)", url, exc)
        return None


def _local_epoch(tod: dt.time, date: dt.date, fold: int) -> int:
    t = dt.time(tod.hour, tod.minute, fold=fold)
    local_dt = dt.datetime.combine(date, t, tzinfo=AMSTERDAM)
    return int(local_dt.timestamp())


def _closest_local_epoch(tod: dt.time, base_date: dt.date, target_epoch: int) -> tuple[int, dt.date]:
    """Among (base_date-1..+1, fold 0/1) candidates for ``tod``, return the
    (epoch, date) whose epoch is closest to ``target_epoch``.

    Trying both neighbouring dates handles local-midnight rollover; trying
    both folds handles the DST fall-back night, where a local clock time
    like 02:30 names two different real instants an hour apart. Picking by
    absolute distance to a target (rather than a "did the clock go
    backwards" heuristic) is what keeps epochs monotonic straight through
    both a rollover and a DST transition.
    """
    best_epoch = None
    best_date = None
    best_diff = None
    for delta in (-1, 0, 1):
        date = base_date + dt.timedelta(days=delta)
        for fold in (0, 1):
            epoch = _local_epoch(tod, date, fold)
            diff = abs(epoch - target_epoch)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_epoch = epoch
                best_date = date
    return best_epoch, best_date


def parse_raintext(text: str, issue_time_utc: dt.datetime) -> list[tuple[int, float]]:
    """Parse a raintext payload into (valid_epoch_utc, mm_per_h) pairs.

    ``issue_time_utc`` anchors the (dateless) local clock times onto real
    calendar instants: the first line is placed at whichever (date, fold)
    candidate lands closest to the issue instant, and every following line
    is placed closest to "previous line's instant + 5 minutes". That keeps
    the result monotonic across local midnight and across both directions
    of a DST transition -- see the module docstring.

    Lines are skipped (not raised on) if the ``VVV|HH:MM`` shape doesn't
    parse, or if ``VVV`` falls outside the feed's defined 0-255 byte range.
    """
    if issue_time_utc.tzinfo is None:
        issue_time_utc = issue_time_utc.replace(tzinfo=dt.timezone.utc)
    issue_epoch = int(issue_time_utc.timestamp())

    parsed_lines: list[tuple[dt.time, int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        value_str, _, time_str = line.partition("|")
        try:
            # VVV is documented as an integer byte 0-255. Require an actual
            # integer token (after stripping incidental whitespace) rather
            # than float() -- float() would happily accept "50.5" or "1e2"
            # as a byte value, which are not legal encodings of one.
            value = int(value_str.strip())
            hh, mm = time_str.split(":")
            tod = dt.time(int(hh), int(mm))
        except (ValueError, IndexError):
            LOG.debug("skipping malformed raintext line: %r", raw_line)
            continue
        if not (0 <= value <= 255):
            LOG.debug("skipping out-of-range raintext value: %r", raw_line)
            continue
        parsed_lines.append((tod, value))

    if not parsed_lines:
        return []

    base_date = issue_time_utc.astimezone(AMSTERDAM).date()
    target = issue_epoch
    out: list[tuple[int, float]] = []
    for tod, value in parsed_lines:
        epoch, base_date = _closest_local_epoch(tod, base_date, target)
        out.append((epoch, value_to_mm_per_h(value)))
        target = epoch + 300

    first_lead_min = (out[0][0] - issue_epoch) / 60.0
    if not (-FIRST_LINE_SLACK_MIN <= first_lead_min <= FIRST_LINE_SLACK_MIN):
        LOG.warning(
            "raintext first line lead %.1f min looks wrong (issue=%s, first=%s) -- "
            "date anchoring may be off",
            first_lead_min,
            issue_time_utc.isoformat(),
            dt.datetime.fromtimestamp(out[0][0], dt.timezone.utc).isoformat(),
        )

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
    """Fetch every station and flatten to scoreboard-ready rows.

    ``lead_min`` is measured from the feed's own first valid time (its t0),
    not from the fetch wall-clock: the feed floors its first line to the
    previous 5-minute step, so a wall-clock-referenced lead comes out
    fractional and off-grid ([-3, 2, 7, ...] rather than [0, 5, 10, ...]),
    which fragments per-lead scoreboard buckets across runs. ``issue_epoch``
    on each row is snapped to that same t0 for the same reason. The raw
    fetch instant is kept separately as ``fetch_epoch`` for latency/staleness
    diagnostics.

    With this feed's usual ~24 five-minute lines, the lead grid is
    0, 5, 10, ..., 115: benchmark leads 30/60/90 always land exactly on a
    grid point, and so does 120 whenever the feed happens to return a 25th
    line (it sometimes does; when it doesn't, 115 is the last lead sampled).
    """
    if issue_time is None:
        issue_time = dt.datetime.now(dt.timezone.utc)
    elif issue_time.tzinfo is None:
        issue_time = issue_time.replace(tzinfo=dt.timezone.utc)
    stations = STATIONS if stations is None else stations
    source = source or BuienradarSource()
    fetch_epoch = int(issue_time.timestamp())

    rows: list[dict] = []
    for i, (name, lat, lon) in enumerate(stations):
        points = source.fetch_point(lat, lon, issue_time)
        if points:
            t0 = points[0][0]
            for valid_epoch, mm_per_h in points:
                lead_min = round((valid_epoch - t0) / 60.0)
                if not (LEAD_MIN_MIN <= lead_min <= LEAD_MIN_MAX):
                    LOG.warning(
                        "station %s: lead %d min outside [%d, %d], dropping row",
                        name, lead_min, LEAD_MIN_MIN, LEAD_MIN_MAX,
                    )
                    continue
                rows.append(
                    {
                        "source": source.name,
                        "station": name,
                        "lat": lat,
                        "lon": lon,
                        "issue_epoch": t0,
                        "fetch_epoch": fetch_epoch,
                        "valid_epoch": valid_epoch,
                        "lead_min": lead_min,
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
#
# Files are keyed by the UTC day of each row's valid_epoch (not issue time --
# forecast_archive elsewhere in this repo keys by issue time instead). A
# scoreboard join against truth must therefore key on (valid_epoch, lat, lon)
# and, for any station near a UTC-day boundary, open both that day's file and
# its neighbours (day - 1 / day + 1), since a single fetch's rows can span
# the boundary.
# ---------------------------------------------------------------------------

def _archive_path(root: pathlib.Path, source_name: str, day: dt.date) -> pathlib.Path:
    return pathlib.Path(root) / source_name / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl"


def _index_path(jsonl_path: pathlib.Path) -> pathlib.Path:
    return jsonl_path.with_suffix(jsonl_path.suffix + ".idx")


def _index_header(jsonl_size: int, batch_count: int) -> str:
    return f"# size={jsonl_size} batches={batch_count}\n"


def _read_index(idx_path: pathlib.Path, jsonl_path: pathlib.Path) -> set[tuple[str, int]] | None:
    """Read the sidecar index, but only trust it if it is self-consistent
    with the JSONL it claims to describe.

    Returns ``None`` -- never a partial or wrong set -- when the index is
    missing, its header's recorded JSONL byte size doesn't match the
    JSONL's actual current size (the index is stale, e.g. from a run that
    appended to the JSONL but crashed or hit ENOSPC before updating the
    index), or the number of batch lines it actually contains doesn't match
    its own declared count (the index file itself was truncated or
    corrupted independent of the JSONL). Any of those must fall back to
    rebuilding from the JSONL rather than risk under-counting what's
    already written and re-appending duplicates.
    """
    if not idx_path.exists():
        return None
    try:
        jsonl_size = jsonl_path.stat().st_size if jsonl_path.exists() else 0
    except OSError:
        return None
    try:
        with idx_path.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip()
            if not header.startswith("# size="):
                return None
            fields = dict(part.split("=", 1) for part in header[2:].split() if "=" in part)
            if int(fields.get("size", -1)) != jsonl_size:
                return None
            declared_batches = int(fields["batches"]) if "batches" in fields else None

            keys: set[tuple[str, int]] = set()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                station, sep, issue_epoch_str = line.partition("|")
                if not sep:
                    return None
                keys.add((station, int(issue_epoch_str)))

            if declared_batches is not None and len(keys) != declared_batches:
                return None
            return keys
    except (OSError, ValueError):
        return None


def _distinct_batches_from_jsonl(path: pathlib.Path) -> set[tuple[str, int]]:
    """Ground truth: every distinct (station, issue_epoch) actually present
    in the day's JSONL, read directly from it. Used whenever the sidecar
    index can't be trusted -- never trust the index over the data it's
    supposed to summarize."""
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((rec.get("station"), rec.get("issue_epoch")))
    return keys


def _write_index_atomic(idx_path: pathlib.Path, jsonl_path: pathlib.Path, keys: set[tuple[str, int]]) -> None:
    jsonl_size = jsonl_path.stat().st_size if jsonl_path.exists() else 0
    tmp = idx_path.with_suffix(idx_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(_index_header(jsonl_size, len(keys)))
        for station, issue_epoch in sorted(keys, key=lambda k: (str(k[0]), k[1])):
            fh.write(f"{station}|{issue_epoch}\n")
    os.replace(tmp, idx_path)  # atomic on POSIX and Windows alike


def append_archive(rows: list[dict], root: str | pathlib.Path) -> int:
    """Append rows to per-day JSONL files, skipping ones already recorded.

    Idempotency is tracked per (station, issue_epoch) *batch* -- all the
    lead-time rows one station produces in one fetch share the same
    (snapped) issue_epoch and are written or skipped together -- via a
    small sidecar ``.idx`` file next to the day's ``.jsonl``, so that
    repeated calls across a day of 10-minute cadence usually don't need to
    re-parse an ever-growing JSONL file. That index is only ever a cache,
    though: it is verified self-consistent (see ``_read_index``) before
    being trusted, and rebuilt from the JSONL -- the actual ground truth --
    whenever it's missing, stale, or corrupted. The JSONL is appended to
    before the index is updated, so the worst a crash between the two can
    do is leave the index looking stale, which forces a rebuild on the next
    call rather than a silent duplicate.

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
        idx_path = _index_path(path)

        idx_result = _read_index(idx_path, path)
        rebuilt = idx_result is None
        written_batches = idx_result if idx_result is not None else _distinct_batches_from_jsonl(path)

        by_batch: dict[tuple[str, int], list[dict]] = {}
        for row in day_rows:
            key = (row.get("station"), row.get("issue_epoch"))
            by_batch.setdefault(key, []).append(row)

        new_lines = []
        final_batches = set(written_batches)
        for key, batch_rows in by_batch.items():
            if key in final_batches:
                continue
            final_batches.add(key)
            for row in batch_rows:
                new_lines.append(json.dumps(row, sort_keys=True))

        if new_lines:
            with path.open("a", encoding="utf-8") as fh:
                for line in new_lines:
                    fh.write(line + "\n")
            written += len(new_lines)

        # Persist the index whenever the JSONL changed (so its recorded
        # size stays current) or whenever we had to rebuild it above (so
        # that rebuild sticks instead of re-triggering on every call).
        if new_lines or rebuilt:
            _write_index_atomic(idx_path, path, final_batches)

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


def _is_missing(truth) -> bool:
    """True for None or NaN, regardless of scalar type.

    Truth values coming from this repo's composites are commonly numpy
    float32, not the builtin float, so an ``isinstance(truth, float)`` gate
    lets a numpy NaN straight through -- it then poisons bias/RMSE for the
    whole lead bucket and silently counts as an unearned false alarm. NaN is
    the only value that is never equal to itself, in any numeric dtype, so
    that comparison is the dtype-agnostic test.
    """
    if truth is None:
        return True
    try:
        return truth != truth
    except TypeError:
        return False


def score_against_truth(
    rows: list[dict],
    truth_lookup,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[int, dict]:
    """Per-lead verification stats.

    ``truth_lookup(lat, lon, valid_epoch) -> mm_per_h | None`` supplies the
    observed rate (typically from the composite) for each row; rows with no
    truth (``None`` or NaN -- a composite gap is reported as NaN, not None,
    and left unhandled it would poison bias/RMSE and read as an unearned
    false alarm) are skipped. Returns {lead_min: {n, bias, rmse, csi_<thr>: ...}}.
    """
    by_lead: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        truth = truth_lookup(row["lat"], row["lon"], row["valid_epoch"])
        if _is_missing(truth):
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
    order = [s.strip() for s in arg.split(",") if s.strip()]
    by_name = {name: (name, lat, lon) for name, lat, lon in STATIONS}
    missing = [n for n in order if n not in by_name]
    if missing:
        raise SystemExit(f"unknown station(s): {missing}")
    seen: set[str] = set()
    result = []
    for n in order:
        if n not in seen:
            seen.add(n)
            result.append(by_name[n])
    return result


def _cmd_sample(args: argparse.Namespace) -> None:
    if not args.archive:
        raise SystemExit("--archive is required (or set PLUVIO_EXTERNAL_ROOT)")
    stations = _parse_stations_arg(args.stations)
    rows = sample_all(stations=stations, delay_s=args.delay)
    if args.dry_run:
        print(f"sampled {len(rows)} rows from {len(stations)} station(s), not written (--dry-run)")
        return
    written = append_archive(rows, args.archive)
    print(f"sampled {len(rows)} rows, wrote {written} new rows to {args.archive}")


def _cmd_show(args: argparse.Namespace) -> None:
    if not args.archive:
        raise SystemExit("--archive is required (or set PLUVIO_EXTERNAL_ROOT)")
    day = dt.date.fromisoformat(args.day)
    rows = load_archive(args.archive, day, source_name=args.source)
    print(f"{len(rows)} rows for {args.source} on {day}")
    for row in rows[: args.limit]:
        print(json.dumps(row, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    default_archive = os.environ.get("PLUVIO_EXTERNAL_ROOT")
    parser = argparse.ArgumentParser(prog="external_baselines")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="fetch every station and archive")
    p_sample.add_argument("--archive", default=default_archive)
    p_sample.add_argument("--stations", default="all")
    p_sample.add_argument("--delay", type=float, default=0.5)
    p_sample.add_argument("--dry-run", action="store_true")
    p_sample.set_defaults(func=_cmd_sample)

    p_show = sub.add_parser("show", help="print an archived day")
    p_show.add_argument("--archive", default=default_archive)
    p_show.add_argument("--day", required=True, help="YYYY-MM-DD")
    p_show.add_argument("--source", default="buienradar")
    p_show.add_argument("--limit", type=int, default=20)
    p_show.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
