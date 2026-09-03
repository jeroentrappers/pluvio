"""Public scoreboard (3.4): nightly job that scores "yesterday" (UTC) against
the composite truth and renders a static, self-contained HTML page.

Two things are scored, on the same truth, so they're comparable:

  (a) **grid**: every archived forecast run (``tools/forecast_archive.py``'s
      per-issue npz archive, kinds ``forecast``/``nowcast``) scored per lead
      against the QPE composite (``tools/qpe_archive.py``'s day-zarrs) —
      CSI/FSS/RMSE/mean_error via the same sufficient-statistic + block-
      bootstrap machinery ``tools/benchmark.py`` uses (``tools/_stats.py``),
      so a scoreboard number and a benchmark number mean the same thing.
  (b) **points**: Buienradar's station rows (``tools/external_baselines.py``'s
      JSONL archive) scored against the composite sampled AT the stations,
      and — for a like-for-like comparison — our own archived forecast
      sampled at those same stations and the same valid times, scored
      against the *same* truth values (not a second, independently-rounded
      lookup) via ``external_baselines.score_against_truth``.

Every scored day appends one JSON record to ``<out_root>/scoreboard/
YYYY/MM/DD.json`` (permanent, like every other archive in this repo — see
research/docs/ops_schedule.md) and, when ``--html`` is given, renders a
static page: one table per lead with model rows and bootstrap CI columns, an
"events yesterday" adequacy line, and a 30-day trend table read back from the
archive. Honest by construction: every table carries its own ``n`` and an
inadequate day is labelled, never silently dropped.

Usage (production defaults from ops_schedule.md — the storagebox mount):

    python -m tools.scoreboard --day 2026-09-02 \\
        --forecast-archive /mnt/storagebox/forecast_archive \\
        --qpe-root /mnt/storagebox/qpe \\
        --external-archive /mnt/storagebox/external_baselines \\
        --out-root /mnt/storagebox/scoreboard \\
        --html /mnt/storagebox/scoreboard/index.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import json
import logging
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.metrics import fss_components  # noqa: E402
from tools import external_baselines as eb  # noqa: E402
from tools._stats import SampleStats, block_bootstrap  # noqa: E402

LOG = logging.getLogger("pluvio.scoreboard")

DEFAULT_FORECAST_ARCHIVE = "/mnt/storagebox/forecast_archive"
DEFAULT_QPE_ROOT = "/mnt/storagebox/qpe"
DEFAULT_EXTERNAL_ARCHIVE = "/mnt/storagebox/external_baselines"
DEFAULT_OUT_ROOT = "/mnt/storagebox/scoreboard"

# Same fixed research-grid bounds QPE day-zarrs carry no attrs for (see
# backend/src/pluvio_backend/verify.py QPE_BOUNDS — same product, same
# convention, kept here rather than imported so this tool has no backend
# dependency).
DEFAULT_QPE_BOUNDS = (1.5, 48.9, 7.5, 52.5)
QPE_SLOTS_PER_DAY = 288

DEFAULT_KINDS = ("forecast", "nowcast")
DEFAULT_THRESHOLDS = (0.1, 1.0)
DEFAULT_FSS_SCALES = (1, 3)
DEFAULT_POINT_THRESHOLDS = (0.1, 1.0)


# ─────────────────────────────────────────────────────────── forecast archive


def iter_forecast_issues(archive_root: pathlib.Path, day: dt.date, kind: str):
    """Yield (issue_epoch, path) for every archived ``kind`` npz whose issue
    falls on the UTC calendar ``day`` — the archive is keyed by issue time
    (unlike the external-baselines archive, keyed by valid time), so a single
    day directory is exactly the day's issues."""
    day_dir = archive_root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    if not day_dir.is_dir():
        return
    for p in sorted(day_dir.glob(f"{kind}_*.npz")):
        hhmm = p.stem.split("_", 1)[1]
        try:
            ts = dt.datetime.combine(day, dt.time(int(hhmm[:2]), int(hhmm[2:4])),
                                      tzinfo=dt.UTC)
        except ValueError:
            continue
        yield int(ts.timestamp()), p


def load_forecast_npz(path: pathlib.Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"leads": [int(x) for x in z["leads"]],
            "rates": z["rates"].astype("float32"),
            "bounds": [float(x) for x in z["bounds"]],
            "issue_epoch": int(z["issue_epoch"])}


# ─────────────────────────────────────────────────────────────────── truth


def _area_resample(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Block-mean resample (downsampling only), no cv2 dependency — mirrors
    ``backend/src/pluvio_backend/verify.py``'s helper of the same name."""
    H, W = a.shape
    oh, ow = out_hw
    ri = np.clip(np.linspace(0, H, oh + 1).astype(int), 0, H)
    ci = np.clip(np.linspace(0, W, ow + 1).astype(int), 0, W)
    rows = np.add.reduceat(a, ri[:-1], axis=0) / np.maximum(np.diff(ri), 1)[:, None]
    return np.add.reduceat(rows, ci[:-1], axis=1) / np.maximum(np.diff(ci), 1)[None, :]


class QpeTruth:
    """Composite truth reader over ``tools/qpe_archive.py``'s day-zarrs, with
    a tiny same-process cache (at most a handful of days touched by one
    nightly run)."""

    def __init__(self, root: pathlib.Path, bounds=DEFAULT_QPE_BOUNDS):
        self.root = pathlib.Path(root)
        self.bounds = tuple(float(x) for x in bounds)
        self._slots: dict[dt.date, "np.ndarray | None"] = {}

    def _day_slots(self, day: dt.date):
        if day not in self._slots:
            zp = self.root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.zarr"
            if zp.exists():
                import zarr
                root = zarr.open_group(str(zp), mode="r")
                self._slots[day] = root["rate"] if "rate" in set(root.array_keys()) else None
            else:
                self._slots[day] = None
        return self._slots[day]

    def field(self, valid_epoch: int) -> np.ndarray | None:
        """Full (H, W) composite frame at ``valid_epoch``, or ``None`` if the
        day/slot isn't archived."""
        day = dt.datetime.fromtimestamp(valid_epoch, dt.UTC).date()
        arr = self._day_slots(day)
        if arr is None:
            return None
        slot = round((valid_epoch % 86400) / 300)
        if not 0 <= slot < arr.shape[0]:
            return None
        rate = np.asarray(arr[slot], dtype="float32")
        return rate if np.isfinite(rate).any() else None

    def field_on(self, valid_epoch: int, out_hw: tuple[int, int], out_bounds) -> np.ndarray | None:
        """Composite frame resampled onto ``out_bounds``/``out_hw`` (a
        forecast run's grid), NaN outside the composite's own domain."""
        rate = self.field(valid_epoch)
        if rate is None:
            return None
        qb = self.bounds
        H, W = rate.shape
        w, s, e, n = out_bounds
        c0 = int((w - qb[0]) / (qb[2] - qb[0]) * W)
        c1 = int((e - qb[0]) / (qb[2] - qb[0]) * W)
        r0 = int((qb[3] - n) / (qb[3] - qb[1]) * H)
        r1 = int((qb[3] - s) / (qb[3] - qb[1]) * H)
        if not (0 <= c0 < c1 <= W and 0 <= r0 < r1 <= H):
            return None
        return _area_resample(np.nan_to_num(rate[r0:r1, c0:c1]), out_hw)

    def point(self, lat: float, lon: float, valid_epoch: int) -> float | None:
        """Nearest-cell composite value at a lat/lon, or ``None`` if outside
        the domain / not archived / not finite."""
        rate = self.field(valid_epoch)
        if rate is None:
            return None
        qb = self.bounds
        H, W = rate.shape
        w, s, e, n = qb
        if not (w <= lon <= e and s <= lat <= n):
            return None
        c = int((lon - w) / (e - w) * W)
        r = int((n - lat) / (n - s) * H)
        c, r = min(max(c, 0), W - 1), min(max(r, 0), H - 1)
        v = float(rate[r, c])
        return v if np.isfinite(v) else None


# ────────────────────────────────────────────────────────────── grid scoring


def _grid_sample_record(pred_sel: np.ndarray, obs_sel: np.ndarray, pred_fss: np.ndarray,
                        obs_fss: np.ndarray, thresholds, fss_scales, issue_epoch: int) -> dict:
    """Sufficient-statistic record for one (sample, model) grid pair —
    categorical counts and pointwise error sums from the SELECTED (valid-only)
    cells, FSS from the full fill-consistent fields (a neighbourhood score, so
    it always needs the whole field, per tools/benchmark.py's convention)."""
    e = (pred_sel - obs_sel).astype("float64")
    cat: dict[float, tuple[int, int, int]] = {}
    fss: dict[float, dict[int, tuple[float, float]]] = {}
    for thr in thresholds:
        p_wet, o_wet = pred_sel >= thr, obs_sel >= thr
        hits = int((p_wet & o_wet).sum())
        misses = int((~p_wet & o_wet).sum())
        fa = int((p_wet & ~o_wet).sum())
        cat[thr] = (hits, misses, fa)
        fss[thr] = {sc: fss_components(pred_fss, obs_fss, threshold=thr, scale_px=sc)
                   for sc in fss_scales}
    return {"issue_epoch": issue_epoch, "n": int(pred_sel.size), "sum_e": float(e.sum()),
           "sum_abs_e": float(np.abs(e).sum()), "sum_sq_e": float((e ** 2).sum()),
           "cat": cat, "fss": fss}


def score_grid_day(day: dt.date, forecast_archive: pathlib.Path, truth: QpeTruth, *,
                   kinds=DEFAULT_KINDS, thresholds=DEFAULT_THRESHOLDS,
                   fss_scales=DEFAULT_FSS_SCALES, bootstrap_cfg: dict | None = None,
                   adequacy_threshold_mm_h: float = 5.0) -> dict:
    """Per-lead grid scores for every archived kind on ``day``, against the
    QPE composite resampled onto each run's own grid. Returns
    ``{kind: {lead_min: per_threshold_row}}`` (same row shape as
    ``tools/benchmark.py``'s ``per_threshold``), with bootstrap CIs merged in
    when ``bootstrap_cfg`` is given."""
    thresholds = [float(t) for t in thresholds]
    fss_scales = [int(s) for s in fss_scales]
    fss_fill = min(thresholds) - 1.0

    stats: dict[str, dict[int, SampleStats]] = defaultdict(dict)
    n_issues: dict[str, int] = defaultdict(int)
    issue_event_max: dict[int, float] = defaultdict(lambda: float("-inf"))

    for kind in kinds:
        for _issue_epoch, path in iter_forecast_issues(forecast_archive, day, kind):
            try:
                fc = load_forecast_npz(path)
            except Exception as exc:  # pragma: no cover - defensive, matches repo convention
                LOG.warning("unreadable forecast %s (%s)", path, exc)
                continue
            n_issues[kind] += 1
            for li, lead in enumerate(fc["leads"]):
                pred = fc["rates"][li]
                valid_epoch = fc["issue_epoch"] + lead * 60
                obs = truth.field_on(valid_epoch, pred.shape, fc["bounds"])
                if obs is None:
                    continue
                valid = np.isfinite(pred) & np.isfinite(obs)
                if not valid.any():
                    continue
                p_fss = np.where(valid, np.nan_to_num(pred), fss_fill).astype("float64")
                o_fss = np.where(valid, np.nan_to_num(obs), fss_fill).astype("float64")
                p_sel, o_sel = pred[valid].astype("float64"), obs[valid].astype("float64")
                record = _grid_sample_record(p_sel, o_sel, p_fss, o_fss, thresholds, fss_scales,
                                            fc["issue_epoch"])
                if lead not in stats[kind]:
                    stats[kind][lead] = SampleStats(thresholds, fss_scales)
                stats[kind][lead].add(**record)
                finite_obs = o_sel[np.isfinite(o_sel)]
                if finite_obs.size:
                    issue_event_max[fc["issue_epoch"]] = max(
                        issue_event_max[fc["issue_epoch"]], float(finite_obs.max()))

    leads_seen = sorted({lead for by_lead in stats.values() for lead in by_lead})
    results: dict[str, dict[str, dict]] = {}
    for kind in kinds:
        results[kind] = {}
        for lead in leads_seen:
            st = stats[kind].get(lead)
            if st is None or len(st) == 0:
                continue
            results[kind][str(lead)] = st.aggregate()

    if bootstrap_cfg:
        for lead in leads_seen:
            by_kind_stats = {k: stats[k][lead] for k in kinds if lead in stats[k] and len(stats[k][lead])}
            if len(by_kind_stats) < 1:
                continue
            boot = block_bootstrap(by_kind_stats, blocks_h=float(bootstrap_cfg["blocks_h"]),
                                  n_boot=int(bootstrap_cfg["n"]), ci=float(bootstrap_cfg["ci"]),
                                  seed=int(bootstrap_cfg["seed"]),
                                  ref_model=bootstrap_cfg.get("reference_model"))
            ci_by_kind = boot.get("ci", {}) if boot else {}
            for kind, row_by_thr in ((k, results[k].get(str(lead))) for k in by_kind_stats):
                if row_by_thr is None:
                    continue
                for thr in thresholds:
                    row_by_thr[str(thr)]["ci"] = ci_by_kind.get(kind, {}).get(str(thr))

    n_events = sum(1 for v in issue_event_max.values() if v > adequacy_threshold_mm_h)
    return {"results": results, "n_issues": dict(n_issues), "n_events": n_events}


# ───────────────────────────────────────────────────────────── point scoring


def _nearest_forecast_point(forecast_index: list[tuple[int, dict]], lat: float, lon: float,
                            lead_min: float, valid_epoch: int, *,
                            issue_tolerance_s: float = 150.0,
                            lead_tolerance_min: float = 2.5) -> float | None:
    """Sample our own archived forecast at ``(lat, lon)`` for the run/lead
    that best matches ``valid_epoch`` (issue + lead closest to the station
    row's own issue/lead), or ``None`` if nothing archived is close enough."""
    issue_epoch = round(valid_epoch - lead_min * 60)
    best = None
    best_diff = None
    for fc_issue, fc in forecast_index:
        if abs(fc_issue - issue_epoch) > issue_tolerance_s:
            continue
        w, s, e, n = fc["bounds"]
        if not (w <= lon <= e and s <= lat <= n):
            continue
        for li, lead in enumerate(fc["leads"]):
            fc_valid = fc_issue + lead * 60
            diff = abs(fc_valid - valid_epoch)
            if diff > lead_tolerance_min * 60:
                continue
            if best_diff is None or diff < best_diff:
                rate = fc["rates"][li]
                H, W = rate.shape
                c = min(max(int((lon - w) / (e - w) * W), 0), W - 1)
                r = min(max(int((n - lat) / (n - s) * H), 0), H - 1)
                v = float(rate[r, c])
                if np.isfinite(v):
                    best_diff = diff
                    best = v
    return best


def score_points_day(day: dt.date, external_archive: pathlib.Path, forecast_archive: pathlib.Path,
                     truth: QpeTruth, *, kind: str = "forecast",
                     thresholds=DEFAULT_POINT_THRESHOLDS,
                     source_name: str = "buienradar") -> dict:
    """Buienradar vs our own forecast, scored at the SAME station points and
    the SAME truth values (a single truth lookup feeds both series), for a
    like-for-like comparison. Returns
    ``{"buienradar": {...}, "ours": {...}, "n_matched": int}``."""
    rows = eb.load_archive(external_archive, day, source_name=source_name)
    forecast_index = [(issue_epoch, load_forecast_npz(p))
                      for issue_epoch, p in iter_forecast_issues(forecast_archive, day, kind)]

    truth_by_key: dict[tuple, float] = {}
    matched: list[dict] = []
    for row in rows:
        key = (row["station"], row["valid_epoch"])
        t = truth.point(row["lat"], row["lon"], row["valid_epoch"])
        if t is None:
            continue
        ours = _nearest_forecast_point(forecast_index, row["lat"], row["lon"],
                                       row["lead_min"], row["valid_epoch"])
        if ours is None:
            continue
        truth_by_key[key] = t
        matched.append({**row, "truth": t, "ours_mm_per_h": ours})

    def _rows_of(value_key: str) -> list[dict]:
        return [{"lat": m["lat"], "lon": m["lon"], "valid_epoch": m["valid_epoch"],
                "lead_min": m["lead_min"], "mm_per_h": m[value_key]} for m in matched]

    def _truth_lookup(lat, lon, valid_epoch):
        # Every argument here came straight out of `matched`, keyed exactly
        # as `truth_by_key` was built, so this can never diverge from the
        # value the buienradar/ours split used — the two series' scores are
        # therefore built from bit-identical truth, not two lookups that
        # happen to usually agree.
        for m in matched:
            if m["lat"] == lat and m["lon"] == lon and m["valid_epoch"] == valid_epoch:
                return m["truth"]
        return None

    buienradar_scores = eb.score_against_truth(_rows_of("mm_per_h"), _truth_lookup, thresholds=thresholds)
    ours_scores = eb.score_against_truth(_rows_of("ours_mm_per_h"), _truth_lookup, thresholds=thresholds)
    stations = sorted({m["station"] for m in matched})
    return {"buienradar": {str(k): v for k, v in buienradar_scores.items()},
           "ours": {str(k): v for k, v in ours_scores.items()},
           "n_matched": len(matched), "stations": stations}


# ──────────────────────────────────────────────────────────────────── record


def build_record(day: dt.date, *, grid: dict, points: dict, adequacy: dict, config: dict) -> dict:
    return {
        "day": day.isoformat(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "grid": grid,
        "points": points,
        "adequacy": adequacy,
        "config": config,
    }


def _nan_to_null(obj):
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


def archive_path(out_root: pathlib.Path, day: dt.date) -> pathlib.Path:
    return pathlib.Path(out_root) / "scoreboard" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.json"


def write_record(record: dict, out_root: pathlib.Path) -> pathlib.Path:
    day = dt.date.fromisoformat(record["day"])
    path = archive_path(out_root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_nan_to_null(record), indent=2, allow_nan=False))
    return path


def load_record(out_root: pathlib.Path, day: dt.date) -> dict | None:
    path = archive_path(out_root, day)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_trend(out_root: pathlib.Path, end_day: dt.date, n_days: int = 30) -> list[dict]:
    """Up to ``n_days`` most recent daily records ending at ``end_day``
    (inclusive), oldest first — missing days are simply absent, not padded,
    so the trend table's row count is itself an honesty signal."""
    out = []
    for i in range(n_days - 1, -1, -1):
        day = end_day - dt.timedelta(days=i)
        rec = load_record(out_root, day)
        if rec is not None:
            out.append(rec)
    return out


# ────────────────────────────────────────────────────────────────────── html


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


def _fmt_ci(row: dict, key: str) -> str:
    val = _fmt(row.get(key))
    ci = row.get("ci")
    if not ci or ci.get(key) is None:
        return val
    lo, hi = ci[key].get("ci_lo"), ci[key].get("ci_hi")
    return f"{val} [{_fmt(lo)}, {_fmt(hi)}]"


_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:960px;
     margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fff;}
h1{font-size:1.4rem;} h2{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;
   padding-bottom:.25rem;}
table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem;font-size:.85rem;}
th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:right;}
th:first-child,td:first-child{text-align:left;}
th{background:#f4f4f4;}
.adequate{color:#0a7d2c;font-weight:600;}
.inadequate{color:#b3261e;font-weight:600;}
.meta{color:#555;font-size:.85rem;}
.n{color:#888;font-size:.8rem;}
"""


def _grid_table_html(results: dict) -> str:
    leads = sorted({int(lead) for by_kind in results.values() for lead in by_kind}, key=int)
    if not leads:
        return "<p class='n'>no grid scores for this day.</p>"
    out = []
    for lead in leads:
        out.append(f"<h3>lead {lead} min</h3>")
        out.append("<table><tr><th>model</th><th>threshold</th><th>CSI</th><th>RMSE</th>"
                   "<th>mean_error</th><th>FSS</th><th>n</th></tr>")
        for kind, by_lead in results.items():
            row_by_thr = by_lead.get(str(lead))
            if not row_by_thr:
                continue
            for thr, row in sorted(row_by_thr.items(), key=lambda kv: float(kv[0])):
                fss_txt = ", ".join(f"{k}px={_fmt(v)}" for k, v in row["fss"].items())
                out.append(
                    f"<tr><td>{html_lib.escape(kind)}</td><td>{thr}</td>"
                    f"<td>{_fmt_ci(row, 'csi')}</td><td>{_fmt_ci(row, 'rmse')}</td>"
                    f"<td>{_fmt(row['mean_error'])}</td><td>{html_lib.escape(fss_txt)}</td>"
                    f"<td>{row.get('n_valid_cells', 0)}</td></tr>")
        out.append("</table>")
    return "\n".join(out)


def _points_table_html(points: dict) -> str:
    buien, ours = points.get("buienradar", {}), points.get("ours", {})
    leads = sorted({int(k) for k in buien} | {int(k) for k in ours})
    if not leads:
        return "<p class='n'>no point scores for this day.</p>"
    out = ["<table><tr><th>model</th><th>lead (min)</th><th>n</th><th>bias</th><th>RMSE</th>"
          "<th>CSI@0.1</th><th>CSI@1.0</th></tr>"]
    for lead in leads:
        for name, by_lead in (("buienradar", buien), ("ours (same stations)", ours)):
            row = by_lead.get(str(lead))
            if not row:
                continue
            out.append(
                f"<tr><td>{html_lib.escape(name)}</td><td>{lead}</td><td>{row['n']}</td>"
                f"<td>{_fmt(row['bias'])}</td><td>{_fmt(row['rmse'])}</td>"
                f"<td>{_fmt(row.get('csi_0.1'))}</td><td>{_fmt(row.get('csi_1.0'))}</td></tr>")
    out.append("</table>")
    out.append(f"<p class='n'>{points.get('n_matched', 0)} matched station-times across "
              f"{len(points.get('stations', []))} stations "
              f"(identical truth samples for both rows above).</p>")
    return "\n".join(out)


def _trend_table_html(trend: list[dict]) -> str:
    if not trend:
        return "<p class='n'>no prior days in the archive yet.</p>"
    out = ["<table><tr><th>day</th><th>events</th><th>adequate</th>"
          "<th>grid issues</th><th>points matched</th></tr>"]
    for rec in trend:
        adq = rec.get("adequacy", {})
        cls = "adequate" if adq.get("adequate") else "inadequate"
        n_issues = sum(rec.get("grid", {}).get("n_issues", {}).values())
        n_matched = rec.get("points", {}).get("n_matched", 0)
        out.append(
            f"<tr><td>{rec['day']}</td><td>{adq.get('n_events', 0)}</td>"
            f"<td class='{cls}'>{'yes' if adq.get('adequate') else 'no'}</td>"
            f"<td>{n_issues}</td><td>{n_matched}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def render_html(record: dict, trend: list[dict]) -> str:
    adq = record.get("adequacy", {})
    adequate = bool(adq.get("adequate"))
    cls = "adequate" if adequate else "inadequate"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pluvio scoreboard — {record['day']}</title>
<style>{_CSS}</style></head>
<body>
<h1>Pluvio scoreboard — {record['day']}</h1>
<p class="meta">generated {html_lib.escape(record.get('generated_at', ''))}</p>
<p class="{cls}">events yesterday: {adq.get('n_events', 0)} issue-times with domain max
truth &gt; {adq.get('threshold_mm_h')} mm/h (min {adq.get('min_events')} required) &mdash;
<strong>{'adequate' if adequate else 'NOT adequate — read this day&#39;s numbers with that in mind'}</strong></p>
<h2>Grid scores (per lead, per model)</h2>
{_grid_table_html(record.get('grid', {}).get('results', {}))}
<h2>Point scores at Buienradar stations</h2>
{_points_table_html(record.get('points', {}))}
<h2>30-day trend</h2>
{_trend_table_html(trend)}
</body></html>
"""


# ─────────────────────────────────────────────────────────────────────── run


def run(day: dt.date, *, forecast_archive: pathlib.Path, qpe_root: pathlib.Path,
       external_archive: pathlib.Path, out_root: pathlib.Path, kinds=DEFAULT_KINDS,
       point_kind: str = "forecast", thresholds=DEFAULT_THRESHOLDS,
       fss_scales=DEFAULT_FSS_SCALES, point_thresholds=DEFAULT_POINT_THRESHOLDS,
       qpe_bounds=DEFAULT_QPE_BOUNDS, bootstrap_cfg: dict | None = None,
       adequacy_threshold_mm_h: float = 5.0, adequacy_min_events: int = 5) -> dict:
    truth = QpeTruth(qpe_root, bounds=qpe_bounds)

    grid = score_grid_day(day, forecast_archive, truth, kinds=kinds, thresholds=thresholds,
                          fss_scales=fss_scales, bootstrap_cfg=bootstrap_cfg,
                          adequacy_threshold_mm_h=adequacy_threshold_mm_h)
    points = score_points_day(day, external_archive, forecast_archive, truth,
                              kind=point_kind, thresholds=point_thresholds)

    n_events = grid["n_events"]
    adequacy = {
        "threshold_mm_h": float(adequacy_threshold_mm_h),
        "min_events": int(adequacy_min_events),
        "n_events": int(n_events),
        "adequate": n_events >= adequacy_min_events,
    }
    config = {
        "kinds": list(kinds), "point_kind": point_kind,
        "thresholds_mm_h": [float(t) for t in thresholds],
        "fss_scales_px": [int(s) for s in fss_scales],
        "point_thresholds_mm_h": [float(t) for t in point_thresholds],
        "qpe_bounds": [float(x) for x in qpe_bounds],
        "bootstrap": bootstrap_cfg,
    }
    record = build_record(day, grid=grid, points=points, adequacy=adequacy, config=config)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day", default=None,
                  help="UTC date YYYY-MM-DD to score (default: yesterday UTC)")
    p.add_argument("--forecast-archive", default=DEFAULT_FORECAST_ARCHIVE)
    p.add_argument("--qpe-root", default=DEFAULT_QPE_ROOT)
    p.add_argument("--external-archive", default=DEFAULT_EXTERNAL_ARCHIVE)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--html", default=None, help="optional output HTML path")
    p.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    p.add_argument("--point-kind", default="forecast")
    p.add_argument("--thresholds-mm-h", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    p.add_argument("--fss-scales-px", default=",".join(str(s) for s in DEFAULT_FSS_SCALES))
    p.add_argument("--point-thresholds-mm-h", default=",".join(str(t) for t in DEFAULT_POINT_THRESHOLDS))
    p.add_argument("--qpe-bounds", default=",".join(str(x) for x in DEFAULT_QPE_BOUNDS),
                  help="west,south,east,north")
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--bootstrap-ci", type=float, default=0.9)
    p.add_argument("--bootstrap-blocks-h", type=float, default=6.0)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--bootstrap-reference-model", default=None)
    p.add_argument("--adequacy-threshold-mm-h", type=float, default=5.0)
    p.add_argument("--adequacy-min-events", type=int, default=5)
    p.add_argument("--trend-days", type=int, default=30)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    day = (dt.date.fromisoformat(args.day) if args.day
          else dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1))

    bootstrap_cfg = {"n": args.bootstrap_n, "ci": args.bootstrap_ci,
                     "blocks_h": args.bootstrap_blocks_h, "seed": args.bootstrap_seed,
                     "reference_model": args.bootstrap_reference_model}

    record = run(
        day,
        forecast_archive=pathlib.Path(args.forecast_archive),
        qpe_root=pathlib.Path(args.qpe_root),
        external_archive=pathlib.Path(args.external_archive),
        out_root=pathlib.Path(args.out_root),
        kinds=tuple(args.kinds.split(",")),
        point_kind=args.point_kind,
        thresholds=tuple(float(x) for x in args.thresholds_mm_h.split(",")),
        fss_scales=tuple(int(x) for x in args.fss_scales_px.split(",")),
        point_thresholds=tuple(float(x) for x in args.point_thresholds_mm_h.split(",")),
        qpe_bounds=tuple(float(x) for x in args.qpe_bounds.split(",")),
        bootstrap_cfg=bootstrap_cfg,
        adequacy_threshold_mm_h=args.adequacy_threshold_mm_h,
        adequacy_min_events=args.adequacy_min_events,
    )

    out_root = pathlib.Path(args.out_root)
    path = write_record(record, out_root)
    LOG.info("wrote %s", path)

    if args.html:
        trend = load_trend(out_root, day, n_days=args.trend_days)
        html_path = pathlib.Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(record, trend))
        LOG.info("wrote %s", html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
