"""Score the served composite against the best known composites, region by region.

Truth is always an independent gauge network, never another radar product:
  DE  DWD 10-min tipping-bucket stations (opendata, near-real-time 'now' files)
      -> ours vs RADOLAN RY (DWD's national 5-min composite) vs OPERA
  UK  Environment Agency 15-min gauges (OGL, ~1000 stations)
      -> ours (= UKMO national composite through our fill chain) vs OPERA

Windows are gauge-native (10 min DE / 15 min UK) over the served npz's own span,
ours = mean served rate in the window, RADOLAN = mean of its 5-min rates, OPERA =
the 15-min RATE field covering the window end. Bootstrap CIs by resampling
station-times, same protocol as the BE/NL evaluation.
"""
import csv, glob, io, json, pathlib, sys, zipfile
import datetime as dt
import numpy as np

import os
NPZ = os.environ.get("EVAL_NPZ", "/opt/pluvio/serve/observed.npz")
THRESHOLDS = (0.1, 0.5, 1.0, 2.0)

z = np.load(NPZ)
TIMES = z["times"].astype("int64")
RATES = z["rates"].astype("float32")
W, S, E, N = [float(x) for x in z["bounds"]]
H, WD = RATES.shape[1:]

def ours_window(lat, lon, t_end, span_s, halo=1):
    c = int((lon - W) / (E - W) * WD); r = int((N - lat) / (N - S) * H)
    if not (0 <= c < WD and 0 <= r < H): return np.nan
    m = (TIMES > t_end - span_s) & (TIMES <= t_end)
    if not m.any(): return np.nan
    blk = RATES[m][:, max(0, r-halo):r+halo+1, max(0, c-halo):c+halo+1]
    flat = blk.reshape(blk.shape[0], -1)
    if not np.isfinite(flat).any(): return np.nan
    return float(np.nanmean(np.nanmax(flat, axis=1)))

# --- OPERA: nearest 15-min RATE tif at/before window end -------------------------
import rasterio
from pyproj import Transformer
_OPERA = {}
def opera_at(lat, lon, t_end):
    t = dt.datetime.fromtimestamp(t_end, dt.UTC)
    t -= dt.timedelta(minutes=t.minute % 15, seconds=t.second)
    key = t.strftime("%Y%m%dT%H%M")
    if key not in _OPERA:
        fp = glob.glob(f"/mnt/storagebox/opera/RATE/{t:%Y/%m/%d}/{key}_RATE.tif*")
        if not fp:
            _OPERA[key] = None
        else:
            src = rasterio.open(fp[0])
            a = src.read(1).astype("float32")
            if src.nodata is not None: a[a == src.nodata] = 0.0
            _OPERA[key] = (a, src.transform,
                           Transformer.from_crs("EPSG:4326", src.crs, always_xy=True))
    got = _OPERA[key]
    if got is None: return np.nan
    a, tr, tf = got
    x, y = tf.transform(lon, lat)
    r, c = rasterio.transform.rowcol(tr, x, y)
    if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]): return np.nan
    blk = a[max(0, r-1):r+2, max(0, c-1):c+2]
    return float(blk.max())

# --- RADOLAN RY: mean of 5-min rates in window ------------------------------------
import wradlib.io as wio
import wradlib.georef as wgeo
from scipy.spatial import cKDTree
_RY = {}
_RY_TREE = None
def _ry_field(stamp):  # stamp = "YYmmddHHMM"
    if stamp not in _RY:
        fp = f"/tmp/radolan_ry/raa01-ry_10000-{stamp}-dwd---bin.bz2"
        try:
            d, meta = wio.read_radolan_composite(fp)
            d = np.asarray(d, "float32")
            d[d < 0] = np.nan
            _RY[stamp] = d * 12.0        # mm/5min -> mm/h
        except Exception:
            _RY[stamp] = None
    return _RY[stamp]
def radolan_window(lat, lon, t_end, span_s):
    global _RY_TREE
    vals = []
    for k in range(span_s // 300):
        t = dt.datetime.fromtimestamp(t_end - k * 300, dt.UTC)
        f = _ry_field(t.strftime("%y%m%d%H%M"))
        if f is None: continue
        if _RY_TREE is None:
            grid = wgeo.get_radolan_grid(*f.shape, wgs84=True)
            _RY_TREE = cKDTree(np.column_stack([grid[..., 0].ravel(),
                                                grid[..., 1].ravel()]))
        _, idx = _RY_TREE.query([lon, lat], k=9)
        v = np.nanmax(f.ravel()[idx])
        if np.isfinite(v): vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan

# --- scoring ----------------------------------------------------------------------
def csi(pred, obs, thr):
    p, o = pred > thr, obs > thr
    hit = int((p & o).sum()); miss = int((~p & o).sum()); fa = int((p & ~o).sum())
    return hit / max(1, hit + miss + fa)
def boot_diff(a, b, obs, thr, n=2000, seed=7):
    rng = np.random.default_rng(seed); idx = np.arange(len(obs)); out = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        out.append(csi(a[i], obs[i], thr) - csi(b[i], obs[i], thr))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi
def report(region, rows, competitors):
    if not rows:
        print(f"=== {region}: no comparable station-times ===")
        return
    obs = np.array([r[0] for r in rows]); ours = np.array([r[1] for r in rows])
    wet = obs > 0.1
    bias_bits = ["ours %+.2f" % (float((ours[wet] - obs[wet]).mean()) if wet.any() else 0)]
    for i, nm in enumerate(competitors):
        comp = np.array([r[i + 2] for r in rows])
        okw = wet & np.isfinite(comp)
        bias_bits.append("%s %+.2f" % (nm, float((comp[okw] - obs[okw]).mean()) if okw.any() else 0))
    print(f"=== {region} n={len(rows)} wet={int(wet.sum())} | bias@wet " + " ".join(bias_bits) + " ===")
    for thr in THRESHOLDS:
        line = f"  thr {thr}: ours {csi(ours, obs, thr):.3f}"
        for i, nm in enumerate(competitors):
            comp = np.array([r[i + 2] for r in rows])
            ok = np.isfinite(comp)
            if not ok.any():
                line += f" | {nm} n/a"
                continue
            c = csi(comp[ok], obs[ok], thr)
            co = csi(ours[ok], obs[ok], thr)
            lo, hi = boot_diff(ours[ok], comp[ok], obs[ok], thr)
            verdict = "WE WIN" if lo > 0 else ("THEY" if hi < 0 else "tie")
            line += (f" | {nm} {c:.3f} (ours {co:.3f} on n={int(ok.sum())}) "
                     f"d[{lo:+.3f},{hi:+.3f}] {verdict}")
        print(line)

# --- DE ---------------------------------------------------------------------------
def run_de():
    coords = {}
    for ln in open("/tmp/dwd_stations.txt", encoding="latin-1").read().splitlines()[2:]:
        p = ln.split()
        if len(p) >= 6:
            try: coords[p[0].zfill(5)] = (float(p[4]), float(p[5]))
            except ValueError: pass
    t0 = int(dt.datetime(2026, 8, 31, 16, 10, tzinfo=dt.UTC).timestamp())
    t1 = int(TIMES[-1])
    rows = []
    for zp in sorted(glob.glob("/tmp/dwd_gauges/10minutenwerte_nieder_*_now.zip")):
        sid = pathlib.Path(zp).stem.split("_")[2].zfill(5)
        if sid not in coords: continue
        lat, lon = coords[sid]
        try:
            with zipfile.ZipFile(zp) as zf:
                txt = zf.read(zf.namelist()[0]).decode("latin-1")
        except Exception:
            continue
        for row in csv.DictReader(io.StringIO(txt), delimiter=";"):
            try:
                te = int(dt.datetime.strptime(row["MESS_DATUM"].strip(), "%Y%m%d%H%M")
                         .replace(tzinfo=dt.UTC).timestamp())
                v = float(row["RWS_10"])
            except (KeyError, ValueError):
                continue
            if v < 0 or not (t0 <= te <= t1): continue
            g = v * 6.0
            o = ours_window(lat, lon, te, 600)
            if not np.isfinite(o): continue
            rows.append((g, o, radolan_window(lat, lon, te, 600),
                         opera_at(lat, lon, te), lon))
    # Split by provenance: west of 12E our own radars carry the composite; east of
    # 12E today's field is mostly OPERA fill (the eastern DE radars are still on the
    # wet-day verification hold). Lumping them scores two different products.
    west = [r[:4] for r in rows if r[4] < 12.0]
    east = [r[:4] for r in rows if r[4] >= 12.0]
    report("DE-west <12E (own radar core)", west, ("RADOLAN", "OPERA"))
    report("DE-east >=12E (mostly OPERA fill)", east, ("RADOLAN", "OPERA"))

# --- UK ---------------------------------------------------------------------------
def run_uk():
    meta = {}
    for ln in open("/tmp/ea_ids.txt").read().splitlines():
        p = ln.split(",")
        if len(p) == 3:
            meta[p[0]] = (float(p[1]), float(p[2]))
    t1 = int(TIMES[-1])
    rows = []
    for fp in glob.glob("/tmp/ea_gauges/*.csv"):
        sid = pathlib.Path(fp).stem
        if sid not in meta: continue
        lat, lon = meta[sid]
        try:
            rd = list(csv.DictReader(open(fp)))
        except Exception:
            continue
        for row in rd:
            try:
                te = int(dt.datetime.strptime(row["dateTime"], "%Y-%m-%dT%H:%M:%SZ")
                         .replace(tzinfo=dt.UTC).timestamp())
                v = float(row["value"])
            except (KeyError, ValueError, TypeError):
                continue
            if v < 0 or te > t1 or te <= t1 - 3 * 3600: continue
            g = v * 4.0                      # mm/15min -> mm/h
            o = ours_window(lat, lon, te, 900)
            if not np.isfinite(o): continue
            rows.append((g, o, opera_at(lat, lon, te)))
    report("UK (truth: EA 15-min gauges; ours = UKMO composite via fill)", rows, ("OPERA",))

# --- NL guard: the served cube must not regress where we already tie RTCOR ------
def run_nl():
    """Ours vs RTCOR against KNMI 10-min gauges, on the served cube's NL slice.

    This is the deployment guard for serving-side changes (gauge adjustment uses
    KNMI gauges over NL; calibration touches neighbours): the BE/NL standing —
    statistical ties with RTCOR — must survive on the SERVED product too."""
    from tools.gauge_validate import fetch_knmi_10min, read_gauges
    from tools import knmi_rtcor as kr

    t1 = int(TIMES[-1])
    t0 = t1 - 3 * 3600
    rows = []
    NLB, NLS = (3.3, 50.7, 7.3, 53.7), (166, 142)      # ~2-km comparison grid
    _rt = {}
    def rtcor_at(lat, lon, te):
        t = dt.datetime.fromtimestamp(te, dt.UTC)
        t -= dt.timedelta(minutes=t.minute % 5, seconds=t.second)
        vals = []
        for k in (0, 1):                       # the two 5-min fields of the window
            key = (t - dt.timedelta(minutes=5 * k)).strftime("%Y%m%dT%H%M")
            if key not in _rt:
                try:
                    _rt[key] = kr.rate(key, NLB, NLS)
                except Exception:
                    _rt[key] = None
            f = _rt[key]
            if f is None:
                continue
            c = int((lon - NLB[0]) / (NLB[2] - NLB[0]) * NLS[1])
            r = int((NLB[3] - lat) / (NLB[3] - NLB[1]) * NLS[0])
            if not (0 <= r < NLS[0] and 0 <= c < NLS[1]):
                continue
            blk = f[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            blk = blk[np.isfinite(blk)]
            if blk.size:
                vals.append(float(blk.max()))
        return float(np.mean(vals)) if vals else np.nan

    te = t0 - (t0 % 600) + 600
    while te <= t1:
        stamp = dt.datetime.fromtimestamp(te, dt.UTC).strftime("%Y%m%dT%H%M")
        gp = fetch_knmi_10min(stamp)
        if gp is not None:
            for st, la, lo, obs in read_gauges(gp):
                if not (3.3 <= lo <= 7.3 and 50.7 <= la <= 53.7):
                    continue
                o = ours_window(la, lo, te, 600)
                if not np.isfinite(o):
                    continue
                rows.append((obs, o, rtcor_at(la, lo, te)))
        te += 600
    report("NL guard (truth: KNMI 10-min gauges)", rows, ("RTCOR",))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "de"): run_de()
    if which in ("all", "uk"): run_uk()
    if which in ("all", "nl"): run_nl()
