"""Validate the wradlib RADOLAN parse path. Tries the operational YW dir listing,
then falls back to constructing recent 5-min filenames directly (the listing can
come back empty from some hosts)."""
import bz2
import datetime as dt
import io
import re
import urllib.request

import numpy as np
import wradlib as wrl

base = "https://opendata.dwd.de/weather/radar/radolan/yw/"


def candidates():
    out = []
    try:
        listing = urllib.request.urlopen(base, timeout=60).read().decode(errors="ignore")
        out += [base + h for h in re.findall(r'href="([^"]+\.bin\.bz2)"', listing)]
    except Exception as e:
        print("listing failed:", e)
    # fallback: construct recent timestamps (now-20min .. now-120min, 5-min grid)
    now = dt.datetime.now(dt.UTC).replace(second=0, microsecond=0)
    now -= dt.timedelta(minutes=now.minute % 5)
    for k in range(4, 25):
        t = now - dt.timedelta(minutes=5 * k)
        out.append(base + f"raa01-yw_10000-{t:%y%m%d%H%M}-dwd---bin.bz2")
    return out


raw = None
for url in candidates():
    try:
        raw = bz2.decompress(urllib.request.urlopen(url, timeout=60).read())
        print("parsed file:", url.rsplit("/", 1)[-1])
        break
    except Exception:
        continue
if raw is None:
    print("NO FILE FETCHED"); raise SystemExit(1)

data, attrs = wrl.io.read_radolan_composite(io.BytesIO(raw))
# read_radolan_composite returns a MASKED array (nodata/flags masked); fill the
# mask FIRST (np.asarray would drop it → flag values leak in as huge negatives).
arr = np.ma.filled(data, np.nan).astype("float32")
print("shape:", arr.shape, "precision:", attrs.get("precision"), "attrs keys:", sorted(attrs)[:8])
mmh = np.clip(arr * float(attrs.get("precision", 0.01)) * 12.0, 0.0, None)  # mm/h, ≥0
print("mm/h: finite", round(float(np.isfinite(mmh).mean()), 2),
      "max", round(float(np.nanmax(mmh)), 2), "mean", round(float(np.nanmean(mmh)), 4))
rg = wrl.georef.get_radolan_grid(*arr.shape, wgs84=True)
print("grid lon", round(float(rg[..., 0].min()), 1), "..", round(float(rg[..., 0].max()), 1),
      "lat", round(float(rg[..., 1].min()), 1), "..", round(float(rg[..., 1].max()), 1))
