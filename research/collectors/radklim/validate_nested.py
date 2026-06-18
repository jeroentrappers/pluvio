"""Validate the full RADKLIM nested path on the real 2024-08 monthly tar:
monthly .tar -> daily .tar.gz -> 5-min .bin; confirm TS_RE matches inner names,
parse one composite, and check it regrids onto a BeNeLux target."""
import io
import re
import tarfile
import urllib.request

import numpy as np
import wradlib as wrl
from scipy.spatial import cKDTree

URL = ("https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/"
       "radolan/reproc/2017_002/bin/2024/YW2017.002_202408.tar")
TS_RE = re.compile(r"(\d{10})")

print("downloading monthly tar (~140 MB)...")
raw = urllib.request.urlopen(URL, timeout=600).read()
print("monthly tar bytes:", len(raw))
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as mtf:
    days = [m for m in mtf.getmembers() if m.isfile() and m.name.endswith((".tar.gz", ".tgz", ".tar"))]
    print("daily members:", len(days), "e.g.", days[0].name if days else None)
    dtf = tarfile.open(fileobj=io.BytesIO(mtf.extractfile(days[0]).read()), mode="r:gz")
    bins = [m for m in dtf.getmembers() if m.isfile()]
    print("bins in first day:", len(bins), "e.g.", bins[0].name)
    print("TS_RE matches first bin:", TS_RE.search(bins[0].name).group(1) if TS_RE.search(bins[0].name) else "NO MATCH")
    data, attrs = wrl.io.read_radolan_composite(io.BytesIO(dtf.extractfile(bins[0]).read()))
    arr = np.clip(np.ma.filled(data, np.nan).astype("float32") * float(attrs.get("precision", 0.01)) * 12.0, 0.0, None)
    print("parsed mm/h: shape", arr.shape, "finite", round(float(np.isfinite(arr).mean()), 2),
          "max", round(float(np.nanmax(arr)), 2))
    # NN map to a BeNeLux target grid (~0.02°) — sanity-check overlap
    rg = wrl.georef.get_radolan_grid(*arr.shape, wgs84=True)
    lons = np.arange(-2, 12.0001, 0.02); lats = np.arange(56, 46.9999, -0.02)
    TLON, TLAT = np.meshgrid(lons, lats)
    dist, idx = cKDTree(np.column_stack([rg[..., 0].ravel(), rg[..., 1].ravel()])).query(
        np.column_stack([TLON.ravel(), TLAT.ravel()]))
    cover = (dist <= 0.05).mean()
    print(f"target grid {TLON.shape}: fraction within RADOLAN coverage = {cover:.2f}")
