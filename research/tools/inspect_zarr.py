"""Quick sanity dump of a zarr store: per-array shape + finite frac + min/max/mean."""
import sys

import numpy as np
import zarr

root = zarr.open_group(sys.argv[1], mode="r")
i = int(sys.argv[2]) if len(sys.argv) > 2 else 100
for k in sorted(root.array_keys()):
    a = root[k]
    if a.ndim == 3:
        f = np.asarray(a[i])
        print(f"{k:14} {tuple(a.shape)}  frame[{i}] finite={np.isfinite(f).mean():.2f} "
              f"min={np.nanmin(f):.4g} max={np.nanmax(f):.4g} mean={np.nanmean(f):.4g}")
    else:
        print(f"{k:14} {tuple(a.shape)}  {np.asarray(a[:]).ravel()[:3]}")
