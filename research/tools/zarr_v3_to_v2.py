"""Re-save a zarr v3 group as zarr v2 so older readers (zarr 2.18 on the GPU box,
Python 3.10) can open it. The seamless builder writes v3; the training node pins
`zarr<3`. Copies each array (chunked along axis 0 to bound memory).

    python tools/zarr_v3_to_v2.py /src.zarr /dst_v2.zarr
"""

from __future__ import annotations

import sys

import numpy as np
import zarr


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    src_path, dst_path = argv
    src = zarr.open_group(src_path, mode="r")
    dst = zarr.open_group(dst_path, mode="w", zarr_format=2)
    for k in src.array_keys():
        a = src[k]
        d = dst.create_array(k, shape=a.shape, chunks=a.chunks, dtype=a.dtype)
        if a.ndim >= 1 and a.shape[0] > 1:
            for i in range(0, a.shape[0], 512):  # chunked copy bounds memory
                d[i:i + 512] = a[i:i + 512]
        else:
            d[:] = a[:]
        print(f"  copied {k} {tuple(a.shape)} {a.dtype}", flush=True)
    print(f"done: {dst_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
