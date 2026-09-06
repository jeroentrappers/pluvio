"""Before/after benchmark for the native kernels (research/native).

Runs each kernel with the Python reference and with the compiled extension on
the same inputs, checks they agree, and reports the speed-up. Sizes are the
ones production actually uses: 192² and 100² for the flow (training grid and
serving grid), 768² for the polar binning (the QPE archive's analysis grid).

    python -m tools.bench_kernels [--repeat 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model import motion  # noqa: E402


def _wet_pair(hw, seed=0, n_cells=8, shift=(3, -2)):
    rng = np.random.default_rng(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros(hw, dtype="float32")
    for _ in range(n_cells):
        cy, cx = rng.uniform(0.1 * h, 0.9 * h), rng.uniform(0.1 * w, 0.9 * w)
        a += (rng.uniform(1, 15) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                          / (2 * rng.uniform(4, 12) ** 2))).astype("float32")
    b = np.roll(np.roll(a, shift[0], 0), shift[1], 1) + rng.normal(0, 0.05, hw).astype("float32")
    return a, np.clip(b, 0, None).astype("float32")


def _polar_case(h=768, wd=768, n_bins=360 * 500, seed=1):
    """A sweep's worth of bins scattered over the analysis grid, with the same
    NaN/height/in-bounds masks and hole-fill mapping shape production has."""
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0, 30, n_bins)
    vals[rng.random(n_bins) < 0.15] = np.nan
    row = rng.integers(0, h, n_bins)
    col = rng.integers(0, wd, n_bins)
    heights = rng.uniform(0, 5000, n_bins)
    inb = rng.random(n_bins) > 0.25
    fill_cells = rng.integers(0, h * wd, h * wd // 6)
    fill_bins = rng.integers(0, n_bins, h * wd // 6)
    return dict(vals=vals, row=row, col=col, heights=heights, inb=inb,
                fill_cells=fill_cells, fill_bins=fill_bins, h=h, wd=wd, max_beam_m=2000.0)


def _polar_python(vals, row, col, heights, inb, fill_cells, fill_bins, h, wd, max_beam_m):
    ok = np.isfinite(vals) & (heights <= max_beam_m) & inb
    acc = np.zeros((h, wd), "f8")
    cnt = np.zeros((h, wd), "i8")
    np.add.at(acc, (row[ok], col[ok]), vals[ok])
    np.add.at(cnt, (row[ok], col[ok]), 1)
    out = np.full((h, wd), np.nan, "f4")
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]
    need = ~hit.ravel()[fill_cells]
    good = np.isfinite(vals[fill_bins]) & (heights[fill_bins] <= max_beam_m)
    sel = need & good
    out.ravel()[fill_cells[sel]] = vals[fill_bins[sel]]
    return out


def _time(fn, repeat: int) -> float:
    """Best-of-`repeat` wall time in ms (best, not mean: it is the least
    noisy estimate of the work itself on a shared box)."""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    nat = motion._native()
    if nat is None:
        print("pluvio_native is not importable — build it: "
              "cd research/native && python setup.py build_ext --inplace")
        return 1

    rows = []
    for hw, max_shift, label in [((192, 192), 24, "flow 192² (training/benchmark grid)"),
                                 ((100, 100), 7, "flow 100² (serving grid)"),
                                 ((256, 256), 12, "flow 256² (producer grid)")]:
        a, b = _wet_pair(hw)
        kw = dict(max_shift=max_shift, blocks=4, subpixel=False)
        motion._NATIVE = None
        py_ms = _time(lambda: motion.block_flow(a, b, **kw), args.repeat)
        ref = motion.block_flow(a, b, **kw)
        motion._NATIVE = nat
        nat_ms = _time(lambda: motion.block_flow(a, b, **kw), args.repeat)
        got = motion.block_flow(a, b, **kw)
        same = all(np.array_equal(x, y) for x, y in zip(ref, got, strict=True))
        rows.append({"kernel": label, "python_ms": py_ms, "native_ms": nat_ms,
                     "speedup": py_ms / nat_ms, "identical": same})

    case = _polar_case()
    py_ms = _time(lambda: _polar_python(**case), args.repeat)
    ref = _polar_python(**case)
    call = dict(case)
    call["row"] = np.ascontiguousarray(call["row"], dtype="int64")
    call["col"] = np.ascontiguousarray(call["col"], dtype="int64")
    call["fill_cells"] = np.ascontiguousarray(call["fill_cells"], dtype="int64")
    call["fill_bins"] = np.ascontiguousarray(call["fill_bins"], dtype="int64")
    nat_ms = _time(lambda: nat.polar_bin(**call), args.repeat)
    got = nat.polar_bin(**call)
    same = bool(np.array_equal(np.isnan(got), np.isnan(ref))
                and np.array_equal(got[~np.isnan(ref)], ref[~np.isnan(ref)]))
    rows.append({"kernel": "polar_bin 768² (QPE analysis grid)", "python_ms": py_ms,
                 "native_ms": nat_ms, "speedup": py_ms / nat_ms, "identical": same})

    width = max(len(r["kernel"]) for r in rows)
    print(f"{'kernel'.ljust(width)}  {'python':>10}  {'native':>10}  {'speed-up':>9}  identical")
    for r in rows:
        print(f"{r['kernel'].ljust(width)}  {r['python_ms']:9.2f}ms  {r['native_ms']:9.2f}ms  "
              f"{r['speedup']:8.1f}x  {'yes' if r['identical'] else 'NO'}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0 if all(r["identical"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
