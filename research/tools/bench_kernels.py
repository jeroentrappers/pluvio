"""Before/after benchmark for the native kernels.

Runs each kernel with the Python reference and with every compiled backend
importable on this machine — `pluvio_native` (C++/pybind11, research/native)
and `pluvio_native_rs` (Rust/PyO3, research/native_rs) — on the same inputs,
checks they all agree bit-for-bit, and reports the speed-ups side by side.
Sizes are the ones production uses: 192² and 100² for the flow (training and
serving grids), 768² for the polar binning (the QPE archive's grid).

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


def _backends() -> dict:
    """Every compiled kernel module importable here, by label."""
    out = {}
    for label, mod in (("c++", "pluvio_native"), ("rust", "pluvio_native_rs")):
        try:
            out[label] = __import__(mod)
        except ImportError:
            pass
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    backends = _backends()
    if not backends:
        print("no compiled kernels importable — build research/native (C++) "
              "and/or research/native_rs (Rust)")
        return 1

    rows = []
    for hw, max_shift, label in [((192, 192), 24, "flow 192² (training/benchmark grid)"),
                                 ((100, 100), 7, "flow 100² (serving grid)"),
                                 ((256, 256), 12, "flow 256² (producer grid)")]:
        a, b = _wet_pair(hw)
        kw = {"max_shift": max_shift, "blocks": 4, "subpixel": False}
        motion._NATIVE = None
        py_ms = _time(lambda: motion.block_flow(a, b, **kw), args.repeat)
        ref = motion.block_flow(a, b, **kw)
        row = {"kernel": label, "python_ms": py_ms}
        for name, mod in backends.items():
            motion._NATIVE = mod
            row[f"{name}_ms"] = _time(lambda: motion.block_flow(a, b, **kw), args.repeat)
            got = motion.block_flow(a, b, **kw)
            row[f"{name}_identical"] = all(np.array_equal(x, y)
                                           for x, y in zip(ref, got, strict=True))
        motion._NATIVE = None
        rows.append(row)

    case = _polar_case()
    py_ms = _time(lambda: _polar_python(**case), args.repeat)
    ref = _polar_python(**case)
    call = dict(case)
    for key in ("row", "col", "fill_cells", "fill_bins"):
        call[key] = np.ascontiguousarray(call[key], dtype="int64")
    row = {"kernel": "polar_bin 768² (QPE analysis grid)", "python_ms": py_ms}
    for name, mod in backends.items():
        row[f"{name}_ms"] = _time(lambda: mod.polar_bin(**call), args.repeat)
        got = mod.polar_bin(**call)
        row[f"{name}_identical"] = bool(np.array_equal(np.isnan(got), np.isnan(ref))
                                        and np.array_equal(got[~np.isnan(ref)], ref[~np.isnan(ref)]))
    rows.append(row)

    names = list(backends)
    width = max(len(r["kernel"]) for r in rows)
    header = f"{'kernel'.ljust(width)}  {'python':>10}"
    for n in names:
        header += f"  {n:>9}  {'x':>6}"
    print(header + "  identical")
    for r in rows:
        line = f"{r['kernel'].ljust(width)}  {r['python_ms']:9.2f}ms"
        ok = True
        for n in names:
            line += f"  {r[f'{n}_ms']:8.2f}ms  {r['python_ms'] / r[f'{n}_ms']:5.1f}x"
            ok = ok and r[f"{n}_identical"]
        print(line + f"  {'yes' if ok else 'NO'}")
    if len(names) == 2:
        a, b = names
        print()
        for r in rows:
            ratio = r[f"{a}_ms"] / r[f"{b}_ms"]
            faster = b if ratio > 1 else a
            print(f"{r['kernel'].ljust(width)}  {a} vs {b}: {ratio:.2f}x — {faster} faster")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0 if all(r.get(f"{n}_identical", True) for r in rows for n in names) else 1


if __name__ == "__main__":
    sys.exit(main())
