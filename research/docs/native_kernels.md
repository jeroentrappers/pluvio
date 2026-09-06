# Native kernels (research/native, research/native_rs)

Two hot loops are compiled; everything else stays Python. There are two
interchangeable implementations of the same kernels — C++/pybind11
(`research/native`) and Rust/PyO3 (`research/native_rs`) — held to one
contract: identical to the Python reference and to each other, bit for bit. The
Python implementations remain the reference: `research/tests/test_native_kernels.py`
asserts the compiled kernels are **bit-for-bit identical** to them, and
`PLUVIO_NATIVE=0` (or an unbuilt extension) falls back to Python, so a box
without a compiler behaves the same, only slower.

| kernel | replaces | why Python was slow |
|---|---|---|
| `block_flow` | `model.motion.block_flow`'s search loop | (2m+1)² × blocks² tiny numpy reductions — call overhead dominates; blocks are independent, so it also threads |
| `polar_bin` | the `np.add.at` scatter-mean + hole fill in `tools.radar_single_site.polar_to_grid` | `np.add.at` is an unbuffered gather-scatter, the slowest add numpy has |

## Build

C++ (what production runs today):

```
cd research/native && python setup.py build_ext --inplace   # needs pybind11
cp pluvio_native*.so "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

Rust (same kernels, same arithmetic; needs a toolchain via rustup):

```
cd research/native_rs && PYO3_PYTHON=../.venv/bin/python cargo build --release
cp target/release/libpluvio_native_rs.{so,dylib} ./pluvio_native_rs.so   # whichever exists
```

`model.motion` imports `pluvio_native`; point it at the Rust build by setting
`motion._NATIVE` (the benchmark does this) — the modules are drop-in for each
other.

`-ffp-contract=off` keeps the arithmetic identical to numpy's (no FMA
contraction). Threads: one per block, capped at the machine's cores;
`PLUVIO_NATIVE_THREADS` overrides, and the training loader passes
`threads=1` because its parallelism is the DataLoader worker pool.

## Measured (best of 5, `python -m tools.bench_kernels`)

hetz1 (16 cores, gcc 14 / rustc 1.98, best of 21, box under its normal load):

| kernel | python | C++ | Rust | best speed-up |
|---|---|---|---|---|
| flow 192² (training/benchmark grid) | 500.66 ms | 13.05 ms | 13.79 ms | **38×** |
| flow 100² (serving grid) | 38.21 ms | 0.79 ms | 0.75 ms | **51×** |
| flow 256² (producer grid) | 163.26 ms | 7.28 ms | 7.00 ms | **23×** |
| polar_bin 768² (QPE analysis grid) | 13.33 ms | 2.86 ms | 3.53 ms | **4.7×** |

laptop (10 cores, clang 21 / rustc 1.98, best of 21):

| kernel | python | C++ | Rust |
|---|---|---|---|
| flow 192² | 143.78 ms | 6.64 ms | 6.61 ms |
| flow 100² | 10.33 ms | 0.34 ms | 0.36 ms |
| flow 256² | 49.07 ms | 3.45 ms | 3.45 ms |
| polar_bin 768² | 5.14 ms | 1.48 ms | 1.58 ms |

**C++ vs Rust: a tie.** Across the eight measurements the ratio spans
0.81–1.32× with no consistent winner, and repeated runs on the same machine
move it by more than the gap (hetz1's Python baseline itself moved 20 %
between runs under normal load). Both are the same algorithm with the same
float64 accumulation order; the remaining difference is scheduling noise.
Production runs the C++ build because it was first and needs no extra
toolchain on the box; the Rust build is kept as an equal-footing alternative
(`cargo build --release`), and the test suite fails if the two ever diverge.

End to end, on the paths that matter:

| path | before | after |
|---|---|---|
| benchmark advection baseline @192² (`flow_for_pair`) | 157 ms | 7.6 ms |
| `advect_forecast` @192², lead 120 | 148 ms | 8.7 ms |
| training `build_input` @192² with Lagrangian channels, cold flow (threads=1) | 169 ms | 64 ms |
| QPE composite of 3 radars @768², real stamp on hetz1 | 1.94 s | 0.98 s |

What this does **not** change: the served nowcast's latency. That is set by
radar publication (~20–30 min), not compute — see TODO 4.1. The wins land in
the benchmark (2000 samples × 3 baselines), the Lagrangian channel render,
and the QPE archiver's CPU minutes.

## Deployment

* `research/` on hetz1: `.so` in the venv's site-packages; `model.motion`
  picks it up automatically.
* `radarproc/` on hetz1 (the QPE archiver runs from there): the same `.so`
  plus the `polar_to_grid` fast path, wired 2026-09-06.
* The backend container carries its own `morph.py` copy and is unaffected
  (its morph calls are milliseconds and cached per lead pair).
