# Native kernels (research/native)

Two hot loops are compiled C++ (pybind11); everything else stays Python. The
Python implementations remain the reference: `research/tests/test_native_kernels.py`
asserts the compiled kernels are **bit-for-bit identical** to them, and
`PLUVIO_NATIVE=0` (or an unbuilt extension) falls back to Python, so a box
without a compiler behaves the same, only slower.

| kernel | replaces | why Python was slow |
|---|---|---|
| `block_flow` | `model.motion.block_flow`'s search loop | (2m+1)² × blocks² tiny numpy reductions — call overhead dominates; blocks are independent, so it also threads |
| `polar_bin` | the `np.add.at` scatter-mean + hole fill in `tools.radar_single_site.polar_to_grid` | `np.add.at` is an unbuffered gather-scatter, the slowest add numpy has |

## Build

```
cd research/native && python setup.py build_ext --inplace   # needs pybind11
cp pluvio_native*.so "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

`-ffp-contract=off` keeps the arithmetic identical to numpy's (no FMA
contraction). Threads: one per block, capped at the machine's cores;
`PLUVIO_NATIVE_THREADS` overrides, and the training loader passes
`threads=1` because its parallelism is the DataLoader worker pool.

## Measured (best of 5, `python -m tools.bench_kernels`)

hetz1 (16 cores, gcc 14, 2026-09-06):

| kernel | python | native | speed-up |
|---|---|---|---|
| flow 192² (training/benchmark grid) | 525.75 ms | 14.46 ms | **36.4×** |
| flow 100² (serving grid) | 38.84 ms | 0.96 ms | **40.5×** |
| flow 256² (producer grid) | 163.37 ms | 7.80 ms | **21.0×** |
| polar_bin 768² (QPE analysis grid) | 17.34 ms | 3.60 ms | **4.8×** |

laptop (10 cores, clang 21): 21.9× / 34.1× / 13.6× / 3.3× — same ordering.

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
