# c17 nowcast — static terrain channels (+ the two open c16 questions)

Design doc / runbook, 2026-08-20. Successor to `c16` (2026-08-06). The headline
change is the one thing every run so far has silently lacked: the **static
terrain channels** (elevation, landmask, distance-to-coast).

## Where c16 left us

`nowcast_mm_c16_full.pt` (9ch = 6 history + `li_flash` + `oflow_rate` +
`rate_tendency`; loss = `precip_loss + 0.3·multiscale_loss`), vs **real pysteps**
over leads 10–120 (12 leads):

| variant | MAE | CSI@0.1 | CSI@1 | FSS@3km | FSS@9km | FSS@15km |
|---|---|---|---|---|---|---|
| **c16_full** | 12/12 | 10/12 | **12/12** | 10/12 | 0/12 | 0/12 |
| c16_nofss | 12/12 | 9/12 | 12/12 | 9/12 | 0/12 | 0/12 |
| c16_noadv | 12/12 | 10/12 | 10/12 | 10/12 | **5/12** | 0/12 |
| mm_c15_0724 (control) | 11/12 | 2/12 | 10/12 | 2/12 | 0/12 | 0/12 |

The promotion gate (`roadmap.md` §"Verification / promotion gate") wants CSI
**and** FSS at the served leads. c16 passes on CSI and at the 3 km grid scale
from 30 min out, and **still fails at 9 km and 15 km at every lead** (−0.02 to
−0.07). Closing that is what c17 is for.

Two leads out of the c16 ablations:

- **The advection prior is what suppresses large-scale FSS.** `noadv` is the only
  variant to win *any* FSS@9km (5/12); both advection arms are 0/12. Advection
  buys heavy-rain CSI (12/12 vs 10/12) and pays in 9 km structure.
- **The FSS-aligned loss barely moved the metric it targets** (full vs nofss:
  +1 lead on CSI@0.1, +1 on FSS@3km, 0→0 at 9/15 km). Likely cause:
  `multiscale_loss` average-pools **rain rate**, but the eval's FSS pools an
  **exceedance indicator** (`pr >= 0.1`, `eval_nowcast.py:114`). Matching pooled
  intensity is not the same objective as matching the pooled wet mask.

## 0. Prerequisites — ALL DONE as of 2026-08-20 (kept for the rationale)

**P1 — build `static.npz` at 256².** It has never been built; no such file exists
in the repo. `model/build_static.py` pulls ETOPO1 elevation from the public
OpenTopoData API, derives `landmask = elevation > 0`, and computes
`distance_to_coast_km` by distance transform. At `PLUVIO_GRID_N=256` that is
65,536 cells / 100 per request ≈ **656 requests at ~1 req/s ≈ 11 min**. Needs
only internet (not the Storage Box), so run it anywhere and relay the file.

```
PLUVIO_GRID_N=256 python -m model.build_static --out research/data/static.npz
```

**P2 — fix the path mismatch (the reason this was silently missing).**
`build_static.py` documents `--out data/static.npz`; `build_seamless_zarr.py:172`
reads `parents[1]/"model"/static.npz`. Settle on `research/data/static.npz`, add
an explicit `--static PATH` flag, and **log a warning when the file is absent**
instead of skipping in silence.

**P3 — add the static arrays to the existing zarr in place.** New
`tools/add_static_channels.py`, mirroring `add_nowcast_channels.py`'s idempotent
guard, so `nowcast_mm_c15_0724_v2.zarr` gains `static_elevation_m`,
`static_landmask`, `static_distance_to_coast_km` without a ~20 h full rebuild.

**P4 — pin the channel layout in the checkpoint.** ⚠️ Adding `static_*` to the
shared zarr **changes `SeamlessDataset.n_channels` for every existing
checkpoint** — `_discover(root, False)` picks static up automatically with
`include_static=True` (the default), so re-evaluating c16_full (9ch) against the
updated store breaks on `load_state_dict`. Fix it properly rather than by
duplicating the zarr: have `train_seamless.py` record `aux_channels`,
`static_channels`, `history_steps` and the `advection` flag in the checkpoint
dict (it currently saves only `in_channels`/`base_channels`/`val_loss`/
`quantiles`, `train_seamless.py:202`), and have `eval_nowcast.py` and
`produce_forecast.py` honour it. This also removes the same failure class from
the hetz1 serving path, which is a prerequisite for promotion anyway.

**P5 — verify before spending GPU.** Probe that each `static_*` array is finite
and **non-constant**, and assert `SeamlessDataset.n_channels` equals the expected
count. The c15 dead-`li_flash` confound is the precedent: a silently NaN or
constant channel invalidates the entire run, and we only caught it after the
fact.

Also check free space on asusprime before starting (`/home` was 194 G at the c15
relay; three more checkpoints and any zarr growth are small, but confirm).

## 0b. What the build smoke test taught us (2026-08-20)

- The window is **much wider than assumed**. OPERA truth spans 2024-08-14 →
  2026-08-20 (68,394 analyses) and MTG-LI has near-complete daily coverage back to
  2024-07 — the June "LI history is short" assumption is stale. c15/c16 trained on
  14 months (39,454 issues); c17 uses **24 months / 68,361 issues (+73%)**.
- `li_flash` reads as **100% NaN on quiet windows**, which looks exactly like the
  c15-v1 dead-channel confound but isn't. The LI crops set `nodata=0.0` while 0 also
  means "no flashes", so a flash-free frame reprojects to all-NaN and
  `nan_to_num` maps it back to 0 — the physically correct value. Verified real
  signal: 16 of 60 frames sampled across Aug 15-19 carry flashes, max 21.2.
  ⚠️ Consequence: the model cannot distinguish missing data from zero flashes.
- Cost at this window: build ~6.3 h (measured 3.0 issues/s), precompute ~65 min,
  ~165 min/epoch → ~31 h per arm.

## 1. Arms

A **chain** of single-variable steps, so each delta is attributable to one change.
~31 h per arm at the 24-month window (c16 measured 95 min/epoch on 39k issues →
~165 min/epoch on 68k, patience-4 stopping around epoch 12–17).

| arm | channels | change vs the arm above it | question |
|---|---|---|---|
| *(base)* c16_full re-eval | 9 | — (scored on the c17 zarr, `--no-static`) | the honest baseline on this validation window |
| **A · static** | 12 | + 3 static channels | **Do terrain channels help?** (the deliverable) |
| **C · static_exceed** | 12 | loss pools wet-mask, not intensity | Does matching the loss to the gate metric move FSS@9/15km? |
| **B · noadv_exceed** | 10 | − advection prior | Does dropping advection lift large-scale FSS further (c16_noadv was the only 5/12 on FSS@9km)? |

**A is the required run** (~31 h); all three ≈ 4 days. Each arm is independent, so
the chain can stop after any of them.

⚠️ **c17 changes two things at once relative to c16**: the static channels *and* a
window that grows from 14 to 24 months. A-vs-c16_full therefore mixes both. That is
why step 0b re-evaluates `nowcast_mm_c16_full.pt` on the **c17** zarr with
`--no-static` — that re-eval, not the published c16 table, is the baseline every
c17 delta should be read against. It costs ~15 min.

For **C**, `--fss-mode exceedance` compares `avg_pool(σ((pred−0.1)/τ))` against the
same soft transform of the target at scales (3, 5, 11) px, τ = 0.05 mm/h. Measured
on a synthetic blob, this is the change that matters: pooling *intensity* charges a
blurred hedge only 0.071× what it charges a 4-px displacement, whereas pooling the
*wet mask* charges 1.013× — i.e. the c16 term was nearly indifferent to the exact
failure it was meant to fix. `--fss-weight 0.3` is inherited from c16 and is still
worth a sweep (0.5, 1.0) on a subset before spending 31 h.

## 2. Run

Same shape as `run_c16.sh` (`--zarr nowcast_mm_c15_0724_v2.zarr`,
`--history-steps 6`, `--leads 0,…,120`, `--batch-size 16 --workers 4 --augment
--weight-decay 1e-4 --cosine --patience 4 --epochs 25 --aux-channels li_flash`),
and per arm `--advection` / `--fss-weight` / `--fss-mode {rate,exceedance}`.
No training flag is needed to *include* static: `SeamlessDataset` discovers
`static_*` automatically (`include_static=True` by default), so the channels
appear as soon as the zarr has them. The opposite direction is what needed a
flag, and it lives on the eval: `eval_nowcast.py --no-static`, for scoring a
pre-static checkpoint against a static-bearing store.

The concrete script is `gpu_results/run_c17.sh`.

Keep `watch_c16.sh` / `notify_c16.sh` running — and given the box suspended
mid-session on 2026-08-20 and has an interrupted `do-release-upgrade` in its
history, **checkpoint every epoch** and confirm the run survives a suspend/resume
before trusting a 60 h sequence.

## 3. Eval + write-up

`eval_nowcast.py --samples 4000` per arm, then the same per-lead table as
`gpu_results/mm_ro_c15_20260619/DELTA_c15.md`. **Report FSS at all three scales
in the write-up** — omitting it is what made c15 and c16 read as bigger wins than
the gate allows. Score each arm as wins/12 vs pysteps on MAE, CSI@0.1, CSI@1 and
FSS@{3,9,15}km, and state explicitly whether the gate is passed.

## 4. Gate decision

- **If any arm wins FSS@9km and @15km at the served leads** (30–120) while
  holding the CSI wins → the gate is passed, and promotion to `--producer model`
  on hetz1 is justified on its own terms.
- **If not** → the honest options are to serve pysteps at 0–30 min and the model
  at 30–120 (per-lead best-of, which the numbers already support), or to keep
  classical serving and run the model as a shadow producer. Either way the FSS
  deficit gets stated in `paper_draft.md`, not omitted.
