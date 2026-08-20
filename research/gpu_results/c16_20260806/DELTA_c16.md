# c16 nowcast — advection prior + tendency + FSS-aligned loss — 2026-08-06

Stacked improvements over `c15_0724`, nowcast leads 0–120 only. Three arms on the
free RTX-2060 (asusprime), ~95 min/epoch, 3.5 days total:

- **full**  = advection + tendency + FSS loss (the deliverable) — `nowcast_mm_c16_full.pt`
- **noadv** = tendency + FSS loss, no advection prior (ablation)
- **nofss** = advection + tendency, per-pixel loss only (ablation)

Zarr `nowcast_mm_c15_0724_v2.zarr`, `--aux-channels li_flash`, `--history-steps 6`,
batch-16, AdamW wd 1e-4, dihedral augment, cosine, patience-4. Baseline/control is
the existing `c15_0724` MM run (not retrained). Bar is **real pysteps** LK
optical-flow (py3.10 wheel, `pystepsrc` loaded), not the phase-correlation fallback.

## Wins vs pysteps across leads 10–120 (12 leads)
| variant | MAE | CSI@0.1 | CSI@1 | FSS@3km | FSS@9km | FSS@15km |
|---|---|---|---|---|---|---|
| c16_full | 12/12 | 10/12 | 12/12 | 10/12 | 0/12 | 0/12 |
| c16_nofss | 12/12 | 9/12 | 12/12 | 9/12 | 0/12 | 0/12 |
| c16_noadv | 12/12 | 10/12 | 10/12 | 10/12 | 5/12 | 0/12 |
| c15_0724 (control) | 11/12 | 2/12 | 10/12 | 2/12 | 0/12 | 0/12 |

c16 is a large step up from c15: **heavy-rain CSI@1 now beats pysteps at every
lead**, and CSI@0.1 / FSS@3km win at 30–120 (losing only 10 and 20 min, which is
advection's home turf). Training also converged properly — best epoch 8–10 of 12,
versus c15's epoch 2 — so the earlier under-training is resolved.

## Per-lead, c16_full vs pysteps
| lead | CSI@1 | OF | ΔCSI1 | CSI@0.1 | OF | ΔCSI.1 | FSS3km | OF | Δ | FSS9km | OF | Δ | FSS15km | OF | Δ |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 0.555 | 0.528 | +0.027 | 0.663 | 0.666 | -0.003 | 0.798 | 0.800 | -0.002 | 0.885 | 0.903 | -0.018 | 0.915 | 0.937 | -0.022 |
| 20 | 0.529 | 0.504 | +0.025 | 0.635 | 0.649 | -0.014 | 0.777 | 0.787 | -0.010 | 0.866 | 0.896 | -0.030 | 0.897 | 0.932 | -0.035 |
| 30 | 0.430 | 0.372 | +0.058 | 0.536 | 0.524 | +0.012 | 0.698 | 0.687 | +0.011 | 0.781 | 0.808 | -0.027 | 0.819 | 0.860 | -0.041 |
| 40 | 0.355 | 0.291 | +0.064 | 0.479 | 0.452 | +0.027 | 0.647 | 0.622 | +0.025 | 0.722 | 0.739 | -0.017 | 0.759 | 0.797 | -0.038 |
| 50 | 0.370 | 0.304 | +0.066 | 0.477 | 0.464 | +0.013 | 0.646 | 0.633 | +0.013 | 0.712 | 0.744 | -0.032 | 0.746 | 0.799 | -0.053 |
| 60 | 0.295 | 0.233 | +0.062 | 0.404 | 0.386 | +0.018 | 0.576 | 0.557 | +0.019 | 0.640 | 0.672 | -0.032 | 0.673 | 0.734 | -0.061 |
| 70 | 0.283 | 0.218 | +0.065 | 0.381 | 0.364 | +0.017 | 0.552 | 0.533 | +0.019 | 0.607 | 0.639 | -0.032 | 0.636 | 0.698 | -0.062 |
| 80 | 0.262 | 0.206 | +0.056 | 0.363 | 0.353 | +0.010 | 0.532 | 0.522 | +0.010 | 0.584 | 0.625 | -0.041 | 0.611 | 0.683 | -0.072 |
| 90 | 0.235 | 0.178 | +0.057 | 0.359 | 0.341 | +0.018 | 0.528 | 0.509 | +0.019 | 0.578 | 0.608 | -0.030 | 0.604 | 0.666 | -0.062 |
| 100 | 0.201 | 0.146 | +0.055 | 0.311 | 0.289 | +0.022 | 0.474 | 0.448 | +0.026 | 0.519 | 0.542 | -0.023 | 0.543 | 0.599 | -0.056 |
| 110 | 0.215 | 0.161 | +0.054 | 0.316 | 0.299 | +0.017 | 0.480 | 0.460 | +0.020 | 0.525 | 0.554 | -0.029 | 0.548 | 0.610 | -0.062 |
| 120 | 0.183 | 0.135 | +0.048 | 0.308 | 0.272 | +0.036 | 0.470 | 0.428 | +0.042 | 0.514 | 0.515 | -0.001 | 0.536 | 0.568 | -0.032 |
| **mean Δ** | | | **+0.053** | | | **+0.014** | | | **+0.016** | | | **-0.026** | | | **-0.050** |

## Verdict — better, but the gate is still not passed
The promotion gate (`roadmap.md` §"Verification / promotion gate") requires beating
real pysteps on CSI **and** FSS at the served leads.

- **CSI: passed** from 30 min out (and CSI@1 at every lead).
- **FSS@3km: passed** from 30 min out.
- **FSS@9km and @15km: failed at all 12 leads** (−0.02 to −0.07, worst around
  80–110 min).

So the model is better at *intensity* and still worse at *extent*: it wins per-pixel
heavy-rain detection while placing the light-rain envelope less well than advection
at 9–15 km neighbourhoods. Reporting CSI alone would overstate this result — that
omission is what made c15 read as a bigger win than it was.

## The two findings that matter for c17

**1. The advection prior is what suppresses large-scale FSS.** `noadv` is the only
arm to win *any* FSS@9km (5/12); both advection arms are 0/12. Advection buys
heavy-rain CSI (12/12 vs 10/12) and pays for it in 9 km structure. That trade-off is
exploitable and unexplored: nobody has yet run `noadv` *with* a stronger structure
term.

**2. The FSS-aligned loss barely moved the metric it targets.** full vs nofss is
+1 lead on CSI@0.1, +1 on FSS@3km, and 0→0 at 9/15 km. The likely cause is a
mismatch: `multiscale_loss` average-pools **log1p rain rate**, but the eval's FSS
pools an **exceedance indicator** (`pr >= 0.1`, `eval_nowcast.py:114`). Matching
pooled intensity is not the same objective as matching the pooled wet mask — a field
can agree on mean rate over a 9 km box while misplacing the wet/dry boundary FSS
actually scores. Pooling a soft exceedance (`sigma((pred-0.1)/tau)`) targets the gate
metric directly. The `--fss-weight 0.3` may also simply be too small.

## Caveats
Single seed per arm. Still **no static terrain channels** — `static.npz` has never
been built and `build_seamless_zarr.py:172` reads it from a different path than
`build_static.py` writes, so every run to date silently had none (fixed in the c17
runbook). Lightning-only aux (GII/cloud remain a fast-follow). 256 grid (~3 km),
2060-scale training. `--advection` is not recorded in the checkpoint dict, so the
channel layout is not reproducible from the artifact alone.

Next: `docs/c17_static_experiment_runbook.md`.
