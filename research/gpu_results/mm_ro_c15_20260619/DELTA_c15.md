# Lightning multimodal vs radar-only — cadence-15 (the fidelity rerun) — 2026-06-19

The definitive lightning A/B. v1 (2026-06-17) was confounded (dead `li_flash`
channel). v2 cadence-30 (2026-06-18) had real lightning but **coarse 3-h history →
only 5 leads**, which depressed absolute skill (both models lost to optical-flow)
and showed a slightly *negative* lightning delta. This run restores fidelity:
**cadence-15 (90-min/15-min history, all 13 leads)**, full annual cycle, regularised.

## Setup (identical except the lightning channel)
- `nowcast_mm_c15_v2.zarr`: 36,113 issues, 2025-06→2026-06, 256² (~3 km), cadence-15.
  **mm** = 7ch (+`li_flash`); **ro** = 6ch (same zarr, `--aux-channels none`).
- Regularised: AdamW wd1e-4 + dihedral augment + cosine + patience-4, batch-16.
  Free RTX-2060 (asusprime), ~20 h both + eval (~80 min/epoch).
- **mm** best val **0.0243** (ep2). **ro** best val **0.0245** (ep2). mm marginally lower.

## Per-lead delta (model rows; bar = optical-flow)
| lead | mm CSI1 | ro CSI1 | ΔCSI1 | mm CSI.1 | ro CSI.1 | ΔCSI.1 | OF CSI1 |
|----:|----:|----:|----:|----:|----:|----:|----:|
| 10 | 0.497 | 0.495 | +0.002 | 0.622 | 0.593 | +0.029 | 0.501 |
| 20 | 0.517 | 0.513 | +0.004 | 0.611 | 0.585 | +0.026 | 0.527 |
| 30 | 0.434 | 0.434 | +0.000 | 0.530 | 0.511 | +0.019 | 0.392 |
| 40 | 0.355 | 0.353 | +0.002 | 0.477 | 0.455 | +0.022 | 0.298 |
| 50 | 0.348 | 0.355 | −0.007 | 0.452 | 0.424 | +0.028 | 0.309 |
| 60 | 0.320 | 0.315 | +0.005 | 0.433 | 0.395 | +0.038 | 0.259 |
| 70 | 0.255 | 0.255 | +0.000 | 0.373 | 0.332 | +0.041 | 0.213 |
| 80 | 0.261 | 0.257 | +0.004 | 0.373 | 0.318 | +0.055 | 0.220 |
| 90 | 0.223 | 0.211 | +0.012 | 0.341 | 0.285 | +0.056 | 0.198 |
| 100 | 0.187 | 0.170 | +0.017 | 0.320 | 0.266 | +0.054 | 0.160 |
| 110 | 0.199 | 0.180 | +0.019 | 0.322 | 0.262 | +0.060 | 0.186 |
| 120 | 0.138 | 0.123 | +0.015 | 0.265 | 0.209 | +0.056 | 0.145 |
| **mean** | | | **+0.006** | | | **+0.040** | |

## Verdict — lightning HELPS (modestly, consistently)
- **ΔCSI@0.1 is positive at every lead** (mean +0.040), growing from +0.02 (short) to
  +0.06 (long). Robust, not noise.
- **ΔCSI@1 (heavy rain) is positive at most leads** (mean +0.006), growing to +0.015–
  0.019 at 90–120 min. Small but lead-correlated.
- **The benefit grows with lead time** — physically sensible: lightning flags active
  convection that radar-only advection loses track of further out, so the model
  keeps placing/sustaining convective rain better at longer horizons.

This reverses the earlier negatives, which were artifacts: v1's dead channel, and
v2-cadence-30's coarse history (which crippled both models and the comparison).

## Also: strong absolute skill restored
Both models **beat optical-flow on heavy-rain CSI@1 at 30–110 min** (mm) / 30–100 (ro)
— e.g. +60 min mm 0.320 vs OF 0.259 — with MAE ≤ optical-flow at every lead. The
cadence-30 absolute-skill regression is gone; the 3 km / 90-min-history model is
genuinely good, and lightning extends the heavy-rain win one lead further (to 110).

## Caveats
Single seed (mm/ro val within 0.0002); lightning-only (GII/cloud are a fast-follow
once those backfills finish); 256² grid; 2060-scale training (patience-stopped ~ep6).
The CSI@0.1 consistency across all 13 leads is the strongest evidence; CSI@1 is
directionally positive but small. A multi-seed run would tighten the heavy-rain claim.
