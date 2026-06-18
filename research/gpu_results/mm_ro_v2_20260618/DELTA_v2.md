# Lightning multimodal vs radar-only — v2 (CORRECTED) — 2026-06-18

The v1 (2026-06-17) result was **confounded**: its `li_flash` channel was 100% NaN
(builder timestamp bug), so "mm" was just radar-only + a dead channel + a different
seed. This v2 is the first *valid* test: real lightning, fixed builder, complete
all-season truth (incl. backfilled OPERA 2026-01→05), regularized training.

## Setup (identical except the lightning channel)
- One zarr `nowcast_mm_full_v2.zarr`: 18,064 issues, **full annual cycle 2025-06→2026-06**,
  256² (~3 km), cadence-30. **mm** = 7ch (radar+history+static+`li_flash`); **ro** =
  6ch (same zarr, `--aux-channels none`).
- Regularized: AdamW wd=1e-4, dihedral augmentation, cosine LR, patience-4, batch-8,
  history-steps 6. Free RTX-2060 (asusprime), ~7.5 h for both + eval.
- **mm** best val **0.0253** (epoch 5, stopped 9). **ro** best val **0.0244** (epoch 14,
  stopped 18). ro reaches a *lower* val loss and keeps improving longer.

## Per-lead delta — RELIABLE leads only
cadence-30 stores frames on a 30-min grid, so only leads 0/30/60/90/120 have full
samples. Leads 40/70/80/100/110 are sparse artifacts (MAE ~0.01, near-zero CSI, too
few matched frames) and are **excluded** — the eval's auto HEADLINE keys on them and
is therefore misleading.

| lead | mm CSI1 | ro CSI1 | ΔCSI1 | mm CSI.1 | ro CSI.1 | ΔCSI.1 | ΔMAE | OF CSI1 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 30 | 0.349 | 0.380 | −0.031 | 0.443 | 0.480 | −0.037 | +0.002 | 0.379 |
| 60 | 0.231 | 0.263 | −0.032 | 0.364 | 0.395 | −0.031 | 0.000 | 0.251 |
| 90 | 0.151 | 0.151 | 0.000 | 0.287 | 0.316 | −0.029 | −0.001 | 0.184 |
| 120 | 0.113 | 0.089 | +0.024 | 0.242 | 0.271 | −0.029 | +0.004 | 0.147 |

Mean (30–120): **ΔCSI@1 ≈ −0.010**, **ΔCSI@0.1 ≈ −0.032**, ΔMAE ≈ 0.

## Verdict
**The lightning channel did not improve nowcast skill — it was neutral-to-slightly
negative.** Radar-only matched or beat multimodal: lower val loss, better CSI@0.1 at
*every* lead, better/equal CSI@1 at 3 of 4 leads (mm only edges ahead at 120 min).
This is consistent in *direction* with the (confounded) v1 but now rests on a real
lightning channel — so it is a sound negative result: a raw lightning-accumulation
channel, added naively, does not help heavy-rain nowcasting on this setup.

## Important caveats
- **Both models LOSE to optical-flow** at all reliable nowcast leads (30–120). This is
  worse than the earlier 3 km radar-only result (which beat pysteps on heavy-rain
  CSI@1 at 30–70 min). The cause is almost certainly **cadence-30's coarse history**
  (3 h @ 30-min steps vs the prior 90-min @ 15-min) — the fine recent motion that
  drives short-lead advection is lost. So v2's *absolute* skill is not comparable to
  the finer v1-era runs; only the mm−ro *delta* is the controlled quantity here.
- **Single seed**, and mm plateaued earlier (epoch 5) than ro (epoch 14) — the gap is
  small enough that seed/optimization variance can't be fully excluded.
- Lightning-only (GII/cloud fast-follow once their backfills finish), 256² grid.

## Recommended follow-up
A **cadence-15 rerun** (13 leads, 90-min/15-min history) would (a) restore absolute
skill comparable to the prior radar-only win over pysteps, and (b) give a cleaner,
finer-lead test of the lightning delta. ~20–30 h on the 2060, or ~5 h / ~$3 on a
rented 4090. Until then: lightning, added alone and naively, is not a win.
