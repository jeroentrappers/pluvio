# Pluvio roadmap — best-now vs data-gated-later

Decision doc (2026-06-16). What we can do **right now** on open data + ~zero
compute, and the staged plan to grow into a genuinely better model **as forward
data accumulates**. Grounds out the strategy in `seamless_model_plan.md` and
`nowcast_deliverable.md` against what we've actually measured.

## Honest current state (what we've verified)

- **0–2 h nowcast (radar-only) already beats optical-flow** where it matters:
  `eval_nowcast` (4000 val samples, ~7.7 km grid) — model > optical-flow on
  CSI@0.1 at **+40→+120 min**, > on **heavy rain (CSI@1) at nearly every lead**
  (e.g. +60: .37 vs .25), MAE ≤ at most leads. Optical-flow only holds +10–30 min
  (advection's home turf). *Caveat:* baseline was the phase-corr **fallback**
  (no pysteps py3.13 wheel) — re-check vs real pysteps before publishing.
- **NWP downscaling works:** ERA5→OPERA downscaler beats raw NWP by **+33–43%**
  (`pretrain_downscale.pt`). This is the outlook value-add — local correction of
  AIFS, *not* replacing it.
- **The monolithic from-scratch unified model is the wrong target:** Stage B
  (hourly history) lost to persistence at nowcast — coarse history starved the
  nowcast head. Keep the heads **modular**, not one end-to-end net.
- **Producer already shipped** (`produce_forecast.py --producer classical`):
  pysteps-or-fallback optical-flow + cosine-taper 2–6 h blend (≈ INCA ramp) +
  lead-widening confidence + source tags; `--producer model` is the gated upgrade.

## Resolution target

- **Truth-bound ceiling = 2 km** (OPERA composite native). Don't train/verify
  finer than the truth.
- **Now → 3 km** (`PLUVIO_GRID_N=256`, 256² over the domain): cheap, no data wait,
  recovers convective structure the 7 km grid averages away.
- **1 km: not on open data.** Needs a national-radar 1 km mosaic truth (real
  data-eng) AND wouldn't add forecast skill (AIFS is 28 km). Defer.
- Resolution sharpens nowcast + downscaling detail; it does **not** add
  medium-range predictability.

---

## NOW — no-wait wins (data is already deep enough), in priority order

Everything here uses data we already have (22 mo OPERA, 102 mo ERA5) or no new
data, and costs ~€0 (hetz1 CPU) + at most a ~$1 GPU run.

1. **Resolution → 3 km (256²).** Set `PLUVIO_GRID_N=256`; rebuild `static.npz` +
   the OPERA-history & downscaling zarrs; retrain the nowcast head + downscaler.
   *Value:* verifiable convective/heavy-rain structure. *Cost:* a few hours of
   free hetz1 builds + ~$1 GPU. *Gate:* report grid km alongside every CSI; FSS
   must agree.
2. **Real pysteps baseline.** Build a py3.11 image with a pysteps wheel; re-run
   `eval_nowcast` vs real pysteps (the honest bar) and use pysteps (not the
   fallback) in the classical producer. *Value:* honest validation + a stronger
   shipped nowcast. *Cost:* ~1 h image work.
3. **IDR probability calibration.** Fit Isotonic Distributional Regression of
   forecast → OPERA exceedance P(rain>τ) per lead on the held-out history;
   replace the heuristic confidence anchors with calibrated probabilities; add a
   reliability-diagram check. *Value:* trustworthy "chance of rain" — the
   product's headline number. *Cost:* a few lines of sklearn + an eval. *No GPU.*
4. **Wire the downscaler into the outlook band.** Feed the live AIFS cube through
   `pretrain_downscale.pt` (instead of raw AIFS) for the 6 h–240 h leads in the
   producer. *Value:* the +43 % local correction lands in the actual product.
5. **Promote the learned nowcast head** (0–2 h) as a gated upgrade once (2)
   confirms it beats real pysteps; classical stays the fallback.

Recommended first execution: **#1 (3 km) + #3 (IDR)** give the biggest visible +
trust improvement for the least effort; #2 unblocks honest promotion.

---

## LATER — data-gated (waiting on forward accumulation)

Trigger = enough forward-collected depth across enough weather regimes. AIFS/MTG
started ~2026-06-15, so these unlock over the coming weeks–months.

| Improvement | What | Trigger (rough) | Expected value |
|---|---|---|---|
| **Multimodal nowcast** | Add MTG lightning/GII/CTTH/OCA/CT/OLR channels to the nowcast head (the convective-initiation signal radar can't see) | ≥6–8 weeks MTG across a convective season | Widen the heavy-rain nowcast win — the **publishable** contribution |
| **AIFS forecast-error fine-tune** (Stage B, done right) | Swap ERA5-anchor → live **AIFS forecast-at-lead**; fine-tune the outlook head on real forecast error; dense **15-min** radar history for the nowcast head | ≥8–12 weeks AIFS det | Real multi-day local skill vs raw AIFS |
| **Probabilistic outlook** | AIFS-ENS (aifs-ens) → calibrated spread; pair the quantile head with IDR | ensemble collection running + weeks depth | Honest day-5 "chance of rain" intervals |
| **2 km resolution** | Densify once 3 km is validated and worth the compute | after #1 NOW proven | Sharper convective verification |
| **1 km** | Only if a national-radar 1 km mosaic truth is built | separate data-eng project | Sharper 0–2 h nowcast only |

---

## Verification / promotion gate (enforced, unchanged)

- **Nowcast:** beat **real pysteps** on CSI **and** FSS at the served leads.
- **Outlook:** beat **raw AIFS** on the OPERA truth grid (RMSE + CRPS +
  reliability), never globally.
- A learned head is promoted into serving **only** on passing its gate; the
  classical producer stays the always-on fallback. Always print grid resolution
  with CSI; FSS must agree before any heavy-rain claim.

## One-line strategy

Borrow the expensive part (medium-range skill) from **AIFS for free**; spend our
cheap effort on the parts we've shown we can win — a **multimodal 0–2 h nowcast**
and **local AIFS downscaling + calibration** — at **3 km now, 2 km eventually**,
never 1 km until a 1 km truth exists.
