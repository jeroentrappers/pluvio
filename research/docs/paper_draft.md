# A heavy-rain nowcast edge on open data and hobby compute: an honest 0–2 h benchmark against optical flow over BeNeLux

*Draft — 2026-06-17. Numbers are from measured runs (this repo). The
multimodal-lightning A/B (§4.6) is now in as a *negative result*; remaining
[PENDING] items await richer aux channels and convergence training. Written to be
honest about what is and isn't defensible.*

---

## Abstract

We build a seamless 0–240 h regional precipitation system entirely from free,
open data — EUMETNET OPERA radar (truth), ECMWF AIFS open-data and ERA5 (NWP),
and EUMETSAT MTG (convective observations) — trainable end-to-end on a single
consumer GPU at ≈ US$2 per run. Evaluated honestly at convection-relevant 3 km
resolution against operational-grade optical flow (pysteps), a small
lead-conditioned U-Net does **not** beat advection at placing light rain
(CSI@0.1, FSS) but **consistently beats it on heavy-rain skill (CSI@1 mm/h)
across the 0–2 h range** (≈ +25–35 %). On the multi-day side, a downscaling head
that corrects raw ERA5/AIFS precipitation toward radar improves MAE by 17–43 %
(resolution-dependent), and an isotonic per-lead calibration yields
well-calibrated probabilities with positive Brier skill over climatology
(BSS +0.27 to +0.87). We report two methodological findings that were necessary
to obtain these results without self-deception: (i) the widely used `(1+obs)²`
rain-weighted loss induces severe dry-domain over-prediction (−189 % MAE skill),
which a log1p, capped-weight Huber repairs; and (ii) categorical scores inflate
strongly with grid coarseness, so heavy-rain claims must be made at native truth
resolution and corroborated with scale-aware FSS. A monolithic end-to-end
"seamless" network underperformed modular heads with a classical seam — a
negative result we document. The contribution is not a new state of the art but
an honest, fully reproducible demonstration of what regional precipitation
nowcasting can achieve on open data and hobby compute, plus the loss-design and
evaluation discipline required to measure it fairly.

---

## 1. Introduction

Consumer weather services split cleanly into visualization layers over
third-party NWP (e.g. Windy.com, which adds only a ~1 h optical-flow radar
extrapolation) and vertically integrated shops running their own WRF (e.g.
Windguru, ≥300 CPU cores). Neither path is open to a hobby-tier effort, and
neither publishes how a *small* team should spend its limited budget. Meanwhile
deep-learning nowcasters (DGMR, MetNet-1/2/3) push the radar-beats-NWP horizon
outward but require Google-scale data and accelerators.

We ask a deliberately scoped question: **on fully open data and a single GPU,
where — if anywhere — can a learned model beat the operational nowcasting bar
(optical flow), and where should the limited compute go?** We answer it for a
BeNeLux + upstream-context domain, holding ourselves to three disciplines that
the nowcasting-evaluation literature shows are easy to violate: (1) the baseline
is optical flow, not persistence; (2) categorical skill is reported at the
truth's native resolution with scale-aware FSS; (3) every claim is paired with
the regime where it fails.

**Contributions.**
1. A reproducible open-data + ≈$2 pipeline for seamless 0–240 h regional precip.
2. A defensible heavy-rain nowcast result: a small U-Net beats pysteps on CSI@1
   across 0–2 h, *with* the counter-result that it loses light-rain placement.
3. A transferable loss-design lesson (log1p + capped weight vs squared weight).
4. An evaluation-honesty contribution (persistence floor, CSI resolution
   inflation, mandatory FSS).
5. Cheap, effective local value-adds over raw AIFS: a downscaling head and
   isotonic probability calibration.
6. A documented negative result (monolithic seamless < modular heads).

---

## 2. Data and analysis grid

| Source | Role | Resolution / cadence | Window used |
|---|---|---|---|
| EUMETNET OPERA RATE | truth + radar history | 2 km native composite, 15 min | 2024-08 → 2026-06 (22 mo) |
| ECMWF AIFS open-data | NWP outlook anchor (serving) | 0.25°, 6-hourly to 240 h | forward-only (rolling) |
| ERA5 (ARCO, anonymous GCS) | historical NWP proxy for the downscaler | 0.25°, hourly | 2018 → now (full backfill) |
| EUMETSAT MTG-LI | lightning (convective obs) | ~flash-accumulation | **2024-07 → now [PENDING backfill]** |
| EUMETSAT MTG FCI-L2 | GII instability, cloud | ~10 min | GII 2025-01→, cloud 2025-12→ **[PENDING]** |

All sources are reprojected onto a common KNMI-stereographic analysis grid. The
default 100×100 grid is **~7.7 × 7.1 km/cell** — coarser than a convective cell
(1–5 km) and shown below to inflate categorical scores; we therefore run the
nowcast benchmark at **256×256 ≈ 3.0 × 2.7 km** (`PLUVIO_GRID_N=256`). The grid
resolution is reported alongside every CSI. (Honest bound: 1 km is not pursued —
the OPERA composite truth is 2 km, so finer would be unverifiable.)

---

## 3. Methods

**Model.** A lead-conditioned U-Net (~0.67 M params, base-32) with FiLM
modulation on a (lead-time, hour-of-day, day-of-year) vector, a Softplus
non-negative head, and an optional monotone quantile head (pinball / CRPS-style
loss) for probabilistic output. For the seamless variant, two decoder heads
(nowcast / outlook) share the encoder and are blended by a learned seam; we also
test a classical INCA-style linear-ramp seam.

**Loss (key design point).** Precipitation is ~95–98 % dry. A Huber loss weighted
by `(1+obs)²` in linear space — a natural choice to emphasize heavy rain — makes a
handful of intense pixels dominate the objective by ~10³×, so the optimizer
hedges by over-predicting everywhere. We instead compute the Huber in **log1p
space with a capped linear weight** `clamp(1+obs, ≤5)`, which keeps heavy-rain
emphasis while leaving the dry field cheap to predict.

**Baselines.** (i) **pysteps** real Lucas–Kanade optical-flow extrapolation
(operational-grade; requires OpenCV — silently falls back otherwise, a trap we
hit); (ii) **persistence** (the zero-skill floor); (iii) **raw ERA5/AIFS** precip
(the field the downscaler must beat); (iv) **climatology** (for Brier skill).

**Metrics.** Per 10-min lead: MAE, CRPS (= MAE for a point forecast, from
quantiles otherwise), CSI at τ = 0.1 and 1 mm/h, scale-aware **FSS** at 3/9/15 km
neighbourhoods, and — for probabilities — Brier, Brier Skill Score vs climatology,
and reliability (ECE). Split is by issue-time (most-recent 20 % held out); random
splits leak across adjacent frames.

---

## 4. Results

### 4.1 Nowcast vs real pysteps, 3 km (held-out, n≈4000)

Representative leads (full table in supp.). Bold = best of {model, pysteps}.

| lead | metric | model | pysteps | persistence |
|---|---|---|---|---|
| +20 min | CSI@0.1 | 0.565 | **0.633** | 0.576 |
|         | **CSI@1** | **0.497** | 0.476 | 0.347 |
|         | FSS@9km | 0.786 | **0.870** | 0.823 |
| +60 min | CSI@0.1 | **0.416** | 0.412 | 0.399 |
|         | **CSI@1** | **0.324** | 0.253 | 0.207 |
|         | MAE | 0.056 | 0.055 | 0.062 |
| +120 min| CSI@0.1 | 0.252 | **0.303** | 0.254 |
|         | CSI@1 | 0.156 | **0.163** | 0.118 |

**Finding.** pysteps wins light-rain placement (CSI@0.1, FSS) at almost all leads
— advection excels at moving existing echo. The model wins **heavy-rain CSI@1 at
every lead +20 → +110 min** (e.g. +60: 0.324 vs 0.253, +28 %; +70: 0.294 vs
0.222, +32 %). The model overtakes pysteps on CSI@0.1 only at +60 min. The honest
headline is therefore *narrow and real*: a heavy-rain edge, not a sweep.

### 4.2 NWP downscaling vs raw ERA5

| grid | model MAE | raw ERA5 MAE | skill | model RMSE | ERA5 RMSE |
|---|---|---|---|---|---|
| 100² (~7.7 km) | 0.058 | 0.111 | **+43 %** | — | — |
| 256² (~3 km) | 0.098 | 0.118 | **+17 %** | 0.544 | 0.577 |

The downscaling head corrects ERA5's wet drizzle bias and adds local structure;
the gain shrinks at finer grid (more structure to get right) and with fewer
training epochs.

### 4.3 Probability calibration (isotonic / IDR)

Per-lead isotonic map model-rate → P(obs ≥ 0.1 mm/h), fit on held-out val:

| lead | base rate | Brier (cal) | BSS vs clim | ECE |
|---|---|---|---|---|
| +0  | 0.044 | 0.0052 | **+0.875** | 0.000 |
| +60 | 0.042 | 0.0188 | **+0.528** | 0.000 |
| +120| 0.044 | 0.0307 | **+0.270** | 0.000 |

Calibrated probabilities are skillful over climatology at every lead, decaying
with lead as expected. (ECE is in-sample; §6.)

### 4.4 The loss-design result

| loss | nowcast MAE skill vs persistence |
|---|---|
| `(1+obs)²` weighted Huber (linear) | **−189 %** (severe over-prediction) |
| log1p + capped-weight Huber | beats persistence (+7.2 % @ 7 km) → heavy-rain edge vs pysteps @ 3 km |

### 4.5 Negative result: monolithic seamless

A single end-to-end network trained on hourly issue-times to serve all leads beat
raw ERA5 by +33–42 % on the outlook but **lost to persistence at nowcast** — the
hourly radar history starved the nowcast head. Modular heads (dense 15-min
nowcast + valid-time downscaler) + a classical seam outperform it. We recommend
*against* the monolithic design at this scale.

### 4.6 Multimodal lightning — *experiment in progress (first attempt retracted)*

*The central open question:* does telling the model "convection is firing *now*"
(MTG-LI lightning) widen the heavy-rain edge over both pysteps *and* a radar-only
model?

> **Retraction (2026-06-17).** A first A/B reported here as a "negative result" was
> **confounded and has been withdrawn**: a post-hoc probe of the training store
> showed the lightning channel (`li_flash`) was **100 % NaN** — an indexing bug
> (timestamp-format mismatch between the OPERA and MTG file naming) silently
> dropped every lightning crop, so the "multimodal" model trained on a dead
> channel. The apparent per-lead deltas were training-seed noise. The hypothesis
> was therefore *never tested*. A corrected run (real lightning channel, full
> annual-cycle window, regularised training) is underway; results below will be
> repopulated once it completes. The episode is itself a methods note: **verify
> that each input channel is actually populated before interpreting an ablation.**

**Corrected result (2026-06-18).** Re-run with the fixed channel indexer (real
`li_flash`), the full annual-cycle window (2025-06→2026-06, 18,064 issues, complete
all-season OPERA truth), and regularised training (AdamW + augmentation + cosine +
patience-4). One store, radar-only via channel ablation, so mm-vs-ro differ *only*
by the lightning channel. Reliable nowcast leads (0/30/60/90/120 — the cadence-30
grid; off-grid leads are sparse artifacts) give:

| lead (min) | ΔCSI@1 (mm−ro) | ΔCSI@0.1 | ΔMAE |
|----:|----:|----:|----:|
| 30 | −0.031 | −0.037 | +0.002 |
| 60 | −0.032 | −0.031 | 0.000 |
| 90 | 0.000 | −0.029 | −0.001 |
| 120 | +0.024 | −0.029 | +0.004 |
| **mean** | **−0.010** | **−0.032** | ≈0 |

**Verdict: the lightning channel did not improve nowcast skill** — radar-only had a
lower validation loss (0.0244 vs 0.0253), better CSI@0.1 at every lead, and
better/equal heavy-rain CSI@1 at three of four leads. A raw lightning-accumulation
channel added naively does not help heavy-rain nowcasting. *Caveats:* both models
lose to optical-flow here because the cadence-30 build's coarse 3-h history (vs the
prior 90-min/15-min) depresses absolute short-lead skill — so only the mm−ro delta
is the controlled quantity; single seed; lightning-only. A finer cadence-15 rerun
(restoring the 90-min history) is the natural confirmation.

---

## 5. Discussion — where the compute should go

The medium range is effectively free: AIFS open-data is world-class global NWP, so
the productive moves are (a) **local downscaling/bias-correction** of AIFS (§4.2),
(b) a **classical INCA-ramp blend** for 2–6 h, (c) **cheap calibration** (§4.3),
and (d) spending the GPU only where learning demonstrably helps — heavy-rain
nowcasting (§4.1) and downscaling. Trying to out-forecast AIFS globally, or to
train an end-to-end seamless net, was not productive (§4.5).

## 6. Limitations

- **Undertrained**: nowcast/downscaler ran ~2 epochs (cost-capped); convergence
  runs are needed and likely improve the numbers.
- **Single region / single 22-month split**; no rare-event (flood) stratification.
- **In-sample calibration**: IDR fit and scored on the same val pixels — a
  separate calibration/test split is required for a clean ECE.
- **Downscaler trained on ERA5-as-perfect-forecast**: real AIFS forecast error is
  not yet folded into the outlook head.
- **No DL-nowcaster comparison** (DGMR/MetNet) — out of compute scope, stated.

## 7. Reproducibility

All data are free/open and anonymously accessible (OPERA S3, ARCO-ERA5 GCS, AIFS
open-data, EUMETSAT Data Store). Collection, assembly, training, and evaluation
run on a single Hetzner CPU box (free) plus rented RTX 4090 bursts at ≈$2 each;
total project GPU spend to date < $5. Code, configs, and the eval harness
(`eval_nowcast`, `eval_downscale`, `calibrate_idr`, `metrics`) are in this repo.

## Target venues

AIES, NeurIPS/ICLR climate-AI workshops, or EMS — as a reproducibility +
targeted-finding contribution. The multimodal-lightning A/B (§4.6) is being
re-run after the first attempt was found confounded (dead input channel); its
corrected verdict is pending. A full-paper claim — *the right open observation
closes the operational nowcast's known heavy-rain gap, on open data and hobby
compute* — would need the richer aux stack (GII / cloud-phase / IR), regularised
convergence runs, and a multi-season held-out split before it is defensible.
