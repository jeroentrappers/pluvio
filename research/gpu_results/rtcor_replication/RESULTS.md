# Replicating the RTCOR chain on our own sources

Goal: match or beat KNMI RTCOR, the best-in-class product (beats us 15/16, beats OPERA
everywhere). Its chain is published — Overeem et al. 2025 (ESSD 17:4715) plus Overeem et
al. 2020 (JTECH 37:1643) for the clutter classifier — and it is built on wradlib, which
we already run. So this phase replicates their chain step by step on our own volumes.

## The decomposition that set priorities

RTCOR ships the gauge adjustment it applied as a data layer (`image3`,
ADJUSTMENT_FACTOR_[DB]). Backing it out and scoring both against gauges:

| NL halo 0, thr 0.5 | CSI |
|---|---|
| ours (composite v2) | 0.335 |
| **RTCOR unadjusted** | **0.585** |
| RTCOR adjusted | 0.599 |

**The radar chain is the bulk of RTCOR's edge; gauge adjustment is worth ~0.01–0.09.**
Most hours the applied factor is exactly 1.0 (-0.08 dB encoded); it only varies during
rain (-3.8 to +5.7 dB, spatially).

## What was implemented (tools/rtcor_chain.py, gauge_adjust.py)

Fuzzy-logic clutter removal (wradlib's classifier — the paper names it — with KNMI's
weights: texture(ZDR) .20, RhoHV .15, texture(RhoHV) .25, depolarization .20, CPA .20,
threshold 0.6; RhoHV membership 0.80–0.85 per the 2020 paper); K_dp attenuation
(A = 0.081 K_dp, two-way, below freezing level); Appendix-A quality index; multi-sweep
merge; simplified apparent-profile VPR to 800 m; Gabella; Marshall-Palmer; Appendix-B
spatial gauge adjustment (T = 0.25 mm, Gaussians at 30/500 km, v = 0.1). All three feeds
flow through one dispatcher — KNMI archive (dual-pol, to 2019), DWD opendata (17 sites),
OPERA-capture ODIM — and the 8 radars RTCOR itself uses, Wideumont and Helchteren
included, composite from our own sources.

## First measured result: the chain beats our old product everywhere

Five days, 3,917 NL / 1,079 BE gauge-times, chain still in the paper-verbatim weighted
merge (its WORST configuration, see below):

| NL halo 0, CSI | old | **chain** | RTCOR | gap closed |
|---|---|---|---|---|
| thr 0.1 | 0.465 | **0.491** | 0.573 | 24% |
| thr 0.5 | 0.400 | **0.414** | 0.583 | 8% |
| thr 1.0 | 0.348 | **0.405** | 0.591 | 23% |
| thr 2.0 | 0.307 | **0.379** | 0.532 | 32% |

| BE halo 0, CSI | old | **chain** | RTCOR |
|---|---|---|---|
| thr 0.5 | 0.371 | **0.412** | 0.523 |
| thr 2.0 | 0.286 | **0.365** | 0.412 |

Intensity at wet gauges (NL, gauge mean 1.82 mm/h): old 0.94 → chain 1.21 → RTCOR 1.48.
The attenuation and VPR corrections recover much of the under-reading.

⚠️ RTCOR's NL rows carry its in-sample advantage: it is adjusted with the same KNMI
gauges it is scored on here. The BE rows are clean for all products.

## Findings that were not in the paper

1. **Eq. A5 as printed cannot satisfy its own endpoints.** S(h, 4, 1) = 0.05 at h = 4 by
   definition of S, so the subtractive form gives Q_H ≈ 1.0 at 4 km where the text
   demands 0.05. Implementing the typography gave behel Q = 0.95 at 200 km and composite
   POD collapsed to 0.35 while RTCOR sat at 0.91. The product of the two sigmoids
   satisfies every stated property and is what we run.

2. **Averaging across sweeps/radars dilutes shallow rain.** Eq. 1–2 averaging lets
   dry-aloft readings (legitimate measurements!) from higher sweeps and overshooting
   radars drag wet cells below threshold. Measured (nlhrw wet fraction, old = 1.33%):
   weighted 0.63%, argmax-Q_T 0.46% (worse — A5 punishes the lowest 500 m by design, so
   a higher DRY sweep outscores the lowest in shallow drizzle below 1 km), lowest-only
   1.24%. Production mode is **lowest usable sweep wins** — the beam-geometry rule
   already validated on gauges — with the chain's QC intact. How RTCOR reaches POD 0.91
   under averaging is the open question; their full Hazenberg VPR must be doing more
   work than our simplified profile.

3. **"undetect" is a measurement.** All three feeds encode scanned-but-dry; reading it
   as NaN removed every dry cell from evaluation and biased FAR. DWD encodes dry as
   undetect in 98.9% of a dry scan's bins.

4. **Source traps**, each of which would have corrupted the composite: DWD ships RhoHV/
   ZDR only unfiltered (u-prefixed, different path), its unfiltered moments carry noise
   in echo-free bins (classifier must only judge echo), and KNMI's fuzzy variables are
   textures of NaN-bounded fields (filling with zeros manufactures clutter at echo
   edges).

## In flight

Lowest-mode 5-day eval; Aug-30 eight-radar composite (all sources, corrected merge);
cross-network gauge adjustment (adjust with NL, score BE and vice versa — same-network
numbers are in-sample and will be labelled).
