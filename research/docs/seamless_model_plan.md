# Pluvio seamless precipitation model — design

A single, multimodal, **seamless 0 → 240 h** precipitation system for BeNeLux +
upstream context, trained on free/open data, verified head-to-head against the
operational nowcast (0–2 h) and AIFS/ECMWF (multi-day). Target venues: AIES /
NeurIPS-climate-workshop. Compute is hobby-tier (a single 4090-class GPU), so
the design wins by being **regional + multimodal + observation-anchored**, not
by out-training global NWP.

This doc covers four things: the **data/channels**, the **model architecture**,
the **live inference loop** (how predictions stay current in production), and
the **10-day prediction-window assembly** (how every source combines into one
forecast cube).

---

## 1. The three physical regimes (why "seamless" ≠ one trick)

| Lead | Regime | Where skill comes from | Backbone source |
|---|---|---|---|
| 0–2 h | **Nowcast** | radar extrapolation + convective cues (lightning, instability, cloud-top cooling) | OPERA radar history + MTG |
| 2–24 h | **Blend** | radar skill decays into NWP; obs still anchor the initial state | ALARO + AIFS, obs-anchored |
| 1–10 d | **Outlook** | NWP dynamics; local detail from downscaling + bias-correction | AIFS (det + ensemble) |

A single network can't be best at all three. We train **two heads + a learned
seam**, lead-conditioned, sharing the multimodal encoder.

> **Resolution caveat (honest bound on the convective claim).** The default
> analysis grid is **100×100 over the ~707×773 km radar domain → ~7–8 km/cell**
> (`model.geo.grid_resolution_km()`), *not* 2 km as earlier drafts stated. A
> convective cell is 1–5 km, so at this resolution sub-cell heavy-rain structure
> is **averaged away** and categorical scores at fine thresholds are inflated by
> the coarse grid (a 7-km pixel is far easier to "hit" than a 1-km one). Two
> consequences we hold ourselves to: (1) any convective heavy-rain claim is
> scoped to what ~7 km can represent, or the grid is refined first
> (`PLUVIO_GRID_N=256` → ~3 km, then rebuild + retrain); (2) verification reports
> the grid resolution alongside CSI and uses scale-aware FSS (§5) so the number
> isn't a coarsening artefact.

---

## 2. Channels (all now collecting on hetz1 → Storage Box)

All resampled to the analysis grid (BeNeLux + context). Per-source crops are
already landing; `build_zarr` aligns them into one store.

**Target (truth):** OPERA `RATE` (instantaneous mm/h, nowcast leads) and `ACRR`
(1-h accumulation, longer leads), with the OPERA quality band as a **loss
weight** (down-weight low-confidence radar pixels).

**Inputs**
- *Radar history* — OPERA RATE, last K=6 frames (30 min @ 5-min).
- *Lightning* — MTG-LI Accumulated Flashes (density) [+ flash area].
- *Instability / moisture* — GII: k-index, lifted-index, total precipitable water.
- *Cloud state* — CTTH (top temperature/height/pressure), OCA (optical depth, phase, top height), CT (type/phase), OLR.
- *NWP context* — AIFS det `tp` at matching valid-time (+ ensemble spread for the outlook); legacy ALARO fields.
- *Static* — elevation, land–sea mask, distance-to-coast.
- *Encodings* — lead-time (normalised + sinusoidal), hour-of-day, day-of-year.

Convective-initiation signal (the operational nowcast's blind spot) is now
covered three ways the radar can't see: **lightning** (it's firing), **GII**
(it's about to), **CTTH/OLR cooling** (tops growing).

---

## 3. Architecture

```
            multimodal channel stack (C≈40, ~7-km grid; configurable)
                          │
                ┌─────────▼──────────┐
                │  shared encoder    │  (UNet encoder, ~1–5 M params)
                │  + FiLM(lead, hour)│  lead/time conditioning via FiLM
                └───┬────────────┬───┘
        ┌───────────▼──┐   ┌─────▼───────────┐
        │ NOWCAST head │   │  OUTLOOK head    │
        │ 0–6 h, decode│   │ 6–240 h, decode  │  conditioned on AIFS fields
        │ OPERA RATE   │   │ downscale+correct│  at the target valid-time
        └───────┬──────┘   └────────┬─────────┘
                └──────► learned SEAM ◄────────┘   blend weight = f(lead, agreement)
                          │
                 precip(lead)  +  uncertainty
```

- **Lead-conditioning via FiLM**: one network serves all leads; lead time (and
  hour/season) modulate features — lifts the old `lead/120` hard cap to a
  smooth 0–240 h embedding.
- **Nowcast head** (0–6 h): observation-driven; residual-corrects the radar
  extrapolation using the multimodal cues. Evolution of today's correction UNet.
- **Outlook head** (6 h–10 d): **AIFS-conditioned downscaler** — input is the
  AIFS field at the target valid-time + static + climatology + the latest obs
  state; learns the local correction (orography, coastal, bias). This is the
  CorrDiff idea — stand on AIFS, don't replace it.
- **Seam** (2–6 h): a small learned gate blends the two by lead and by
  nowcast/NWP agreement (disagreement → trust NWP sooner).
- **Uncertainty**: the **quantile outlook head is implemented now** (rec #3),
  not deferred — `SeamlessNet(quantiles=(0.1,0.5,0.9))` emits monotone
  non-crossing per-pixel quantiles trained with a CRPS-consistent pinball loss
  (`train_seamless.quantile_loss`), and the seam collapses the spread to ~0 in
  the sharp radar regime and grows it into the outlook. Calibrated spread is the
  product value ("chance of rain at day 5") and the publishable contribution, so
  it leads rather than trails a deterministic v1. AIFS ensemble spread folds in
  as an additional input channel; a diffusion head remains a later option.

Starts from `model/unet.py` (extend with FiLM + a second decoder head +
multi-channel input). ~1–5 M params → trainable in hours on a 4090.

---

## 4. Training

- **Dataset**: extend `ZarrCorrectionDataset` — new channels, OPERA target,
  lead-conditioning to 240 h, ensemble channels for the outlook head.
- **Sampling**: per (issue_time, lead) pair; oversample rainy/convective cells
  (the 95%-dry distribution otherwise collapses to "predict 0").
- **Loss**: weighted Huber on RATE (weight ∝ (1+obs)² × OPERA-quality);
  accumulation loss on ACRR for long leads; later CRPS for the probabilistic head.
- **Curriculum**: (1) pretrain the shared encoder + nowcast head on the dense
  0–2 h regime (where data is richest); (2) train the outlook head as an AIFS
  downscaler; (3) joint fine-tune with the seam.
- **Split**: by issue-time (most-recent 20 % held out) — never random (adjacent
  frames leak). Matches `evaluate.py`.
- **Compute**: ~€5–10 per run on a 4090; champion/challenger gate (per
  `plan_overview.md §5`) before any promotion.

---

## 5. Verification (extends `model/evaluate.py`)

Head-to-head, identical metrics for model vs baseline, stratified by lead × intensity:
- **0–2 h vs operational nowcast**: CSI/POD/FAR at τ=0.1/1/4 mm/h, RMSE, bias.
  Headline = CSI at high τ / longer leads (the convective win).
- **multi-day vs AIFS/ECMWF**: RMSE + **CRPS** + reliability diagrams (is the
  uncertainty calibrated?). The claim is *local* skill, so verify on the OPERA
  truth grid, not globally.
- Rare-event slice (e.g. 2021 floods, if in window) for tail behaviour.

---

## 6. Live inference loop (production serving)

The model runs as a **scheduled inference worker** (GPU box or hetz1 CPU for the
small net) that keeps the forecast current by **refreshing each regime at its
own cadence**, matching the data:

```
 every ~10 min (fresh radar/lightning/satellite):
   1. assemble LIVE input on the analysis grid for issue_time = now:
        latest OPERA RATE + last 6 frames, latest MTG-LI/GII/CTTH/OCA/CT/OLR,
        latest AIFS run (re-used until the next 6-h run lands)   ← from Storage Box
   2. run NOWCAST head → leads 0…6 h (10-min steps)
   3. SEAM-blend with the standing OUTLOOK at 2–6 h
   4. write leads 0–6 h to the forecast cache

 every 6 h (new AIFS run):
   5. run OUTLOOK head → leads 6 h…240 h (downscale AIFS) → cache
```

- **Output artefact**: one `model_forecast.npz` per issue time — `leads`
  (0…14400 min), `rates` `(n_lead, H, W)` mm/h, per-lead `source` + `confidence`,
  ensemble/quantile fields, `issue_epoch`, `producer`. Generalises the old
  `model_nowcast.npz`; the backend (`backend/.../model.py`) serves **all** bands
  from it and surfaces the provenance via `/v1/forecast`.
- **Product / research decoupling (the producer is swappable)**: the artefact is
  written by `model/produce_forecast.py` with two interchangeable producers —
  `--producer classical` (pysteps optical-flow ⊕ raw-AIFS, `model/classical.py`)
  ships **today** with zero dependence on the research model, and
  `--producer model` (the learned `SeamlessNet`). The learned producer is
  promoted into serving **only** after it beats the classical baseline on the
  champion/challenger gate (§5 / `plan_overview.md §5`) — never before. The
  backend is producer-agnostic; it just serves the cube and its source tags.
- **Serving**: the existing FastAPI backend already serves bands + overlays from
  a cached field. The inference worker writes `model_forecast.npz` to
  `/opt/pluvio/serve`; the backend's `model_band` reads it for **all** bands
  (nowcast → long), replacing the KMI stub for the longer bands. No API change —
  the bands/overlays/`/v1/forecast` machinery already exists.
- **Freshness/cadence** maps onto the backend's existing band cadences
  (`schedules.py`): nowcast 5-min, short hourly, long 12-hourly.
- **Degradation**: if a source is stale/missing, the input assembler zero/NaN-fills
  that channel (model trained with missing-channel dropout) and the band falls
  back to AIFS-direct or the KMI stub — the API never goes dark (same guard as
  the current `model.py`).

---

## 7. The 10-day prediction-window assembly

Each inference cycle produces **one coherent forecast cube** for issue_time `t₀`,
a stack over leads with the source/skill weighting baked in:

| Lead band | Steps | Built from | Method |
|---|---|---|---|
| 0–2 h | 10 min | OPERA radar history + MTG cues | nowcast head (obs-driven, corrected) |
| 2–6 h | 30 min | nowcast ⊕ AIFS | learned seam (blend by lead + agreement) |
| 6–24 h | 1 h | AIFS det + ALARO, obs-anchored | outlook head (downscale + bias-correct) |
| 1–10 d | 3 h | AIFS det + **ensemble** | outlook head + ensemble → calibrated spread |

- The **ensemble** (AIFS-ENS, already collectable) drives the **uncertainty**
  that widens with lead — the honest "chance of rain" the product needs at day 5.
- Every lead is on the **same analysis grid** with the **same units (mm/h)** and
  carries its **source tag** + **confidence** (per `docs/24h_extension.md`'s
  rule: never pretend the day-5 number came from the radar).
- The cube → `model_forecast.npz` → backend cache → `/v1/forecast` (point
  trajectory + chart), `/v1/overlay` (per-lead maps), `/v1/animation` (the loop).
  The PWA's horizon selector (2 h / 12 h / 24 h / 10 d) already renders exactly
  this.

---

## 8. Phased plan (where we are → next)

- **P0 — data foundation** ✅ collecting: OPERA truth + AIFS + MTG-LI + MTG-L2 on hetz1.
- **P1 — assembly**: extend `build_zarr` to fold the new channels + OPERA target
  into the training store; lift the lead cap in `ZarrCorrectionDataset`.
- **P2 — model**: `model/seamless.py` (FiLM lead-conditioning + dual head), CPU
  smoke-tested; nowcast head first (richest data).
- **P3 — train + verify** (GPU): nowcast head vs operational; promote on the gate.
- **P4 — outlook head**: AIFS downscaler vs AIFS-direct; add the seam.
- **P5 — uncertainty**: ensemble → calibrated intervals (diffusion/quantile).
- **P6 — live inference**: the worker of §6 writing `model_forecast.npz`; wire the
  backend's `model_band` to serve all bands from it.

Cost: data ≈ €0; each GPU run ≈ €5–10; full programme a few €100s.
