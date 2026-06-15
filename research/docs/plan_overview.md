# Pluvio — full plan, cost, evaluation & maintenance

## 1. Goal
Beat the operational KMI/KNMI precipitation **nowcast** (0–2 h) with a small
learned model trained only on free/open data, especially on convective
intensification where radar-extrapolation collapses. The model is a **residual
correction**: operational nowcast + context channels → corrected rain field.

## 2. End-to-end pipeline (status)
| Phase | What | Status |
|---|---|---|
| A. Data ingress | 8 sources auto-collecting (radar, BE+NL AWS, Netatmo, 6 MSG, 9 ALARO, SST), reboot-safe timers | ✅ live |
| B. Unified store | `build_zarr.py` → 20-channel `timeseries.zarr`; nightly `--append` | ✅ |
| C. Dataset→model | `ZarrCorrectionDataset` → `train.py --zarr` (UNet) | ✅ |
| D. Training | GPU run on Runcrate 4090 via launch kit | ⬜ next |
| E. Evaluation | `evaluate.py --zarr` head-to-head vs operational | ✅ defined (§4) |
| F. Productionize | swap checkpoint into backend `inference_worker` | ⬜ later |
| G. Keep fresh | data auto-grows; periodic re-train + promote gate | §5 |

## 3. Cost
- **Data + storage ≈ €0 ongoing** — all sources free/non-commercial; storage on
  the existing 500 GB disk.
- **GPU (Runcrate RTX-4090 @ $0.66/hr live):** populate ~$0.25; one full run
  (6–10 h) incl. eval ~$4–7; first evaluated model ~$5–8; sweep (5–8 runs)
  ~$25–50; each periodic re-train ~$5; volume a few $/month.
- **20 credits ≈ ~30 GPU-h ≈ 3–4 runs.** Top up for a full sweep.
- Later/optional: CorrDiff diffusion (~$100–300, heavier GPU); dual-pol / ICON-D2
  are collection effort, not GPU. INCA/RADQPE €0 but pending KMI access.

## 4. Evaluation (model/evaluate.py, docs/verification.md)
- **Held-out split is by time** (most-recent `val_frac`=20% held out) — never
  random, to avoid leakage between adjacent frames.
- For each val `(issue, lead)` + sampled cells, record three mm/h values:
  **operational** (KMI/KNMI nowcast = the baseline, input channel 6), **model**
  (our prediction), **observed** (radar+gauge analysis at issue+lead = truth).
- Every metric is computed **identically for operational and model** → true
  head-to-head ("did we beat what KMI already ships").
- **Continuous (per lead):** MAE, RMSE, bias.
- **Categorical skill (per lead, per intensity τ=0.1/1/4 mm/h):** POD, FAR, CSI
  (headline), HSS.
- **Stratified by lead × intensity** — the win we care about is CSI at high τ /
  longer leads (convective regime the baseline misses).
- **Success = model beats operational on RMSE & CSI at most leads, bias ≈ 0,
  no leak.** Absolute CSI must sit in the INCA-class public range (≈0.45–0.55 at
  +30 min, τ=0.1) — far below ⇒ pairing bug, far above ⇒ leakage.
- Runs on the pod post-training → `checkpoints/eval_report.txt` (per-lead table,
  both forecasts). Shape example: `output/summary.md`.
- **Caveat:** 30-min radar cadence ⇒ leads {30,60,90,120}; finer leads need a
  5-min radar re-collect (dataset already supports it).

## 5. Keeping it fresh
- **Data: automatic.** Forward timers + nightly `pluvio-build-zarr --append` keep
  the store current with zero intervention.
- **Model: periodic re-train with a champion/challenger gate:**
  1. Trigger monthly/seasonally (or when a big source lands, e.g. INCA / 5-min radar).
  2. Full `build_zarr.py` rebuild (folds backfilled/late aux into history).
  3. Train the challenger (~$5).
  4. Evaluate champion + challenger on the **same** held-out window.
  5. **Promote only if the challenger wins** (mean CSI / RMSE by a margin) — never
     auto-promote a regression.
  6. Deploy: swap the checkpoint into `inference_worker.run_tick(infer=…)`.
- Cost of freshness ≈ $60–100/yr GPU + a few $/mo storage.
- **Guardrails (recommended ops):** staleness alerting on the forward timers,
  off-site zarr backup, and the eval-gate above (makes unattended refresh safe).

## 6. Critical path now
Rebuild → first GPU run + eval (one `populate` + `train`, ~$5–8) → read
`eval_report.txt` (beat operational?) → iterate / promote. See
`docs/runcrate_launch.md` for the exact launch commands.
