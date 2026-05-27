# Pluvio nowcast-correction: CPU experiment findings

**Date:** 2026-05-27
**Question:** Can a learned model, fed open data only, beat KMI/KNMI's
operational precipitation nowcast? And does adding more input sources help?
**Short answer:** Yes on intensity accuracy — the final model beats the
operational nowcast on RMSE at 75 % of lead times — and adding sources helps
monotonically. With important caveats (below) that the GPU run is designed to
resolve.

This documents the CPU-only proof-of-concept phase. No GPU was available, so
this is deliberately a small model on a short data window; the goal was to
learn *direction*, not to ship a production model.

---

## 1. Method

### Data, from one product
All signals are derived from the KNMI `radar_forecast` v2.0 HDF5 files:
- `image1` (lead 0) is the radar+gauge **analysis** — verified byte-identical
  to the separate rtcor observation product (mean abs diff 0.0, corr 1.0). So
  "observation at time T" = `forecast(T).frames[0]`.
- `image2..25` are the operational +5…+120 min nowcast.

This sidesteps the unreliable real-time `rtcor` product (≈11 h retention) and
the `_tar` archive (subscription-gated). The 765×700 stereographic grid is
block-mean downsampled to 100×100 for analysis.

### Auxiliary channels (open data, aligned onto the grid)
- **AWS** (KMI 10-min): pressure, temperature, humidity, wind — inverse-
  distance-weighted onto the grid via per-cell lat/lon from the stereographic
  projection (`model/geo.py`), normalised to ~O(1), aligned to each forecast
  issue time.
- **Meteosat RDT** (EUMETSAT, free): the Rapid Developing Thunderstorms
  product. WMS returns rendered imagery, so we take the alpha channel as a
  0/1 convective-cell mask, max-pool to a proximity field, reproject to grid.

### Model & training
- Residual-correction UNet, 119 k params (base-16), CPU.
- Input: 6 radar-history frames + operational nowcast at the lead + lead/time
  planes (+ aux channels). Output: corrected precipitation field at the lead.
- Loss: Huber weighted by `(1+obs)` + a mean-bias penalty.
- Time-based train/val split (most-recent fraction held out — no leakage
  between adjacent frames).

### Verification
Per-lead MAE/RMSE/bias and categorical CSI/POD/FAR at rain thresholds,
comparing **operational nowcast vs. model vs. observation** on the held-out
validation split (`model/evaluate.py`). Public INCA-class benchmarks in
`docs/verification.md`.

---

## 2. The experiment arc

Four models, each adding signal. Runs 1–3 share a 3-day window (May 9–12,
storm on May 10 in *train*, light val on May 11–12). Run 4 uses a 2-storm
window (May 9–15, ~27 k rainy train samples) with the May 14 storm in *val* —
so its val-RMSE isn't directly comparable to runs 1–3, but its vs-operational
comparison is the fairest.

| Model | Chans | Best val-RMSE | Wet bias (mm/h) | RMSE wins¹ | CSI wins¹ |
|---|---|---|---|---|---|
| radar-only            | 10 | 0.324 | +0.085…0.151 | 0 %  | 50 % |
| + AWS                 | 14 | 0.275 | +0.045…0.072 | 0 %  | 58 % |
| + AWS + RDT           | 15 | 0.262 | +0.13…0.15   | 4 %  | 62 % |
| + bias-fix, 2-storm   | 15 | 0.266² | **+0.06…0.08** | **75 %** | 17 % |

¹ fraction of the 24 lead times where the model beats the operational nowcast
(τ = 1 mm/h). ² different/larger val set; not comparable to rows above.

---

## 3. Findings

### 3.1 More relevant inputs → a better model, every time
val-RMSE fell 0.324 → 0.275 → 0.262 as AWS then RDT were added; CSI win-rate
rose 50 → 58 → 62 %. The multi-source hypothesis is confirmed: the model
extracts usable signal from surface pressure/wind and from the satellite
convective mask, on top of radar. The long-lead CSI (where pure radar
extrapolation decays) improved most with the convective channel.

### 3.2 The bias fix unlocked the RMSE win — at a cost
The squared loss weight made the model hedge precipitation upward (+0.13–0.15
mm/h). Softening it to a linear weight and adding a mean-bias penalty halved
the bias (+0.06–0.08) and flipped RMSE: the model now beats the operational
nowcast on RMSE at **75 % of leads — everything past ~30 min**. But being less
wet cost recall: CSI (rain/no-rain detection) fell below operational at most
leads (it still edges at the longest leads).

This is a precision↔recall operating point, governed by `bias_penalty`:
- low penalty  → over-predicts → high CSI/recall, noisy RMSE, false alarms.
- high penalty → calibrated   → low RMSE (beats operational), misses some rain.

Neither extreme is "correct" for a consumer app; the right point is between
them (or an asymmetric loss). Tuning this is cheap and is the obvious next CPU
experiment.

### 3.3 The convective heavy-rain fix is still unproven
At τ = 4 mm/h the model trails operational on detection. But the operational
nowcast's CSI was *not* zero on this val window (≈0.11 at +30 min) — the
catastrophic CSI = 0 collapse we found in the standalone verification was
specific to May 10's explosive convection *in the training split*. The May 14
val storm was more stratiform/predictable, so there was no collapse to fix.
Demonstrating the convective benefit cleanly needs a val split that contains
an explosive-convection event.

---

## 4. Limitations

Everything here is bounded by the CPU-only setting:
- **Model:** 119 k params (base-16). Production CorrDiff is ~100–1000× larger.
- **Training:** 5–9 epochs, 4 000 samples/epoch, ~25-min budget. Undertrained.
- **Data:** 6.4 days, 2 storms, NL coverage. Production needs ~12 months,
  many regimes, BE+NL.
- **Spatial alignment:** AWS/RDT reprojected with a flat-earth approximation
  over the Benelux — adequate for coarse fields, not for sharp gradients.
- **Channels:** ALARO/AIFS NWP context and the Meteosat IR/WV/instability
  fields are collected but not yet wired in.

None of these are dead ends; each is addressed by the GPU run.

---

## 5. Recommendation

The CPU phase has answered its question: **multi-source inputs + calibration
cross the line on intensity accuracy versus KMI's operational nowcast.** That
is the green light for the real run.

Recommended order:
1. **(cheap, CPU)** sweep `bias_penalty` to find the point where RMSE *and*
   CSI are both competitive — establishes the best honest CPU baseline.
2. **(GPU)** the CorrDiff run per `model/corrdiff/`: 12-month data, full
   channel set (AWS + Meteosat IR/WV/RDT/instability + ALARO + AIFS),
   diffusion model, ~€55–100 of A100 time. Target: beat operational on *both*
   RMSE and CSI, including convective heavy rain.
3. **(product)** wire the winning checkpoint into the backend
   `inference_worker` (drop-in; the cache/API/app contract is unchanged).

---

## 6. Reproducibility

```bash
cd research/ && source .venv/bin/activate

# Data (forecast files = observations + nowcast in one product)
python collectors/fetch_knmi_archive.py --start 2026-05-09 --end 2026-05-15T12:00 --dataset radar_forecast

# Aux channels
python -c "import sys;sys.path.insert(0,'.');from model.dataset import load_store;load_store('data/knmi/radar_forecast/2.0')"  # frame cache
python -m model.build_aux     --data data/knmi --start 2026-05-09 --end 2026-05-15T12:00   # AWS
python -m model.build_aux_msg --data data/knmi --start 2026-05-09 --end 2026-05-15T12:00   # RDT

# Train + evaluate
python -m model.train --data data/knmi --base-channels 16 --max-train-samples 4000 \
    --require-rain-fraction 0.01 --bias-penalty 0.5 --max-minutes 25 \
    --checkpoint checkpoints/pluvio_unet_v2.pt
python -m model.evaluate --data data/knmi --checkpoint checkpoints/pluvio_unet_v2.pt --threshold 1.0
python -m model.evaluate --data data/knmi --checkpoint checkpoints/pluvio_unet_v2.pt --threshold 4.0
```

Checkpoints (gitignored): `pluvio_unet.pt` (radar), `_aux.pt` (+AWS),
`_msg.pt` (+RDT), `_v2.pt` (+bias-fix, best).
