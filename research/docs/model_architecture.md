# A learned post-processor for the Pluvio nowcast

## Goal

Outperform the KNMI / KMI extrapolation nowcast on the metric that matters
to a consumer rain app — **rain timing and intensity at the user's location
over the next 0–120 min**, especially **predicting convective storms at
lead times beyond 20 min**, which the verification showed is the operational
nowcast's main weakness (CSI = 0 at +30 min for ≥ 4 mm/h).

## Approach: residual correction, not from-scratch nowcasting

We *don't* train a full nowcast generator (DGMR / MetNet territory). That
requires GPU-months of compute and millions of paired examples. Instead we
train a small model that learns the *correction* to apply to KNMI's
operational nowcast given a richer feature set than just radar history.

```
                    ┌────────────────────────────────────────────┐
KNMI nowcast frame  │                                            │  corrected
   (lead L)         │            small UNet / ConvLSTM           │  precipitation
   +                │            ~5 M parameters                 │      field
   feature stack ──▶│                                            │──▶ (lead L)
                    └────────────────────────────────────────────┘
```

The model adds three pieces of context the radar extrapolation cannot see:

1. **Surface state** — pressure tendency, humidity, surface convergence — from KMI/KNMI AWS.
2. **Vertical state** — cloud-top temperature, water vapour, instability indices — from Meteosat SEVIRI.
3. **NWP context** — what ALARO thinks the atmosphere is doing on a 1-h scale.

Plus the radar history itself, of course.

## Inputs (channels of a 100×100 input tensor)

All resampled to a common 100×100 grid covering Benelux on the KNMI native
projection. Channel counts in parentheses.

| Feature group | Channels | Source | Notes |
|---|---|---|---|
| Radar history | 6 | KNMI rtcor_5m | Last 30 min of observations, 5-min steps. |
| Operational nowcast | 1 | KNMI radar_forecast | The forecast at the target lead L we're correcting. |
| Meteosat IR 10.8 µm | 2 | EUMETSAT `msg_fes:ir108` | Latest + Δ vs 30 min ago (cooling trend = building convection). |
| Meteosat WV 6.2 µm | 1 | EUMETSAT `msg_fes:wv062` | Latest. |
| Meteosat instability | 2 | EUMETSAT `gii_kindex`, `gii_liftedindex` | Most-recent fields. |
| Meteosat RDT mask | 1 | EUMETSAT `msg_fes:rdt` | Polygon → 0/1 raster: "this pixel is inside a rapidly developing storm right now." |
| AWS surface fields | 5 | KMI + KNMI AWS | Pressure, pressure-tendency-3h, surface temp, humidity, surface wind speed — interpolated to grid by IDW. |
| ALARO context | 4 | KMI ALARO WMS | total_precipitation, total_cloud_cover, 10m wind speed, 2m relative humidity — at lead L rounded to the nearest hour. |
| Static | 3 | one-off | Terrain elevation, land/sea mask, distance-to-coast. |
| Time | 4 | derived | sin/cos of hour-of-day, sin/cos of day-of-year. |

**Total: ~29 input channels × 100×100 grid.**

## Output

A 100×100 grid of corrected precipitation rate in mm/h, for the requested
lead time L ∈ {5, 10, …, 120}. Either:

- **Multi-head model** — single forward pass produces all 24 lead steps as 24 output channels. Compact, share-the-encoder. Risk: hard to balance loss across heads.
- **Lead-conditioned model** — lead L injected as a feature plane, model called 24× per inference. Simpler. Slight redundancy.

Recommendation: start with the lead-conditioned model. Easier to debug; we
can fuse heads later if inference latency becomes a problem.

## Backbone

UNet, very small (~1 M params is enough for 100×100 problems):

```
Input (29×100×100)
    ↓ Conv-BN-ReLU ×2  →  32×100×100
    ↓ Down  (maxpool)     32×50×50
    ↓ Conv-BN-ReLU ×2  →  64×50×50
    ↓ Down                64×25×25
    ↓ Conv-BN-ReLU ×2  →  128×25×25  (bottleneck)
    ↓ Up   (transposed conv) + concat skip → 64×50×50
    ↓ Conv-BN-ReLU ×2
    ↓ Up + concat skip → 32×100×100
    ↓ Conv-BN-ReLU ×2
    → 1×1 conv → 1×100×100  (corrected rate, softplus → positive)
```

Why this size: small enough to train on a single consumer GPU in hours,
large enough to learn spatial bias corrections.

## Loss

Precipitation distribution is extremely imbalanced (~95% of cells dry).
A naive MSE loss learns "always predict 0" and looks great. We need:

- **Weighted Huber loss** with sample weights ∝ `(1 + observed_mm_per_h)²`. Penalises being wrong about heavy rain ~100× more than being wrong about drizzle.
- **Auxiliary BCE head** predicting `rain ≥ 0.1 mm/h` — even when the regression head misses the intensity, the binary "is it raining" head should learn cleanly.
- **Optional spectral loss** (FFT of the prediction vs target) so the model doesn't oversmooth into a blurry blob — important for the "right storm in roughly the right place" property.

## Training data

3 months of paired data should be a defensible starting point. From the
verification work we know KNMI has ~2 years of archive available; we don't
need all of it.

Split by time (not by location):
- **train**: months 1–2 of the window (≈ 18 000 issue times, but we sample 1 in every 6 for diversity → ~3 000 examples × 24 leads ≈ 72 000 grids).
- **val**:   month 3 — used for early stopping and hyperparameter pinning.
- **test**:  a fully held-out month 4 — only touched after model is frozen.

## Compute budget

- Data download: ~3 hours (parallelised, mostly I/O bound).
- Data preprocessing into a single zarr store: ~6 hours one-off.
- Training: ~8 hours on a single RTX 4070 / 4080 class GPU; ~3 days on a CPU.
- Inference: < 50 ms per (location, full-horizon) call once exported to ONNX.

The whole project is feasible on a workstation. No cloud GPU needed.

## Evaluation

Re-run `notebooks/01_verification.py` with the model output substituted for
the operational nowcast. Headline numbers we expect to move:

| Metric (τ = 4 mm/h, +30 min) | Operational | Target |
|---|---|---|
| CSI | 0.00 | ≥ 0.15 |
| POD | 0.00 | ≥ 0.30 |
| FAR | 1.00 | ≤ 0.50 |

If those don't move, the model is broken — drop back to data-pipeline
debugging, not architecture tuning.

## Risks / open questions

- **Domain mismatch.** Training on NL coverage and applying to BE relies on
  the radars + NWP behaving similarly. We should evaluate cross-border
  separately and not assume transfer.
- **EUMETSAT licence.** Non-commercial only. If Pluvio ever monetises (paid
  pro tier, ads), we lose Meteosat and the convective-detection edge goes
  with it. Worth getting clarity from EUMETSAT in writing before depending
  on it strategically.
- **Operational drift.** KNMI / KMI occasionally re-tune their nowcasting
  algorithm. The model needs periodic re-training (quarterly) to stay
  calibrated against whatever's currently in production.

## Roadmap

1. [done] Verification baseline — `notebooks/01_verification.py`.
2. **Collectors** — `fetch_kmi_aws.py`, `fetch_eumetsat_msg.py`, extend `fetch_alaro_features.py`. (See `collectors/` after this commit.)
3. **Pre-processing** — `dataset/build_zarr.py` joins all sources into one zarr store on the common grid.
4. **Model + train loop** — `model/unet.py`, `model/train.py`.
5. **Re-verification** — same metrics, hopefully different numbers.
