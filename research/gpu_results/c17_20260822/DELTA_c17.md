# c17 — the exceedance-pooled loss passes the promotion gate — 2026-08-25

**`nowcast_mm_c17_c_exceed.pt` beats real pysteps on every metric at every served
lead (10–120 min).** The promotion gate in `roadmap.md` — CSI **and** FSS at the
served leads — is met for the first time. Three model generations had failed it on
FSS@9km/@15km; the fix was one flag, and it was a measurement error in the
objective, not a shortage of data or capacity.

## Arms

All on `nowcast_mm_c17_v2.zarr` (68,361 issues, 2024-08-14 → 2026-08-19, cadence-15,
13 leads, 256² ≈ 3 km). Shared: `--history-steps 6 --aux-channels li_flash
--batch-size 16 --augment --weight-decay 1e-4 --cosine --patience 4 --epochs 25`.

| arm | ch | differs by | epochs | best val |
|---|---|---|---|---|
| baseline | 9 | c16_full re-scored here, `--no-static` | — | — |
| A | 12 | + 3 static | 15 | 0.0196 |
| **C** | 12 | + `--fss-mode exceedance` | 8 | 0.0206 |
| B | 10 | C − advection prior | 19 | 0.0216 |

⚠️ Val losses are **not comparable across arms** — A uses the rate-pooled term,
C and B the exceedance-pooled one, which is a different objective on a different
scale. Only the eval metrics compare.

## Wins vs real pysteps (12 leads, 10–120 min)
| variant | MAE | CSI@0.1 | CSI@1 | FSS@3km | FSS@9km | FSS@15km |
|---|---|---|---|---|---|---|
| c16_full baseline (9ch, re-eval) | 12/12 | 10/12 | 12/12 | 10/12 | 0/12 | 0/12 |
| c17-A · static (12ch) | 12/12 | 10/12 | 12/12 | 10/12 | 0/12 | 0/12 |
| **c17-C · static + exceedance (12ch)** | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| c17-B · noadv + exceedance (10ch) | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 12/12 |

## Mean FSS margin vs pysteps (negative = losing)
| variant | FSS@3km | FSS@9km | FSS@15km |
|---|--:|--:|--:|
| c16_full baseline (9ch, re-eval) | +0.0154 | -0.0253 | -0.0484 |
| c17-A · static (12ch) | +0.0132 | -0.0274 | -0.0504 |
| **c17-C · static + exceedance (12ch)** | +0.0491 | +0.0466 | +0.0338 |
| c17-B · noadv + exceedance (10ch) | +0.0389 | +0.0376 | +0.0259 |

The deficit that blocked promotion since c15 — −0.025 at 9 km, −0.048 at 15 km —
inverts to **+0.047 / +0.034**. A swing of +0.072 / +0.082.

## c17-C per lead
| lead | CSI@1 | OF | Δ | CSI@0.1 | OF | Δ | FSS@9km | OF | Δ | FSS@15km | OF | Δ |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 0.540 | 0.509 | **+0.031** | 0.668 | 0.649 | **+0.019** | 0.900 | 0.894 | **+0.006** | 0.932 | 0.930 | **+0.002** |
| 20 | 0.550 | 0.524 | **+0.026** | 0.676 | 0.657 | **+0.019** | 0.902 | 0.898 | **+0.004** | 0.934 | 0.933 | **+0.001** |
| 30 | 0.446 | 0.387 | **+0.059** | 0.591 | 0.546 | **+0.045** | 0.845 | 0.821 | **+0.024** | 0.885 | 0.870 | **+0.015** |
| 40 | 0.346 | 0.280 | **+0.066** | 0.504 | 0.446 | **+0.058** | 0.775 | 0.732 | **+0.043** | 0.818 | 0.789 | **+0.029** |
| 50 | 0.375 | 0.305 | **+0.070** | 0.511 | 0.457 | **+0.054** | 0.780 | 0.741 | **+0.039** | 0.822 | 0.798 | **+0.024** |
| 60 | 0.286 | 0.218 | **+0.068** | 0.444 | 0.380 | **+0.064** | 0.723 | 0.663 | **+0.060** | 0.768 | 0.722 | **+0.046** |
| 70 | 0.289 | 0.217 | **+0.072** | 0.408 | 0.351 | **+0.057** | 0.685 | 0.630 | **+0.055** | 0.731 | 0.691 | **+0.040** |
| 80 | 0.290 | 0.220 | **+0.070** | 0.432 | 0.373 | **+0.059** | 0.706 | 0.646 | **+0.060** | 0.747 | 0.703 | **+0.044** |
| 90 | 0.247 | 0.182 | **+0.065** | 0.390 | 0.334 | **+0.056** | 0.667 | 0.602 | **+0.065** | 0.709 | 0.662 | **+0.047** |
| 100 | 0.221 | 0.151 | **+0.070** | 0.337 | 0.277 | **+0.060** | 0.603 | 0.528 | **+0.075** | 0.645 | 0.585 | **+0.060** |
| 110 | 0.224 | 0.165 | **+0.059** | 0.338 | 0.291 | **+0.047** | 0.604 | 0.543 | **+0.061** | 0.644 | 0.599 | **+0.045** |
| 120 | 0.215 | 0.156 | **+0.059** | 0.345 | 0.297 | **+0.048** | 0.613 | 0.546 | **+0.067** | 0.652 | 0.599 | **+0.053** |

Every lead wins all four, MAE and FSS@3km included. Note leads 10–20, advection's
home turf, where every prior generation lost: now +0.031/+0.026 on CSI@1.

## Why this worked — the objective was measuring the wrong quantity

c16's `multiscale_loss` average-pooled **log1p rain rate**; the eval's FSS pools an
**exceedance indicator** (`pr >= 0.1`, `eval_nowcast.py`). Those are different
objectives, and the gap was quantified on a synthetic blob before the run: comparing
a blurred field against a 4-px-displaced one,

    mode=rate        blurred/displaced = 0.071
    mode=exceedance  blurred/displaced = 1.013

so the c16 term charged a blurred hedge **7%** of what it charged a placement
error — nearly indifferent to the exact failure it was added to prevent. Pooling the
wet mask makes blur cost as much as displacement. That prediction held: FSS@9/15km
went from 0/12 to 12/12 while CSI@1 held at 12/12.

The implementation detail mattered too: the soft indicator is applied to **both**
sides. Soft-prediction-vs-hard-target is closer to the metric on paper but is not a
divergence — every dry pixel sits at σ(−0.1/0.05) ≈ 0.12 against 0, an irreducible
floor (measured 0.0139 on identical fields) that wastes gradient pushing an
already-clamped Softplus output down.

## Two negative results, both useful

- **Static terrain channels do nothing** (arm A): identical win-counts to baseline,
  deltas in the third decimal. Terrain is stationary; the 0–2 h task is advection
  correction, and orographic climatology is likely already implicit in the 6-frame
  history. Says nothing about the outlook/downscaling head. See `DELTA_c17_a.md`.
- **More data does nothing** (baseline re-eval): c16_full reproduced its 14-month
  win-counts *exactly* on a 74% larger validation set (98,899 → 171,985 samples).
  So c16 was not a small-sample artifact, and +10 months changed nothing.
- **Keep the advection prior.** Arm B drops it and still passes FSS, but loses a
  lead on CSI@1 (11/12) and −0.013 mean. So c16's inference that advection
  *suppresses* large-scale FSS was **wrong** — the loss was the cause all along, and
  advection is purely additive for heavy rain.

## Caveats before promoting
Single seed per arm. Thresholds are the eval's own (0.1 / 1.0 mm/h) and FSS τ = 0.05
mm/h, `--fss-weight 0.3` un-swept. 256² grid. The gate is met on **CSI, MAE and FSS
at 3/9/15 km**; it does not cover CRPS calibration or the outlook bands. Serving
also needs the live `li_flash` dependency handled — the model now expects a channel
whose feed went down for five days this month, and the LI crops encode "no flashes"
and "no data" identically (`nodata=0.0`).

## Recommendation
Promote `nowcast_mm_c17_c_exceed.pt` to the hetz1 `--producer model` path. The
checkpoint now records its own input recipe (aux/static channels, history steps,
advection flag, fss mode), so `produce_forecast.py` can assemble inputs faithfully
instead of re-deriving them — the failure class that made the old model path
unsafe. Remaining serving work: the live cadence-15 zarr with `li_flash` on hetz1, a
torch producer image, and an LI-staleness guard that falls back to classical.
