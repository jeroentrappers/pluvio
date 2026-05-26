# Verification methodology — nowcast skill vs. KMI / KNMI

How "good" is the radar nowcast? The honest answer is a curve, not a number. Here's how we draw it.

## 1. Pairing forecasts to observations

The KMI mobile-app endpoint ships a sliding window. The *same* timestep appears in successive polls — first as a forecast, then as an observation. Concretely:

| time | poll @ 10:00 (lead=+30) | poll @ 10:10 (lead=+20) | poll @ 10:20 (lead=+10) | poll @ 10:30 (lead=0) |
|---|---|---|---|---|
| T = 10:30 | 1.2 mm/h | 0.8 mm/h | 0.5 mm/h | **0.3 mm/h (obs)** |

So if we persist every poll (Option A) or every archived forecast (Option B), we can — for each `(location, frame_ts)` pair — recover the prediction at lead times `Δ ∈ {10, 20, …, 180}` min and the eventual observation at `Δ = 0`. That gives us N samples per lead-time bin.

Option B is faster to bootstrap because KNMI keeps both products in its archive — see `collectors/fetch_knmi_archive.py`.

## 2. Metrics

### 2.1 Continuous (mm/h)

For each lead-time bin, with forecast values `f_i` and observed values `o_i` (i = 1…N):

- **MAE** = `mean(|f_i − o_i|)` — interpretable in mm/h.
- **RMSE** = `sqrt(mean((f_i − o_i)²))` — penalises big misses harder.
- **Bias** = `mean(f_i − o_i)` — positive = over-prediction.
- **MSE skill score vs. persistence** = `1 − MSE_model / MSE_persistence`. The persistence baseline says "the rate observed at issue time will hold for the whole horizon". A skill > 0 means the nowcast beats that. For radar nowcasts the skill is typically high at small lead-times and crosses zero between 60 and 90 min.

### 2.2 Categorical (rain / no-rain, threshold τ)

Contingency table per lead-time, for a threshold τ (e.g. 0.1 mm/h, 1 mm/h, 4 mm/h):

|  | Obs ≥ τ | Obs < τ |
|---|---|---|
| Fcst ≥ τ | Hit (H) | False alarm (FA) |
| Fcst < τ | Miss (M) | Correct negative (CN) |

- **POD** = `H / (H + M)` — probability of detection (recall).
- **FAR** = `FA / (H + FA)` — false-alarm ratio (1 − precision).
- **CSI** = `H / (H + M + FA)` — critical success index, ignores correct negatives.
- **HSS** = `2(H·CN − M·FA) / [(H+M)(M+CN) + (H+FA)(FA+CN)]` — Heidke skill score, ranges (−∞, 1], > 0 beats random.

### 2.3 Stratify

The headline curve is misleading without slicing. Always also report:

- per **intensity band** (light / moderate / heavy / violent) — nowcasts handle drizzle well and storms poorly.
- per **time of day** — convective storms behave differently to advected stratiform rain.
- per **location** — coastal vs inland coverage varies with radar geometry.

## 3. What "good" looks like for INCA-class extrapolation nowcasts

From the public literature (KNMI's pysteps benchmarks, MeteoSwiss INCA papers, Google's MetNet-3 ablations) you can expect, for a 0.1 mm/h rain/no-rain threshold:

| lead | CSI typical | RMSE (mm/h) typical |
|---|---|---|
| +10 min | 0.65–0.75 | 0.4–0.6 |
| +30 min | 0.45–0.55 | 0.8–1.2 |
| +60 min | 0.30–0.40 | 1.3–1.8 |
| +90 min | 0.20–0.30 | 1.7–2.3 |
| +120 min | 0.12–0.20 | 2.0–2.8 |

If our numbers fall well below those, something's wrong with our pairing (timezone, off-by-one on lead). If they're well above, recheck your test/train split.

## 4. Reproducing the analysis

```bash
# 1. Pull a week of KNMI radar_forecast + nl-rdr-data-rtcor-5m
python collectors/fetch_knmi_archive.py --start 2026-05-19 --end 2026-05-26 \
    --dataset radar_forecast
python collectors/fetch_knmi_archive.py --start 2026-05-19 --end 2026-05-26 \
    --dataset nl-rdr-data-rtcor-5m

# 2. Open the notebook
jupyter lab notebooks/01_verification.ipynb
```

The notebook joins the two products on the radar-pixel grid (KNMI publishes both on the same 765×700 grid in `EPSG:4326`-ish projection), bins by lead-time and intensity, and produces the canonical skill plots.
