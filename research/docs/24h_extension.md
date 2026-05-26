# Can Pluvio honestly forecast precipitation 24 h out?

**Short answer: yes, by blending three sources. The radar nowcast we already use is good for 0–2 h; for 2–24 h we should call ALARO; for tomorrow and beyond, an NWP source (ECMWF Open Data) gives us 10 days.** Each source has known accuracy characteristics and the blend has to respect them — what we *don't* want is to pretend the radar nowcast extends to 24 h, because the underlying algorithm (echo advection) loses skill past ~2 h.

## The horizon ladder

| Horizon | Source | Cadence | Resolution | Notes |
|---|---|---|---|---|
| 0 – 2 h | KMI `getForecasts.animation` (the radar nowcast — what we already use) | 10 min | ~1 km Belgium-wide | The KMI mobile API gives 30 frames; ~3h max is the absolute limit. Skill drops fast past 1 h. |
| 2 – 24 h | KMI ALARO `Total_precipitation` via `opendata.meteo.be/service/alaro/wms` | 1 h | ~4 km Benelux+ | Numerical-weather-prediction model. Time dimension is ~60 h from each run; runs every 6 h. **No API key.** |
| 24 h – 7 d | KMI `getForecasts.for.hourly` | 1 h | point (lat/lon) | The same `getForecasts` call we already make. ~49 hourly entries. Covers our 24h ask and then some. |
| 24 h – 10 d | ECMWF Open Data IFS | 3 h | 0.25° | Free, public — keeps the last 4 days of runs at `data.ecmwf.int`. Use as a longer-range backup for the daily forecast tab. |

## Why the data layer should still treat these as separate sources

Tempting to merge them. Don't.

- **Verification stays clean.** Each source has its own skill curve. If we ever quote "Pluvio is accurate X% of the time at horizon H," we need to know which source drove that horizon.
- **Failure modes are different.** Radar nowcast fails when the storm is born outside the radar's view (a thunderstorm initiating *over* Brussels rather than drifting in from the North Sea). NWP fails differently — phase errors, intensity smoothing.
- **Latency budgets are different.** The nowcast updates every 10 min; ALARO every 6 h. Caching strategy and refresh cadence have to match.

## What changes in the Pluvio architecture

Minimal — the abstractions already in place are correct:

- Keep `RadarRepository` (radar nowcast). Bound to the 0–3 h horizon.
- Add `ForecastRepository` with two implementations:
  - `KmiHourlyForecastRepository` reading `for.hourly` out of the same `getForecasts` response — zero extra network for the first 48 h.
  - `AlaroForecastRepository` for "show me a precipitation map of next Wednesday afternoon" / gridded fields. Optional v2.
- A **stitching service** in the application layer decides which source feeds the UI for a given lead-time, and tags every UI element with its source for the attribution row.

## Doing it honestly

Even with a 24-h forecast available, we should:

1. **Show source attribution per surface** — "Radar nowcast (KMI)" vs "ALARO model (KMI)" vs "ECMWF IFS". Users notice when a "rain in 6 h" prediction is wrong; trust survives if we never pretended the 6h number came from the same radar that's bang-on for 30 min.
2. **Surface confidence**. Skill-based confidence bands (see `verification.md`) get wider with lead-time. The 24-h band is much wider than the 30-min band.
3. **Don't cross-blend within the same chart axis.** Two stacked panels: "Next 2 hours (radar)" / "Next 2 days (model)". The eye reads them as separate readings, not a continuous signal.

## Reproducing the 24 h pull

```bash
python collectors/fetch_alaro_24h.py --hours 24 --out data/alaro/
```

The script reads the time dimension out of `GetCapabilities`, picks every hour within "now → +24 h", and downloads each as a GeoTIFF. Confirmed live: ALARO publishes the next ~60 h on every run, so the 24-h window is always covered.
