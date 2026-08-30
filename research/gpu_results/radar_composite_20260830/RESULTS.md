# Own radar QPE vs OPERA, scored against rain gauges — 2026-08-30

**On this sample our single-radar estimate beats OPERA against independent rain
gauges, by roughly 2x on CSI.** That is one radar, one lowest sweep, no multi-radar
merging — against a professionally quality-controlled European composite.

## Why gauges

OPERA is an *estimate*, not truth, so "as good as OPERA" cannot be settled by
comparing to OPERA. Both estimators are scored against KNMI 10-minute rain gauges
(`rg`, "Precipitation Intensity (Rain Gauge) Mean", mm/h — the same units), ~58 NL
stations. Belgian gauges (`opendata.meteo.be` WFS `aws:aws_10min`,
`precip_quantity`) are wired in too but Belgium was dry on the test day.

## Headline

Full day, 599 station-times within 120 km of nlhrw, strict single-cell sampling:

| | POD | FAR | CSI | corr | detected (of 105 wet) |
|---|---|---|---|---|---|
| **ours** | **0.495** | **0.388** | **0.377** | **0.481** | **52** |
| OPERA | 0.095 | 0.565 | 0.085 | 0.192 | 10 |

Held-out half of the day (300 pairs, 53 wet, 3x3 sampling):

| | POD | FAR | CSI | bias | MAE | corr | mean@wet |
|---|---|---|---|---|---|---|---|
| ours raw | 0.830 | 0.546 | **0.415** | −0.066 | **0.354** | **0.686** | 1.288 |
| ours cal (power) | 1.000 | 0.745 | 0.255 | −0.063 | 0.501 | 0.530 | 0.725 |
| ours cal (scale) | 0.906 | 0.575 | 0.407 | +0.085 | 0.404 | 0.686 | 1.896 |
| OPERA | 0.264 | 0.562 | 0.197 | −0.275 | 0.417 | 0.311 | 0.319 |

gauge mean 0.386 mm/h overall, 2.160 where wet.

## The bug that had to be fixed first

The original comparison was invalid: `reproject_to_analysis_grid` warped OPERA with
bilinear resampling while treating its nodata as missing. OPERA encodes **no rain as
nodata** (~92% of the tiff), so interpolation eroded rain areas from their edges and
deleted small ones — **7031 raw wet pixels inside the analysis bbox became 357 wet
cells, 12% retained**. Fixed via `nodata_as_zero` (see a66f8fc). OPERA's scores
roughly tripled afterwards (CSI 0.038 → 0.250 on the 4-slot sample); the numbers
above are all post-fix.

That function also builds `opera_rate` — the training truth for every model
generation — and the serving-time OPERA history. See the commit for why the fix must
not be deployed to serving before retraining.

## Gauge calibration: a negative result

Neither correction improved held-out skill.

- Free power fit `R = c·R_raw^d` gave d = 0.28 from only 36 mutually-wet training
  pairs — a near-flat exponent that collapses dynamic range. POD rose to 1.000 but
  FAR to 0.745 and CSI **halved** (0.415 → 0.255).
- Mean-field scale (×1.472, d pinned to 1) fixed the intensity under-read
  (mean@wet 1.288 → 1.896 against a gauge 2.160) but bias overshot to +0.085, MAE
  worsened 0.354 → 0.404, and CSI did not improve.

Raw Marshall-Palmer (Z = 200 R^1.6) stays the operating point. The honest reading is
that 36 wet training pairs is far too few to fit anything; this should be revisited
once the archive holds more rain days.

## Caveats — this is one sample, not a general claim

- **One radar (nlhrw), one day, one regime.** Light-to-moderate stratiform rain;
  no convection, no snow, no bright band. Volume capture only began 2026-08-30.
- **Point-to-pixel mismatch is real.** A gauge integrates ~200 cm² over 10 minutes;
  a radar bin samples ~1 km³ aloft, instantaneously. Mitigated by scoring hundreds
  of station-times rather than trusting any pair, and by checking halo 0/1/2 — the
  ranking is unchanged across all three.
- **Both estimators badly under-read intensity** (gauge 2.16 at wet stations; ours
  1.29, OPERA 0.32). Detection is where we lead; absolute rate is not solved.
- **We are not yet a composite.** Single radar, single elevation. Multi-radar
  merging, beam blockage (bewid in the Ardennes is the hard case), VPR and
  attenuation correction are all still absent.
