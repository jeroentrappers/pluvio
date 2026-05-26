# Data inventory for the Pluvio improved-nowcast model

Everything verified live on 2026-05-26. The goal is to assemble enough independent
signals that a small learned model can outperform the operational
extrapolation nowcast — especially on convective intensification, which the
verification showed is where the radar-only nowcast collapses (CSI = 0 at +30 min
for τ = 4 mm/h).

## What each source contributes

| Category | Source | Spatio-temporal | Why it matters for nowcasting |
|---|---|---|---|
| **Radar history** | KNMI `radar_forecast` v2.0 (image1 = analysis) | 5-min, 1 km, NL+border | The actual rain field "now" and the recent past. Backbone input. |
| **Operational nowcast** | KNMI `radar_forecast` v2.0 (image2..25) | 5-min, 1 km, +5..+120 | The baseline we want to beat. Useful as input *and* as target residual. |
| **Ground truth** | KNMI `nl_rdr_data_rtcor_5m` v1.0 | 5-min, 1 km | Radar+gauge corrected observation. Training target. |
| **Belgian radar mosaic** | KMI `belgian_rainfall_composite` WMS | 5-min, 1 km, BE focus | Cross-border verification + Belgian-detail mosaic. |
| **Numerical forecast** | KMI ALARO WMS (`Total_precipitation`, `Total_cloud_cover`, wind, humidity, K-index, …) | 1-hour, ~4 km, +0..+60 h | NWP context — the model can "see" what the atmosphere is doing on 1-h scale. Bridges the 1-3 h gap where radar extrapolation fails. |
| **Surface obs (BE)** | KMI `aws_10min` WFS | 10-min, ~30 stations | Pressure, **pressure-tendency**, temp, humidity, wind, gusts, surface precip at known coordinates. Pressure-tendency is the leading indicator of approaching systems. |
| **Surface obs (NL)** | KNMI AWS (similar product) | 10-min, ~50 stations | Same fields north of the border. |
| **Satellite — instability** | EUMETSAT MSG `gii_kindex`, `gii_liftedindex` | 15-min, ~3-4 km | K-index > 30 / LI < -2 are the textbook thunderstorm-precursor signals. Available *before* echoes show on radar. |
| **Satellite — IR / WV** | EUMETSAT MSG `ir108`, `wv062` | 15-min, ~3-4 km | Cloud-top temperature (ir108) and mid-troposphere moisture (wv062). Cold IR + moist WV ⇒ deep convection. |
| **Satellite — RDT** | EUMETSAT MSG `rdt` | 15-min, polygons | NWCSAF "Rapid Developing Thunderstorms" — the most directly useful signal for the convective intensification radar can't see. |
| **Satellite — cloud-top height** | EUMETSAT MSG `cth` | 15-min, ~3-4 km | Cloud-top height in metres; > 8 km strongly correlates with intense rain. |
| **Lightning** *(future)* | Blitzortung (FOSS), or commercial via EUMETNET | <1-min, ~5 km | Real-time confirmation of active convection. **Not free in operational form** — Blitzortung is community-run but covers Europe acceptably. |
| **Long-range NWP** | ECMWF Open Data IFS, DWD ICON-EU | 3-h / 1-h, 0.25° / 0.06° | Beyond 24 h. Out of scope for the nowcast but worth keeping in the pipeline. |

## What we *don't* get for free

- **KMI's INCA blended product** (`/service/inca/wms`) — exists but returns 403. This is KMI's *own* learned blend of radar + NWP, and it's exactly what we want to beat. Worth a future ask via the KMI Open Data portal — they may grant non-commercial research access.
- **KMI satellite WMS** (`/service/satellite/wms`) — 403. We use EUMETSAT directly instead, same underlying SEVIRI data.
- **Real-time lightning from ATDnet / BELLS** — restricted. Blitzortung is the FOSS workaround.

## Licensing

- KMI / KNMI: CC BY 4.0.
- EUMETSAT: Free for non-commercial / research use; commercial use requires a separate licence. Pluvio's GPL-3.0 status keeps us non-commercial-friendly, but a paid app on the store would need to revisit.
- Blitzortung: For non-commercial research only. Stricter than CC BY.

## Implied training-data volume

- Radar history + nowcast + truth at 5-min cadence: a single timestep ≈ 1.5 MB
  (HDF5). 90 days × 288 / day × 1.5 MB ≈ **40 GB** for the radar side alone.
- Satellite at 15-min cadence × 3 layers: ≈ 4 MB / timestep ⇒ ~3 GB / month.
- AWS 10-min over BE+NL ⇒ negligible (CSV, sub-GB / year).
- ALARO 1-h × 5 layers × ~5° box ⇒ ~500 MB / month.

Plan on **~60 GB for three months** of paired training data. Cheap to store
on a workstation SSD; expensive to ferry around — keep it local.
