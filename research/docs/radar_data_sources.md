# Radar data sources beyond the OPERA 24-h cache

Verified 2026-08-31, each with an actual request, not from documentation alone.

## Already flowing through our capture (OPERA single-site cache, 24 h, no backfill)

Counted in /mnt/storagebox/radar_volumes for 2026-08-31: FR 24 radars, FI 12, NO 12,
SE 11, PL 10, DK 5, IS 4, IE 2, EE 1 — plus BE/NL/DE/CH/CZ/HR/LT/MT/RO/SI. So
"interior France" and the Nordics are ALREADY captured live; the cache's only
limitation is that history before capture started (2026-08-30) is gone.

## Deep open archives (backfillable!)

| source | span | access | format |
|---|---|---|---|
| KNMI (nlhrw, nldhl) | 2008/2019 → now | API key (have) | KNMI HDF5 (reader: tools/knmi_volume) |
| **FMI Finland, full network** | **2007 → now** | S3, anonymous | ODIM PVOL h5 |
| DWD Germany, 17 sites | ~2 day rolling | anonymous | ODIM per sweep (reader: tools/dwd_volume) |

FMI: `https://fmi-opendata-radar-volume-hdf5.s3.eu-west-1.amazonaws.com/?prefix=2026/08/31/`
— keys like `2026/01/01/fianj/202601010000_fianj_PVOL.h5`. ⚠️ Do NOT use a leading
slash in the prefix; `prefix=/2026/` silently returns nothing (measured — cost us a
wrong "archive stops in 2023" conclusion for half an hour).

## United Kingdom — the genuine gap

The UK Met Office does not share single-site data through OPERA's open exchange, and
that remains true. What exists (CEDA, Open Government Licence, registration required —
directory listings are anonymous but file GETs 302-redirect to auth, measured):

* `badc/ukmo-nimrod` single-site `raw-polar/` + `raw-dual-polar/`: OLD years only
  (castor-bay: 2011-2013). Not a live source.
* Nimrod 1 km UK + 5 km Europe COMPOSITES: 2003 → present, 5-min. Nimrod binary format
  (open-source Python reader exists: richard-thomas/MetOffice_NIMROD).

Access verified 2026-08-31 with a CEDA account: tokens mint via
`POST https://services.ceda.ac.uk/api/token/create/` (HTTP basic auth; credentials in
/opt/pluvio/research/.env, current token in .ceda_token, ~3-day lifetime so mint per
run). With the token the **uk-1km composite downloads (HTTP 206 verified on a 2026
file)**; the single-site raw-polar datasets still return 403 — they need a separate
per-dataset access application on the CEDA catalogue, and only cover old years anyway.

So the UK fill layer is unlocked: ingest `data/composite/uk-1km/<year>/*.tar` (Nimrod
binary format, open-source Python reader: richard-thomas/MetOffice_NIMROD) as a
processed-composite fill, same role as OPERA fill. No open path to live UK volumes.

## Role in the product

Our champion QPE only covers where we have polar volumes. For the full-domain training
grid the layering is: our composite where covered → national composites (UK Nimrod)
where available → OPERA elsewhere, with a provenance flag per cell so training can
weight truth quality. Blending INTO our covered area is a separate question — OPERA
detects far less at gauges than we do (measured throughout this campaign), so anything
beyond a weak dry-veto needs a gauge evaluation first.

## Wide-coverage integration (2026-08-31)

Every candidate passed through `tools/verify_radar.py` (self-correlation across two
consecutive scans + geo-correlation against the verified composite) before entering the
composite — the a1gate incident made this gate policy, not preference. Results at
20260831T1300, reference = 6 verified BE/DE radars:

| radar | selfcorr | geocorr | wet% | verdict |
|-------|---------:|--------:|-----:|---------|
| frave | 0.176 | 0.306 | 2.9% | PASS — in |
| frtro | 0.101 | 0.095 | 1.3% | PASS — in |
| frnan | 0.088 | 0.341 | 1.6% | PASS — in |
| dehnr | 0.024 | 0.314 | 1.6% | PASS — in |
| frabb | −0.052 | **−0.138** | 1.0% | FAIL — negative geocorr, the scrambled-geometry signature; re-verify wet day |
| frtra | 0.051 | **−0.271** | 0.6% | FAIL — same; near-dry footprint too |
| deoft | −0.106 | 0.144 | 0.8% | HOLD — positive geocorr but near-dry; re-verify wet day |
| defld | −0.143 | 0.223 | 1.3% | HOLD — same |

Serving configuration (systemd drop-in `pluvio-observed.service.d/wide.conf`):
12 radars (`nlhrw nldhl behel bejab bewid deess denhb deasb frave frtro frnan dehnr`),
bounds **−1.2..12.2 E, 46.3..55.3 N** (Paris latitude to the Danish border), shape
416×400 (~2.4 km/px — same tile budget as the old Belgium box at 1.2 km). New radars
carry calibration offset 0.0 until an overlap fit on a wet day; the research/QPE-archive
grid is unchanged (that is training truth, scoped to the gauge-verified region).

## Continental coverage via the OPERA COMP fill layer (2026-08-31)

The bucket's `OPERA/COMP` prefix carries the FULL pan-European OPERA composite —
`OPERA@<stamp>@0@DBZH.tiff`, 4400×3800 at 1 km LAEA (lon −39.6..57.8, lat 31.7..73.9),
**5-minute cadence, ~4-minute publish latency** (measured: the 14:20Z file was
downloadable at 14:24:34Z). This is distinct from the 531×531 NW-Europe RATE crop we
archive to `/mnt/storagebox/opera/RATE`.

Measured per-country content (internal mask = out of any contributing radar's reach):

| region | in-composite | note |
|--------|-------------:|------|
| IE, PL, SE, NO, DK, CH, CZ | 55–100% of land | national networks contribute — full practical coverage |
| IT | ~45% | partial national contribution |
| UK | ~35% | UKMO does NOT contribute; only what IE/FR/Benelux radars reach (SE England, fringes) |
| AT | ~20% | edges from DE/CZ radars only |
| ES | ~13% | north strip from FR radars; AEMET absent |
| PT | 0% | IPMA absent |

Serving architecture (produce_observed): **own verified 12-radar composite wins
wherever it has coverage — including its zeros — and OPERA COMP fills the rest**,
converted dBZ→rate with the chain's Marshall-Palmer pair; masked pixels stay NaN so
uncovered areas render transparent. Frames whose fill was missing are sidecar-marked
and recomputed inside the upgrade window, exactly like radar-incomplete frames.
Serving box: **−11..24.5 E, 42.5..60 N at ~4 km/px (488×624)** — Ireland to eastern
Poland, northern Spain to mid-Scandinavia.

Remaining gaps and their only real fixes:
- **UK interior — SOLVED**: `s3://met-office-radar-obs-data` (AWS Open Data,
  eu-west-2, anonymous, CC BY-SA) carries UKMO's own national composite:
  `radar/YYYY/MM/DD/YYYYmmddHHMM_ODIM_ng_radar_rainrate_composite_1km_UK.h5`,
  1725x2175 at 1 km, OSGB transverse Mercator, float32 mm/h with -1 = nodata,
  15-min cadence, **measured ~14-min latency** (14:30Z file present 14:44Z). Covers
  Britain AND Ireland (-12..16 E, 43.8..62.9 N). Serves as a second fill layer that
  outranks OPERA over the British Isles (lon <= 2.2, lat >= 49.5) and is masked out
  elsewhere, where continental OPERA radars are closer.
- CEDA Nimrod `uk-1km` (working token access) lags ~2 days (latest 20260829 on
  2026-08-31): QPE-archive/training truth over the UK only, not live fill.
- **ES/PT interior**: AEMET/IPMA national feeds only; no open volume or composite path found yet.
- **IT/AT interior**: national products (DPC mosaic / GeoSphere) — possible future fill layers.
- Own-core upgrades from open volumes, pending the verification gate on a wet day:
  CH(5) CZ(2) DK(5) PL(10) IE(2) radars + ~9 more DE + ~20 more FR.

## Second gate round — DK/PL/CZ/CH/IE + remaining DE (2026-08-31, OPERA reference)

`verify_radar --opera-ref` scores far candidates against the pan-EU OPERA composite in
a box around each country (the own-composite reference can't reach them). A find_volume
fix landed first: PL/CZ package full volumes with elevation-list filename tokens at
stamps offset ~1 min from ours, which the widened search used to reject as "no data".

**Admitted** (clear pass, wet footprint): dkbor dkrom dksam dksin dkste (geocorr
0.40–0.56, 1–12% wet) · deboo (0.579, 1.7%) · plbrz plgsa plleg plpas plpoz plram
plrze pluzr (0.29–0.44, 1.2–3.7%) · czska (0.602, 1.3%). Serving set is now
**27 radars**; all newcomers at calibration offset 0.0 pending overlap fits.

**Held for a wet-day recheck** (near-dry footprint or ambiguous):
- plgdy, plswi (plswi geocorr 0.046 at 1.2% — genuinely suspicious, not just dry)
- czbrd (geocorr 0.635 but 0.96% wet)
- all CH (chalb chdol chlem chppm chwei — bone-dry footprints today)
- iedub/iesha — iedub's site coords and wet-centroid placement check out (~40 km from
  OPERA's, same system over Connacht); its −0.405 geocorr is the union-wet mask
  punishing narrow 250-km coverage against OPERA's Shannon view, so metric artifact
  more likely than geometry bug. Recheck with rain nearer Dublin.
- DE near-dry: dedrs deeis defbg deisn demem deneu depro deros detur deumd
- FR earlier: frabb frtra (negative geocorr — the one class that must never enter
  unresolved), deoft defld

## Forecast benchmarks for the nowcast retrain (noted 2026-08-31)

When the nowcast is retrained on the QPE-archive truth, benchmark it not only against
gauges/RTCOR but against operational FORECASTS on the same windows:
- **Met Office UKV rainfall** (and their nowcast blend) via the Weather DataHub API —
  the public map at weather.metoffice.gov.uk/maps-and-charts/rainfall-radar-forecast-map
  shows exactly what they publish; the DataHub UK-2km atmospheric API is the licensed
  route to the same fields. Scoring our lead-time skill against theirs over the British
  Isles gives an external yardstick nobody can accuse of home-field bias.
- Same idea holds for other nationals later (KNMI harmonie/nowcast, RMI's INCA-BE) —
  one yardstick per region where we now serve coverage.
