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

## Regional evaluation vs best known composites (2026-08-31, window 16:10–19:05Z)

Truth = independent gauges, never another radar product. Scored on the SERVED 4-km
cube (display product with interpolants), not the 1-km research chain that produced
the BE/NL numbers — a deliberate first pass at what users actually see. Harness:
`tools/regional_eval.py`.

**Germany** (truth: 1371 DWD 10-min stations; competitors RADOLAN RY + OPERA):

| region | n / wet | ours @0.1/0.5/1/2 | RADOLAN | verdict |
|---|---|---|---|---|
| DE-west <12E (own core) | 15,556 / 1,556 | .319/.294/.271/.240 | .422/.391/.358/.297 | **RADOLAN, all thresholds** |
| DE-east ≥12E (OPERA fill) | 5,740 / 539 | .260/.246/.200/.177 | .420/.404/.315/.244 | **RADOLAN, all thresholds** |

Wet bias: ours +7.3 mm/h, RADOLAN +2.7 — a convective evening scored without gauge
adjustment against the gauge-adjusted national standard. OPERA columns were
inconclusive (RATE archive covered only 1,034/32 windows; on that joint subset ours
ties OPERA).

**UK** (truth: 1,040 EA 15-min stations; ours = UKMO composite through our fill):
n=10,447, wet=151 (near-dry evening). Wet bias **ours +0.23 mm/h** — the fill chain
(units, regrid, geometry) is faithful. Absolute CSI 0.12–0.16 on scattered drizzle at
4 km/15 min says little with 151 wet windows; vs OPERA tie on the small joint subset.

**Poland**: IMGW synop API gives hourly totals but no station coordinates —
quantitative eval deferred until a coordinate table is wired in.

What this changes:
1. **Gauge adjustment is now the top lever** — the DE gap is mostly bias (+7.3 vs
   +2.7), exactly what Appendix-B spatial adjustment removes; DWD's own 10-min gauge
   feed (proven fetchable here) can drive it operationally, as can EA/KNMI/KMI feeds.
2. Re-run this eval against the 1-km research chain before drawing chain-vs-serving
   conclusions, and on a stratiform day (convective evenings are the hardest case).
3. Calibration offsets for the DE/DK/PL newcomers still default to 0 — the overlap
   fit needs its wet day.

### UK fill fidelity vs the UKMO source (2026-08-31, 17:45–20:00Z, 10 slots)

Per 15-min slot, served UK box vs the source composite warped through the same chain:
cross-correlation displacement **(0,0) km in 8/10 slots** (outliers +7/+8 km, within
one downsampled cell); top-cell centroid distance 0–5 km (one 18 km on a near-dry
slot); p99 intensity identical to ±0.02 mm/h; field maxima track within ~10–20%
lower — the expected cost of average-resampling their 1-km product to our 4-km grid.
Conclusion: cells over the UK are where the Met Office puts them, at the intensity
they publish; a visual A/B against weather.metoffice.gov.uk's radar layer is
trustworthy by construction (same underlying product, licensed via the AWS bucket).

## Overnight stratiform verdict + 1-km cost (2026-09-01, 00:25–03:25Z window)

**DE-west (own radar core) statistically TIES RADOLAN at all four thresholds** in the
stratiform regime (ours .295/.288/.254/.142 vs .288/.314/.304/.176, every bootstrap
CI spanning zero; bias@wet +1.06 vs +0.53) — with NO gauge adjustment active. The
convective-evening gap was therefore bias-dominated, exactly what the adjustment
battery showed (winner: Appendix-B adjustment with r_s=20 km; Z-R variants were
noise). DE-east (OPERA fill) still loses at all thresholds in both regimes: the fill
is the weak half, and the fix is admitting the eastern-DE radars on their wet day,
not more tuning.

1-km continental frame cost, measured warm under contention: ~2 min (expect 60–90 s
quiet), assembly ~3 s/gap — borderline for the live 5-min cadence; deploy plan is
1-km with a measured end-to-end timer cycle and a documented fallback to 1.5 km if
it overruns.

## Deployment gate + winner rollout (2026-09-01 morning)

Winner config (Appendix-B gauge adjustment r_s=20 km + merged newcomer calibration,
clipped ±4 dB) passed all gates and rolled out:

- **Frozen convective window**: campaign-best DE-west (.348/.316/.298/.260),
  every OPERA cell a tie, RADOLAN gap −0.07..−0.08 (was −0.10..−0.13 uncorrected).
- **Overnight stratiform window**: DE-west parity with RADOLAN preserved at all
  thresholds (bias +0.95 vs their +0.52); adjustment did not overcorrect.
- **NL guard improved at every threshold** (0.415→0.444 @0.1 — ahead of RTCOR's
  0.387 on trace detection; mid-threshold gaps narrowed on a 25-wet-window sample).
- UK near-dry, bias +0.04 — sane.

Serving now: 27 radars + dual fill on the 1-km continental grid (1948×2476) with
gauge adjustment live (hourly feeds from KNMI/KMI/DWD/EA), viewport-tiled hi-res
serving + 4x block-mean overview, calibration including the DK/dehnr/deboo relative
fits (the central-DE ring and DK seams). Remaining open items: DE-east stays fill
until its radars' wet-day regates.

**1-km live-cadence verdict (measured 2026-09-01 ~04:45Z):** the first live timer
cycle ran >6 minutes without completing (each catch-up/upgrade frame ~2 min at
1948×2476, plus ~3 s/gap reassembly of 44 gaps) — the cube's newest frame aged to
54 min and the staleness guard was minutes from 503. The documented fallback
executed: serving grid 1301×1652 (~1.5 km, still 2.7x the 4-km era), identical
config otherwise. The road back to true 1-km serving: cache optical flows per gap
(cuts reassembly from ~2.5 min to seconds) and make upgrade recomputes incremental
per radar rather than full-frame.

**1.5-km live cadence confirmed** (2026-09-01 05:29–05:35Z): steady-state timer
ticks complete in 33 s and 70 s (catch-up tick 3:21), cube advancing every 5-min
slot. Serving end state: 27 verified radars + UKMO/OPERA fills, gauge adjustment
(r_s=20 km) from four national networks, merged calibration, motion-morphed
interpolants, viewport tiles (7×6 × 256 px) + 3x overview, Met Office palette.

## ES/PT/IT/AT open-data sweep (2026-09-01, verified empirically)

| country | source | verdict |
|---|---|---|
| **IT** | DPC Radar-DPC API (radar-api.protezionecivile.it) | **OPEN, VERIFIED**: `findLastProductByType?type=SRI` → `POST downloadProduct` → presigned GeoTIFF. SRI = surface rain rate mm/h, 5-min, 1400×1200 covering 4.5–20.5E / 35.1–47.8N (all Italy), CC-BY-SA, no auth. Smoke-tested end to end. |
| **AT** | GeoSphere Data Hub (dataset.api.hub.geosphere.at) | **OPEN, VERIFIED**: `/grid/forecast/nowcast-v1-15min-1km` carries `rr` (INCA analysis+nowcast, radar+stations), bbox 8.1–17.7E / 45.5–49.5N (all Austria), 15-min, 1 km. Needs UKMO-style slot morphing for 5-min display. Also `inca-v1-1h-1km` historical to 2011 for training truth. |
| **ES** | AEMET OpenData | **Dead end for now (probed with a registered key, 2026-09-01):** `/api/red/radar/nacional` returns 404 persistently; `/api/red/radar/regional/{r}` returns non-georeferenced 480×530 GIFs; the help page advertises georeferenced GeoTIFFs (EPSG:4326, RGBA + ESCALA colour map) but no public endpoint exists in the API spec, the product catalog, or the site pages. Key stored in .env (AEMET_API_KEY) — re-probe periodically; their radar modernization and EUMETNET's open-radar-data program may open it. |
| **PT** | IPMA | api.ipma.pt/open-data has NO radar entries — website imagery only. Closed for now. |

Integration plan: IT SRI as a third fill layer (outranks OPERA over Italy, where OPERA
holds ~45%); AT rr as a fourth (outranks OPERA over Austria, ~20%), slot-morphed like
UKMO. Both fit the current box only partially (box S=42.5 cuts Italy at Rome's
latitude) — full-Italy coverage means widening the box to S≈35, a +43% pixel payload
decision to take deliberately.

## GR/MA/Balkans/Baltics sweep (2026-09-01)

- **Greece**: closed — HNMS publishes radar as website imagery only, is absent from
  the OPERA single-site bucket, and contributes nothing to the COMP composite
  (measured 0.0% in-raster over a GR box). Satellite-only territory.
- **Morocco**: radar-dark (0.7% ≈ noise at the Alboran edge). Satellite-only.
- **Satellite frontier** (GR/MA/open sea/beyond-radar): EUMETSAT H-SAF P-IN-SEVIRI
  (15-min rain rate, registration) or GPM IMERG Early (30-min, ~4 h latency — too
  slow for live history). A coarse outermost fill ring is feasible; separate build.
- **Balkans/Baltics — single-site feeds EXIST in the bucket**: HR hrbil hrdeb hrgol
  hrgra hrpun hrulj · RO robar robob robuc rocra romed roora rotim · SI silis sipas ·
  EE eehar · LT ltlau ltvil. All 18 sent through the standard verification gate
  (OPERA reference, per-country boxes). Admission implies an east/南 box decision:
  E→~30 (+16% pixels at 1.5 km) for RO/Baltics; HR fits the current south edge bar
  Dubrovnik.

### Peer-gate round for HR/SI/RO (2026-09-01 afternoon)

Peer-referencing (candidates vs their overlapping neighbours) separated two failure
stories the OPERA-void FAILs had conflated:

- **HR (all 6) + silis: static clutter, not geometry.** Site coords correct, rows
  north-aligned (startazA≈359.6), self-corr 0.86–0.96 — but negative geocorr even
  against their own peers. A ~0.95 self-correlation on a 1–4%-wet field is a STATIC
  pattern: these volumes carry only DBZH/TH/VRADH (no RhoHV), the fuzzy declutter
  cannot run, and Dinaric/Alpine ground echoes dominate. Admission blocked on a
  Doppler-based clutter filter (VRADH is present: near-zero radial velocity + static
  echo → clutter) — engineering item, queued.
- **RO: romed PASSES against peers** (its OPERA-void FAIL was a reference artifact);
  rocra/rotim stay negative even vs peers (Carpathian clutter suspected, same class);
  robar/roora near-dry holds.

**Admission set from the round: robob, robuc, romed, ltlau, ltvil** — pulls the
serving box east (E 24.5→30.0, ≈+15% pixels at 1.5 km). eehar holds (single radar,
no peers, reference void — recheck against Finnish neighbours on a wet day).

## Pan-European sweep round 2 + consolidated rollout results (2026-09-01)

**Rollout landed:** Italy (DPC SRI) and Austria (GeoSphere INCA) fills live at 100%
coverage in their boxes; **Ireland freeze cleared** (watchdog freeze_frac 0.67 → 0.0)
by the UKMO slot-morph + OPERA-first layering. New watchdog flag under
investigation: PARITY-PULSE over BE/DK/IE/UK on the fresh cube (lag-1 −0.49..−0.98)
— morph/mask interaction or thresholds too hot on low-wet morning fields; verdict on
the east-extension cube.

**New sources verified this round:**
- **Slovakia (opendata.shmu.sk, TLS chain broken — fetch with -k):** ODIM HDF5
  composites (zmax/cappi2km/etop/pac01, 5-min, ~9-min latency) AND four single-site
  volumes (skjav skkoj skkub sklaz) with FULL dual-pol (dBZ dBuZ ZDR RhoHV PhiDP KDP
  V W) — chain-grade radars, standard gate applies. Best new catch of the sweep.
- **Hungary (odp.met.hu, CC-BY-SA):** 5-min national composite, classic NetCDF-3
  inside zip (refl2D 813×961, regular lat/lon via La1/Lo1/Dx/Dy), ~9-min latency.
  Fill-layer grade (reflectivity → Marshall-Palmer).
- **Nordics/IS/MT from the bucket:** FI×12 NO×12 SE×11 IS×4 MT×1 — 40 candidates in
  the standard gate (running).
- **Outer ring closed:** RS/BA/BG/TR/UA — website imagery only, nothing open.

Integration ladder (each addition = gate/verify → config → one rebuild):
E-extension (LT/RO) live next; then SK volumes + HU fill (in-box, no geometry
change); then Nordics decision (N→71 doubles rows — payload/cadence tier to choose
deliberately); HR/SI wait on the Doppler declutter build.
