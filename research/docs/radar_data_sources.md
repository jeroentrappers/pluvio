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
