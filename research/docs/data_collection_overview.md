# Pluvio data-collection jobs — overview & audit (2026-06-17)

All collection runs on **hetz1** (`appmire-hetz1`, 136.243.103.58). Three mechanisms:
**(A)** systemd-timer collectors (newer "seamless" stack → `/mnt/storagebox`, ansible
roles under `deploy/ansible`), **(B)** cron collectors (legacy correction-model
inputs → `/opt/pluvio/data` local, via `tools/pull_forward.sh`), **(C)** one-off
backfill units (`systemd-run`).

## A. systemd-timer collectors → `/mnt/storagebox` (training stack)

| Job (unit) | Data | Cadence | Coverage (verified) | Status |
|---|---|---|---|---|
| `opera-forward` | **OPERA radar RATE+ACRR** (truth, LAEA 2 km) | 15 min | forward 2026-06-15→ ; backfill 2024-08→2025-12 | ✅ forward ok (24h bucket still has `.tiff`) |
| `mtg-li-forward` | MTG-LI lightning (flash accum.) | 15 min | 2025-06→now, ~4.3k/mo | ✅ (no source data pre-2025-06) |
| `mtg-l2-forward` | MTG-L2: **GII**, CTTH, OCA, CT, OLR | 15 min | GII 2025-06→now; **CTTH/OCA/CT/OLR only 2026-06→** | ⚠️ cloud products have ~no history |
| `icon-d2-forward` | ICON-D2 `tot_prec` (2.2 km NWP) | 3 h | **only latest ~day on disk (50 files)** | 🔴 not accumulating — see issues |
| `aifs-forward` | AIFS `tp` (global NWP → 240 h) | 6 h | 2026-06-15→ (3 days) | ✅ (rolling source, can't backfill) |
| `era5-forward` | ERA5 7 surface vars (monthly catch-up) | monthly | **2018-01 → 2026-06, complete** | ✅ |
| `pluvio-producer` | *(produces forecast for serving — not collection)* | 15 min | — | serving |

## B. cron collectors → `/opt/pluvio/data` (legacy correction-model inputs)

`ansible` user crontab, all via `/opt/pluvio/research/tools/pull_forward.sh <src>`:

| Job | Data | Cadence | On disk |
|---|---|---|---|
| `knmi-radar` | KNMI radar + operational nowcast | */30 min | 112 |
| `kmi-aws` | KMI automatic weather stations (BE gauges) | */10 min | parquet |
| `knmi-aws` | KNMI AWS (NL gauges) | */15 min | (aws/) |
| `meteosat` | MSG/Meteosat (RDT, IR …) | */30 min | 654 (msg/) |
| `alaro` | ALARO NWP (RMI Belgium) | 0,6,12,18 | 666 |
| `sst` | sea-surface temperature | daily 06:00 | 4 |
| `netatmo` | Netatmo personal weather stations | (cron) | — |

These feed the **older 22-month correction model**; still running. (Note: the 122 GB
`/mnt/storagebox/data` is **not** these — that's other/gpsinfo data.)

## C. backfill units running now (`systemd-run`, transient, resumable)

| Unit | Filling | Status |
|---|---|---|
| `opera-h5-gapfill` | OPERA RATE 2026-01→06-15 (via new ODIM `.h5` reader) | running |
| `mtg-l2-chunk` | GII 2025-08→2025-10 | running |
| `mtg-l2-fill` | GII 2025-10→2026-04 | running |
| `radklim-backfill` | RADKLIM-YW (DE gauge-adj radar) 2024-08→now | running (perm-fixed) |

## Issues found in the audit (2026-06-17) & status

1. **OPERA truth 2026-01→mid-June missing** — archive went HDF5-only in 2026; collector read `.tiff` only. **Fixed**: added ODIM `.h5` reader (`crop_odim_h5`); gap backfilling. 🔧
2. **RADKLIM 0 files (silent, hours)** — CIFS mount root-only-writable, image runs non-root. **Fixed**: remounted `dir_mode=0777`; relaunched non-root. 🔧
3. **GII 2025-09→2026-02 hole** — only one chunk job configured. **Fixed**: fill chunk launched. 🔧
4. **OPERA ACRR also had the 2026 gap** — **fixed**: `opera-acrr-gapfill` launched (ACRR is structurally identical to RATE; reuses the ODIM `.h5` reader). 🔧 backfilling
5. **MTG-L2 cloud products (CTTH/OCA/CT/OLR) had no history** — **fixed**: `mtg-l2-cloud` backfill launched. Verified earliest at EUMETSAT: OCA/OLR ~2025-06, CTTH/CT ~2025-12 (no source data before those). 🔧 backfilling
6. **ICON-D2 "not accumulating"** — **non-issue**: its forward timer first ran 2026-06-17 10:42, so only today's runs exist *because it was deployed today*. No cleanup deletes it; it writes run-stamped files and accumulates normally. ✅ resolved
7. **Mount fix is on the gpsinfo-tiles `mnt-storagebox.mount` unit** (not pluvio's ansible) — codify there or it reverts. ⚠️ open (config hygiene)
8. **Rotation cron + `rotate_to_nas.py` fix** (aws-dir handling, CIFS `copyfile`) live on the box + repo, but the legacy cron collectors aren't in pluvio's ansible — codify to survive a rebuild. ⚠️ open (config hygiene)

## Verified healthy
ERA5 (complete 2018→2026-06), MTG-LI (2025-06→now), GII all 3 datasets consistent,
AIFS (forward, current). Value-sanity probes passed for ICON-D2, AIFS, GII, ERA5, OPERA.
