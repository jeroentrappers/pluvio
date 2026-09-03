# Operations schedule — hetz1 (inventory 2026-09-03, WBS 1.8)

Declared inventory of every recurring job. Target state: all of these are
systemd units generated from this manifest; no crontab. Until then this file
is the single place that says what runs, when, and why.

## Cron (user `ansible`) — to be converted to systemd timers

| cadence | job | purpose |
|---|---|---|
| */30 | `pull_forward.sh knmi-radar` | KNMI radar products → /opt/pluvio/data |
| */10 | `pull_forward.sh kmi-aws` | KMI station observations |
| */15 | `pull_forward.sh knmi-aws` | KNMI station observations |
| */30 | `pull_forward.sh meteosat` | MSG satellite channels |
| 00:30/06:30/12:30/18:30 | `pull_forward.sh alaro` | ALARO NWP fields |
| 06:00 | `pull_forward.sh sst` | OSTIA SST GeoTIFF (arrives ~D+2 06:00) |
| */30 | `pull_forward.sh netatmo` | Netatmo crowd gauges |
| */5 | `build_zarr --append` → `model.infer_latest` | feature store append + v2 nowcast → `serve/model_nowcast.npz` (tightened from */15 on 2026-09-02) |
| 01:45 | `rotate_to_nas.py` | stage → NAS rotation |

## systemd timers (already declared)

| timer | cadence | purpose |
|---|---|---|
| pluvio-observed | every 5 min | composite producer (produce_observed) — the serving cube |
| pluvio-qc | hourly :25 | temporal-consistency watchdog on the served cube |
| pluvio-qc-inputs | hourly :40 | store registration / aux alignment / channel health |
| pluvio-qpe-archive | every 10 min | 768-grid QPE day-zarr archive (permanent). Each day-store states its own georeference in attrs (`bounds` [w,s,e,n] outer edges, `grid_shape`, `grid_crs`, `grid_row_order`, `bounds_convention`, `bounds_source`) since 1b6f023; existing stores backfilled. Readers (scoreboard, backend Verify) treat the attr as mandatory and refuse a store without it — the archiver runs from the `/opt/pluvio/radarproc` checkout, whose `model/geo.py` resolves a different extent/bias than the repo's, so a reader-side derivation is ~60 km out at the south edge |
| pluvio-qpe-prune | daily 04:30 | prunes RAW volumes (3 d) + OPERA (7 d) — never day-zarrs |
| pluvio-wide-archive | hourly :37 | continental 3-km composite archive (permanent) |
| pluvio-forecast-archive | every 5 min | every forecast/nowcast run → storagebox (permanent, feeds Verify) |
| pluvio-scoreboard | daily 02:30 UTC, `RequiresMountsFor=/mnt/storagebox`; installed 2026-09-03 (first record 2026-09-02 written by hand, 2 min 42 s) | scores the PREVIOUS UTC day and appends `/mnt/storagebox/scoreboard/YYYY/MM/DD.json` (permanent) + rewrites `index.html`. Runs on the host, so host paths: `python -m tools.scoreboard --forecast-archive /mnt/storagebox/forecast_archive --qpe-root /mnt/storagebox/qpe --external-archive /mnt/storagebox/external_baselines --out-root /mnt/storagebox/scoreboard --html /mnt/storagebox/scoreboard/index.html` (no `--day`: it defaults to yesterday UTC). 02:30 leaves the daily QPE backfill pass and the 01:45 NAS rotation done and sits well before `pluvio-qpe-prune` at 04:30 |
| pluvio-external-baselines | every 5 min at :30 past the tick (`*:00/5:30`), RequiresMountsFor=/mnt/storagebox; live since 2026-09-03 | Buienradar point forecasts at 20 BE/NL stations → `/mnt/storagebox/external_baselines/buienradar/YYYY/MM/DD.jsonl` (permanent; verification evidence) |

## Static services (triggered by other units, not timers)

| service | trigger | purpose |
|---|---|---|
| pluvio-live-zarr | chained | live store for the hybrid producer |
| pluvio-producer-model | chained after live-zarr | c17 hybrid forecast cube → `serve/model_forecast.npz` |
| pluvio-producer | OnFailure of producer-model | classical fallback producer |
| pluvio-opera-adaptive | enabled | OPERA fill adaptation |

## Docker compose (`/opt/pluvio-backend`)

`api` (FastAPI, serves /v1/*), `worker` (bakes forecast snapshots every 5 min:
Lagrangian blend, 2-min morph, overlays/sprites), `web` (nginx, build context
`/opt/web`), `traefik`, `cache` (named volume).

## Retention classes (audited 2026-09-03)

| class | retention | where |
|---|---|---|
| raw radar volumes / dwd | 3 days (re-processing window, coverage-guarded) | storagebox |
| OPERA RATE/COMP | 7 days | storagebox |
| RAC tar cache | keep (747 daily tars, the pretrain corpus) | storagebox/knmi_rtcor |
| QPE day-zarrs, wide archive, forecast archive, external baselines, scoreboard records | forever | storagebox |
| training stores | versioned, keep last two | /opt/pluvio/zarr, /opt/pluvio/stage |

## Training node (asusprime)

`~/pluvio_v2/train_supervisor.sh` owns the training lifecycle (launch / restart /
exit on "Training done"). Control from laptops is a client role only.

## Conversion plan (1.8)

1. One `pluvio-collect@.timer`/`.service` template parameterised by feed name
   replaces the seven `pull_forward.sh` cron lines.
2. `pluvio-append-infer.timer` (*/5) replaces the append+infer cron line; the
   unit gets `After=pluvio-observed.service` ordering hints.
3. `pluvio-nas-rotate.timer` (daily 01:45).
4. Remove the crontab; `systemctl list-timers` becomes the complete schedule;
   this file is regenerated from the unit files by a small script.
