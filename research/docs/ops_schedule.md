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

## systemd timers (the complete schedule since 2026-09-04 — see research/ops/schedule.yaml; cron is empty)

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
| pluvio-buienradar-eu | every 5 min at :45 past the tick (`*:00/5:45`), RequiresMountsFor=/mnt/storagebox; live since 2026-09-03 21:50 UTC+2 | Buienradar EU radar composite + every forecast run → `/mnt/storagebox/buienradar_eu` (permanent; verification evidence). Invocation: `python -m tools.buienradar_eu collect --root /mnt/storagebox/buienradar_eu`, with the unit carrying `Environment=PLUVIO_BUIENRADAR_EU_INDEX=/opt/pluvio/state/buienradar_eu/index.sqlite` — the sqlite frame index **must** live on local disk, because the archive root is a CIFS mount where SQLite cannot take file locks (without the override the collector dies with "database is locked" on an empty index). Anything reading the index (`verify`) needs the same env var or `--index`. The 5-min cadence is a hard requirement, not a preference: their composite history is only 12 frames × 15 min deep (`history` is capped at 12) and a forecast run's earliest lead times drop out of the metadata as the run ages, so anything slower loses frames permanently. First ticks: 41 frames archived, then 0 downloaded / 41 skipped. Measured frame sizes 38.9 KB (composite) and 42.7 KB (forecast, wet scene) → **18–46 GB/yr** at 96 runs/day × 30 frames, forecast frames dominating (30 per run against 1–2 new composite frames). Retention forever for now; revisit once a year of depth is in sight. Exits 0 on partial success; non-zero only when a metadata document could not be fetched, so an alert on this unit means the endpoint or the API key moved. |

## Static services (triggered by other units, not timers)

Note (2026-09-04): the `pluvio-producer-model` image is built from
`/opt/pluvio/build-model` (a June snapshot of research/model + tools, NOT the
research checkout): deploying producer code means copying the changed files
there and `docker build -t pluvio-producer-model:latest .`; the unit then runs
the new image on its next chained start.

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
| QPE day-zarrs, wide archive, forecast archive, external baselines, scoreboard records, buienradar_eu | forever | storagebox |
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

## Incident 2026-09-06 — storage box unreachable

09:47:52 local the Hetzner storage box stopped answering (`CIFS: VFS: … has
not responded in 180 seconds`); port 445 dead over both IPv4 and IPv6, no
ICMP. CIFS **hung** rather than failing, so:

* every `produce_observed` pool worker blocked in
  `cifs_wait_for_server_reconnect`; the unit was killed on its 900 s
  `TimeoutStartSec` each run, its effective cadence stretched 5 → 15 min and
  then to nothing. The served composite (`serve/observed.npz`) froze at 09:56
  and nothing alerted for ~40 min.
* `qpe-archive`, `wide-archive`, `forecast-archive`, `external-baselines`,
  `buienradar-eu`, `qc`, `qc-inputs`, `qpe-prune` all failed on the mount.
* the nowcast/forecast serving path was unaffected (local `/opt/pluvio/serve`,
  `/opt/pluvio/zarr`): the site stayed up, `append-infer` kept running.

A `umount -l` to force a remount turned `/mnt/storagebox` into a plain local
directory and collectors wrote **2.3 GB onto the root disk** within minutes
(moved aside to `/opt/pluvio/mount-shadow-<stamp>/`, to be merged back or
discarded once the box returns — the collectors re-fetch what is missing).

Actions taken: box-writing timers stopped (`pluvio-observed`, the archives,
and the `aifs/dwd/era5/icon-d2/rtcor/mtg` forwarders), mountpoint emptied and
`chattr +i`-ed, `pluvio-storagebox-watchdog` deployed (2-min I/O probe,
auto-remount, auto re-arm, verdict into the QC report).

Hetzner confirmed it at 08:09 UTC ("Storage Box host FSN1-BX487 not
accessible", investigating); our first CIFS stall was 07:47 UTC, ~20 min
earlier.

### Local capture during the outage (2026-09-06)

Rather than lose every raw feed for the duration, the seven box-bound
collectors write to `/opt/pluvio/outage-capture/<source>/` through systemd
drop-ins (`/etc/systemd/system/<unit>.service.d/outage-local.conf`):

| unit | normally writes | during the outage |
|---|---|---|
| be-radar-volumes | /mnt/storagebox/be_radar | /opt/pluvio/outage-capture/be_radar |
| dwd-sweeps | /mnt/storagebox/dwd_vol | …/dwd_vol |
| knmi-rtcor-forward | /mnt/storagebox/knmi | …/knmi |
| aifs-forward, era5-forward, icon-d2-forward, mtg-l2-forward | /mnt/storagebox/{aifs,era5,icon_d2,mtg_l2} | …/{aifs,era5,icon_d2,mtg_l2} |

The three shell collectors take their target from an env var
(`BE_RADAR_OUT`, `DWD_OUT`, `KNMI_RTCOR_OUT`) and each refuses to run below a
50 GB free-space floor, which now applies to the root filesystem — that is
the guard against filling `/`. The four docker collectors have their `-v`
bind rewritten in the drop-in.

**Merge-back, when the box returns** (the watchdog remounts and re-arms
timers by itself; this part is deliberately manual):

```
systemctl stop be-radar-volumes.timer dwd-sweeps.timer knmi-rtcor-forward.timer \
                aifs-forward.timer era5-forward.timer icon-d2-forward.timer mtg-l2-forward.timer
for d in be_radar dwd_vol knmi aifs era5 icon_d2 mtg_l2; do
  rsync -a --remove-source-files /opt/pluvio/outage-capture/$d/ /mnt/storagebox/$d/
done
rsync -a --remove-source-files /opt/pluvio/mount-shadow-<stamp>/ /mnt/storagebox/   # the umount -l shadow
rm -f /etc/systemd/system/*.service.d/outage-local.conf && systemctl daemon-reload
systemctl start …the timers again…
```

What is NOT recoverable this way: everything that already lived only on the
box (QPE day-zarrs, forecast archive, wide archive, external baselines,
buienradar_eu, RAC corpus) — those come back with the box, or not at all. A
second Storage Box would take new writes and give somewhere to rsync the
local capture, but it does not restore that history.

## Storage-box SSH (fast maintenance, 2026-09-06)

The Storage Box also speaks SSH on **port 23** with the sub-account
(`u614373-sub1`); SSH must be ticked for that sub-account in Hetzner Robot
(it was off until 2026-09-06, which is why the first key install failed with
an empty auth-method list). hetz1's key lives at
`/root/.ssh/storagebox_ed25519`:

```
ssh -p 23 -i /root/.ssh/storagebox_ed25519 u614373-sub1@u614373-sub1.your-storagebox.de "ls radar_volumes/2026/09"
```

It is a **restricted shell** (its own `help` lists the commands: ls/tree/cd,
mkdir, rm, mv, cp, du, df, chmod, quota…), one command per invocation, no
pipes. That is enough for the operations that hurt over CIFS, where every
file costs a network round trip:

| operation | over the CIFS mount | over SSH |
|---|---|---|
| `du -sh dwd_vol` (59 GB, ~50k files) | timed out at 90 s | seconds |
| deleting a pruned raw-volume day | tens of minutes | seconds |
| merging 27 GB of small files back | ~30 min with 12 parallel rsyncs | n/a (use rsync over SSH) |

Use it for bulk deletes and size audits; keep the CIFS mount for the jobs
that read and write data continuously.
