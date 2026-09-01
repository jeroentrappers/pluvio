# Training run v2 — nowcast head retrained on OUR composite

## What changed since v1 (c17-C)

v1 trained against OPERA RATE as truth. We have since measured OPERA to be the
weakest reference on the board (beaten/tied in every gauge-evaluated cell) and we
now produce a composite that ties RADOLAN in stratiform and leads RTCOR on trace
detection. Training against OPERA caps the model at OPERA's ceiling; v2 trains
against our own product and reserves gauges strictly for evaluation.

## Truth: two-stage curriculum (the archive is 3 days old)

| stage | target | period | why |
|---|---|---|---|
| **Pretrain** | KNMI RTCOR 5-min (open tars, 2019→, gauge-adjusted, 1 km) | ≥2 full years | volume + seasons + a consistent high-quality national product over the NL core |
| **Fine-tune** | OUR composite from `/mnt/storagebox/qpe/YYYY/MM/DD.zarr` (rate f16 + quality + n_radars, 768 research grid) | all archived days, growing ~1 day/day | the product we actually serve; quality band becomes the loss weight, n_radars a confidence channel |

Rules: the fine-tune target is the RADAR-ONLY chain (no gauge adjustment baked
in) so gauge-based evaluation stays out-of-sample. Frames with day-coverage
<90% or n_radars below the day's mode are masked from the loss, not imputed.

## Grid

`PLUVIO_GRID_N=256` (~3 km over the research box) — the refinement the seamless
plan already commits to. NOT the 1.5-km serving grid (DGMR-scale compute) and NOT
the 100×100 legacy grid (convective structure averaged away; documented caveat).
Verification always reports grid resolution + scale-aware FSS.

## Model & loss (extends train_seamless.py, no new framework)

- Same encoder/UNet family, `--history-steps 6` (30 min), leads 5…120 min.
- `--fss-mode exceedance --fss-weight` swept: the exceedance-pooled multi-scale
  term is the repo's own finding for wet-mask structure; rate-pooled measured flat.
- Advection baseline channel ON (`--advection`): the model learns residuals over
  the same motion field the display morph uses.
- Aux channels: start radar-only + time encodings; add MTG/lightning/AWS stacks
  as ablations ONLY (c17-A taught us static terrain was a negative result —
  every channel must buy its way in on held-out skill).
- Quality-weighted loss from the archive's quality band.

## Splits — chronological, no exceptions

Pretrain: train ≤2024-12, val 2025-H1, test 2025-H2 (RTCOR period).
Fine-tune: leave-out the two benchmark days we have gauge-scored to death
(2026-08-31 convective, 2026-09-01 stratiform) + every 7th archived day as the
rolling test set. Random splits are banned (autocorrelation leak, documented).

## Baselines & benchmarks (the match-or-beat ladder)

1. persistence, 2. pySTEPS-class advection (our flow), 3. production c17-C,
4. RTCOR-as-forecast (their field frozen/advected), 5. **Met Office UKV via
DataHub over the British Isles** (the external yardstick with no home-field
bias), 6. INCA-class public range (0.45–0.55 CSI @ +30 min) as the sanity band.
Ultimate arbiter: KNMI/KMI/DWD/EA gauges via `tools/regional_eval.py` windows.

## Compute plan (asusprime GPU)

- Shard builder on hetz1 (RTCOR tars + qpe zarr → training shards on the box),
  rsync to asusprime; **runs go to full convergence — patience/epochs are never
  trimmed for wall-clock** (standing rule).
- Pretrain ~2 weeks of GPU-hours at 256 grid; fine-tune cheap (days of data).
- Every run logs: config, git rev of the chain that made its truth, grid, and
  the gauge-eval table of its truth source (so "beat the teacher?" is answerable).

## Success gates (all on held-out, all vs gauges where gauges exist)

1. Beat production c17-C on CSI/FSS at every lead 5–120 min.
2. Beat advection by +30 min at ≥1 mm/h (the operational nowcast's known hole).
3. h0 sanity: model-analysis ties the composite's own gauge scores (no drift).
4. Report vs UKV/INCA band; regression on any gate blocks deployment, same
   discipline as the serving pipeline.

## Immediate next actions

1. Shard builder script (RTCOR reader exists: `tools/knmi_rtcor.py`).
2. ~~Backfill check~~ DONE 2026-09-01: 7/7 quarterly spot-checks 2019–2025 return 200 on `nl_rdr_data_rtcor_5m_tar` — the pretrain corpus is complete.
3. Extend `dataset.py` with the qpe-zarr target loader (quality → loss weight).
4. First pretrain launch on asusprime once shards land.

## Launch log (2026-09-01 evening)

- Store discovery: `timeseries.zarr` already spans **2024-08-14 → now, 35,672
  issues** with the full aux stack — no RAC_FM backfill needed; only truth values.
- Truth backfill running as **4 day-shards in parallel** (8.9k issues each,
  newest-first, resumable; disjoint tars → disjoint zarr chunks). Serial estimate
  was days; sharded ≈ overnight. GPU is irrelevant here — the job is
  network+decode bound; production compositing likewise stays CPU-on-hetz1
  (2:03–2:21 ticks, in budget; WAN-shipping volumes to a home GPU would trade
  reliability for nothing).
- asusprime: RTX 2060 6GB, torch env installing, code synced to ~/pluvio_v2,
  smoke launcher staged. AMP confirmed in train_seamless (autocast + GradScaler);
  6 GB implies batch ~4-8 at the 256 grid with accumulation if needed.

## Trainer/dataset pairing (measured 2026-09-01, smoke bring-up)

Two dataset stacks coexist and are NOT interchangeable:

- `model/zarr_dataset.py` (`ZarrCorrectionDataset`) + `model/train.py` — pairs
  with OUR stores from `tools/build_zarr.py` (arrays `radar`, `truth`,
  `issue_time`; truth excluded from aux = leak protection). This is the v2
  smoke/fine-tune path.
- `model/seamless_dataset.py` + `model/train_seamless.py` — pairs with the
  c15-era stores from `build_seamless_zarr.py` (arrays `opera_rate`,
  `oflow_rate`/`oflow_leads`/`rate_tendency` via `tools/add_nowcast_channels.py`).
  Running it against a v2 store fails: KeyError `opera_rate`.

Follow-ups before the fine-tune phase, if we want advection priors on the v2
store: port `tools/add_nowcast_channels.py` to read `radar` instead of
`opera_rate` (name is hardcoded), or add the advection channels in
`build_zarr.py` itself.

GPU node constraint: asusprime runs python 3.10 → zarr 2.x only. Stores
shipped there must be written `zarr_format=2` (hetz1's radarproc venv is
zarr 3; pass `zarr_format=2` at group creation). Transfer as a single tar —
per-issue chunks make ~10^5 files and per-file rsync crawls.

## Smoke result (2026-09-01 ~23:20 UTC+2)

PASSED on asusprime with `model.train` + the truth store: 1028 train / 258 val
samples (33 channels, chronological split), 2 epochs in 0.7 min, AMP on the
RTX 2060, checkpoint written, val_rmse 0.2388. Pipeline bugs found on the way
(all fixed): the staged launcher pointed at train_seamless (wrong store
format); the live store grows during a slice read so per-issue arrays must be
sliced with a length tolerance and the export length-aligned to
min(per-issue lengths) — shipping unaligned arrays gave lookup indices past
the sliced truth.

Full run: entire aligned store (~35.7k issues, 14 GB zarr-v2, direct
hetz1→asusprime rsync via authorized key), `model.train --epochs 300
--batch-size 32` — no wall-clock cap, early stopping on the val-RMSE plateau
decides convergence.
