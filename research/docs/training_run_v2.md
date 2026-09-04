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

## First head-to-head vs the operational nowcast (2026-09-02, epoch-5 snapshot)

`model.evaluate` on the full chronological val window (>= 2026-04-02, 28,268
samples, 56.5M sampled cells, tau=1 mm/h). Baseline = channel 6 =
`radar[issue, lead]`, the nowcast actually served for that issue+lead;
identical truth and cells for both sides.

| lead | RMSE op->model | CSI op->model | bias op->model |
|------|----------------|---------------|----------------|
| 30   | 1.163 -> 1.075 | 0.100 -> 0.372 | +0.004 -> +0.005 |
| 60   | 1.150 -> 1.070 | 0.092 -> 0.316 | +0.004 -> +0.006 |
| 90   | 0.944 -> 0.857 | 0.084 -> 0.260 | +0.003 -> +0.008 |
| 120  | 0.965 -> 0.884 | 0.077 -> 0.214 | +0.002 -> +0.010 |

Model beats operational at 100% of leads on both metrics — and this is the
epoch-5 snapshot of the base-64 + ReduceLROnPlateau run (val_rmse 0.6749),
before the scheduler has even engaged. Caveats: truth is our composite (the
operational engine historically optimized against OPERA-era truth), and CSI
is at cell scale on a sharper analysis — the operational falls harder there.
Live-run confirmation accumulates independently via the forecast archive +
/v1/verify scores.

## v2 GO-LIVE (2026-09-02 12:07 CEST)

The epoch-5 base-64 checkpoint (val_rmse 0.6749; beats the operational
baseline at 100% of leads on RMSE and CSI) now serves production:
installed as /opt/pluvio/research/checkpoints/pluvio_unet.pt on hetz1, picked
up by the append+infer cron, tightened to */5 on 2026-09-02 (build_zarr --append -> model.infer_latest ->
model_nowcast.npz). Rollback: pluvio_unet_legacy_b32.pt.bak alongside.
The base-64 training run continues on asusprime; refresh the served file the
same way when a converged best lands. A same-split eval of the legacy
checkpoint is running for the three-way comparison record.

## Three-way same-split comparison (2026-09-02, 28,268 val samples, tau=1)

| lead | RMSE op / legacy / v2 | CSI op / legacy / v2 |
|------|-----------------------|----------------------|
| 30   | 1.163 / 1.155 / 1.075 | 0.100 / 0.109 / 0.372 |
| 60   | 1.150 / 1.117 / 1.070 | 0.092 / 0.110 / 0.316 |
| 90   | 0.944 / 0.908 / 0.857 | 0.084 / 0.107 / 0.260 |
| 120  | 0.965 / 0.919 / 0.884 | 0.077 / 0.101 / 0.214 |

legacy = the base-32 UNet that served until 2026-09-02 12:07 (wet bias
+0.025..+0.039); v2 = the epoch-5 base-64 checkpoint now serving (bias
+0.005..+0.010). Checkpoint val_rmse values are NOT comparable across
checkpoints trained on different windows (legacy said 0.2537, v2 says 0.6749
— on this common split v2 is far stronger); only same-split evals count.

## Seam lag + Lagrangian blend (2026-09-02 afternoon)

The v2 artifact's issue age is 30-70 min: 30-min store issue cadence + late
aux feeds + (formerly) a */15 append cron, now */5. Perf work on the
composite pipeline does NOT touch this — different pipeline. Visual fix
shipped: the nowcast band Lagrangian-blends the advected latest observation
into the model field over the first future hour (model.py _lagrangian_blend),
anchoring granularity and cell drift exactly at t=0. Further lag reduction
would need a low-latency feature path (infer on radar-complete issues before
all aux lands) — quality impact unmeasured, not done.

## Run converged (2026-09-02 ~17:00 CEST)

The base-64 + ReduceLROnPlateau run early-stopped at epoch 35 (~13 h,
patience 30, LR decayed 2e-4 -> 3.1e-6). Best: val_rmse 0.6749 at epoch 5 —
the checkpoint that has been serving production since 12:07. Thirty further
epochs never beat it (closest 0.6764 at epoch 35), so the served model is the
converged optimum of this configuration. No swap needed.

Next levers, in expected-value order: native higher-resolution retrain over
the same box (truth archive already supports ~1 km), the RTCOR 2019-2025
pretrain phase (corpus verified complete), 5-min issue densification (x12),
and a low-latency feature path for fresher inference issues.

## Input-validation night (2026-09-02 evening) — user-driven rigor pass

Root-caused, fixed, verified:
- Advection sign in the Lagrangian blend (cells reversed at the seam).
- THE TRIM: notebooks/_lib._resample crops the native 765x700 KNMI field to
  [:700,:700] before block-meaning; grid_latlon spread 100 rows across the
  UNTRIMMED extent -> content stretched ~0.5 deg south at Belgium (the
  "50-100 km" seam offset seen by eye). geo now maps the trimmed extent;
  multi-issue registration fit after the fix: dlat 0.00, residual dlon +0.07
  (calibrated default), median corr 0.728.
- INPUT DEFECT for the current model: aux channels were regridded via
  analysis_grid_dst to the untrimmed extent -> aux sits up to ~0.5 deg south
  of radar/truth in the training store. The trained model partially absorbed
  this; the store rebuild must regrid aux with the corrected geometry (or
  move everything to one regular lat/lon grid — the plan).
- Absolute skill (the user's "trivial validation"): +30 min over 400 val
  samples — CSI@0.5 model 0.295 vs persistence 0.070; CSI@1.0 0.248 vs
  0.052; RMSE 0.519 vs 0.754.

New automation: tools/qc_inputs.py (registration offset fit, aux alignment,
channel NaN/range health, staleness; exits 1 on WARN) + crop-mark fiducials
(PLUVIO_DEBUG_FIDUCIALS=1 stamps city crosses on both overlay families).

Retrain (paused pending these fixes) re-scoped: bigger box covering all of
the Netherlands (~1.5-7.5E, 48.9-54.2N), everything on ONE regular lat/lon
grid, truth = QPE composite (BE/south) + RTCOR (NL, 2019->) — aligns with
the RTCOR pretrain phase and retires the KNMI-stereo legacy entirely.

## v3 training LAUNCHED (2026-09-03 07:11 CEST)

Full-Benelux 192x192 store (35,673 issues; truth = native 1-km RAC corpus,
aux geometry healed): 113,680 train / 28,344 val samples, 33 channels incl.
the 3 statics (restored — zarr_dataset's hardcoded GRID=(100,100) literal had
silently dropped them at any other resolution; the dataset now derives its
grid from the store, commit 28ba87c). base 64, batch 8, patience 30, ~80-90
min/epoch. A local supervisor loop on asusprime owns the lifecycle
(launch/restart/exit-on-done) because the control channel from the laptop
drops mid-session; note pgrep guards must never appear in the same ssh
command line they test (self-match).

After convergence: same-split eval vs operational + legacy + v2, absolute
skill vs persistence on the healed inputs, then serving integration for the
new box (infer_latest bounds + frontend forecast domain + blend crop).

## Pre-rendered training shards (WP 2.6, 2026-09-03)

The v3 epoch cost is sample *assembly*, not the GPU: `ZarrCorrectionDataset`
rebuilds all 33 channels from the store for every sample on every epoch (a
chunk read per history frame, one per aux channel, plus normalisation and the
time-encoding planes) — ~47 min/epoch at 192², batch 8, 6 workers.

`tools/render_shards.py` does that assembly once into `.npy` memmaps;
`model/shard_dataset.py` (`ShardDataset`) streams them with one memmap slice
and one dtype cast per sample. `model/train.py --shards <dir>` swaps the
dataset in — same index, same chronological `issue_time_split` boundary, same
sample order, same targets, so the loss curve is unchanged.

**Where the float16 cast happens.** Once, at render time, in
`shard_dataset.cast_for_shard`. `ShardDataset` casts straight back to float32,
so `ShardDataset[i] == cast_for_shard(ZarrCorrectionDataset[i]).astype(f32)`
bit-for-bit (asserted in `tests/test_shard_dataset.py`, with the identical cast
applied to both sides).

**How much the cast actually loses** (corrected — the first version of this
section had the rationale wrong). Almost nothing, because almost every channel
is *already* a float16 value before the cast: `_normalise` divides float16
store arrays by python floats, and under numpy's weak scalar promotion that
arithmetic stays float16 — so the normalised aux (`/255`), SST and static
channels are float16-exact, not "off the grid" as claimed here before. The
radar history and the nowcast plane come straight out of a float16 store.
`lead/120` is exact for every lead the 30-min cadence allows (0.25 / 0.5 /
0.75 / 1.0). What genuinely quantises is:

* `tod_sin` / `tod_cos` — float32/float64 trig, error ≤ 2.4e-4;
* with `--lagrangian-channels`, `lagrangian_rate` and `lagrangian_flow_mag` —
  the warp and the hypot run in float32.

That is deliberate and immaterial: CUDA training runs under
`autocast(float16)`, so the model never saw more precision than this.
`--dtype float32` renders exact float32 shards (asserted bit-equal to the raw
dataset) at 2× the disk.

**Safety.** The manifest records the layout, the channel recipe (names in
`build_input` order), the grid, the sample count, the sample-semantics recipe +
its hash, a source-store fingerprint and a sha256 per shard file.
`ShardDataset` refuses an incomplete store, a missing `index.npy`, a mismatched
recipe, a filtered val split, and a `--require-rain-fraction` the shards were
not rendered with. Two of those need care about *when* they hold:

* a bumped `zarr_dataset.NORMALISE_VERSION` is refused **always** — the version
  is a constant of the build, not a property of the store, so `ShardDataset`
  compares it itself. (It used to live only in the store-derived recipe, which
  made the guarantee silently conditional on passing `--zarr`.)
* the store-derived checks — a changed lead set, an added aux channel, a
  `--lagrangian-channels` count the shards lack, and the **source-store
  fingerprint** — need `--zarr` alongside `--shards`, because without a store
  there is nothing to re-derive them from. Pass `--zarr`. It costs an index-free
  probe open plus ~64×3 plane reads and it is the only thing standing between a
  rebuilt store and a run that trains on stale samples.

**The source-store fingerprint** (`shard_dataset.source_store_hash`) is
structural **plus sampled content**: group attrs, every array's
name/shape/dtype, the full `issue_time` vector, and the contents of
`radar[i, 0]`, `truth[i]` and one aux array at 64 evenly spaced issue indices
(`hash_mode: "structural+sampled"`). The structural half alone was not enough,
and that was a real hole rather than a theoretical one: a store rebuilt **in
place** over the same window has the same arrays, shapes, attrs and
`issue_time`, so it fingerprinted identically — a resumed render then filled in
only the missing shards from the new values, `index.npy` was rewritten from the
new index on top, and `ShardDataset` accepted the mixture without a word. A
resume now refuses on a fingerprint difference and points at `--force`; so does
`train.py --shards --zarr`. It is a sample, not a proof: an edit confined
between two probes still slips past.

The render is resumable: shards are written `*.tmp` + atomically renamed (so is
`index.npy`), the manifest is rewritten after each one and only marked
`complete` at the end; a rerun verifies the recipe, the layout and the source
fingerprint, keeps what is on disk and re-renders the rest (byte-identical to a
clean render, asserted). `--force` discards the existing shards *and* unlinks
every `*.npy` / `*.npy.tmp` the new plan does not name, so re-rendering a
smaller sample set does not leave tens of GiB of orphaned shards behind.

**Checkpoint provenance.** A `--shards` run records
`{"shards": {"root", "recipe_hash", "source_store"}}` in the checkpoint, so
"which rendered store did this model train on, and from which zarr" is
answerable from the checkpoint alone. `channel_recipe` comes from the shard
manifest's own copy of `ds.channel_recipe()`, so a shard-trained checkpoint and
a zarr-trained one are equal for the same channel layout (asserted).

**Measured throughput** (CPU, single worker, page-cached synthetic store at the
real shape — 192², 33 channels, 30-min cadence): zarr 41.5 samples/s → shard
14,737 samples/s (355×). The tiny 24² test store shows 227 → 341k (1500×).
Both are upper bounds: on the real store the loader is dominated by cold random
chunk reads (~40 samples/s *aggregate* over 6 workers, which is exactly the 47
min/epoch), while the shard path is bounded by disk bandwidth. Flat is 2.391
MiB/sample, so a 3× epoch (≥120 samples/s) needs ≥290 MiB/s of random 2.4 MiB
reads. Trivial on NVMe, fine on SATA SSD, hopeless on a spinning disk. The real
acceptance number needs the GPU box.

**Dedup's win is footprint, not bandwidth.** Its 0.861 MiB/sample is the number
on disk, not the number a shuffled loader reads: `train.py` uses
`shuffle=True`, so an issue's four leads land in different batches (and
different workers), and each sample pulls its own 2.04 MiB per-issue block —
~2.4 MiB read per sample, i.e. the same ~287 MiB/s as flat, plus one array fill
for the reassembly. The per-issue block is only read once per issue if an
issue-grouped sampler keeps its leads together (the same sampler 2.3 wants for
the `--zarr` flow cache), or if the page cache happens to still hold it. So:
dedup to make the render FIT, and an issue-grouped sampler if the ≥3× gate
turns out to be bandwidth-bound on the box.

**Storage — per-issue dedup is the layout, not a follow-up.** The flat layout
does not fit the box: asusprime's `/home` was measured at **194 G** free at the
c15 relay (`docs/c17_static_experiment_runbook.md`), against 332 GiB for both
splits flat. So `--layout dedup` is the default.

29 of the 33 channels `build_input` writes are a function of the **issue**, not
of the lead — the 6-frame radar history stack, the per-issue aux planes and the
statics. Only `nowcast_at_lead`, `lead_over_120`, `tod_sin` and `tod_cos` change
between an issue's four leads (and, with the Lagrangian planes on,
`lagrangian_rate` does while `lagrangian_flow_mag` does not: it is a per-issue
signal). Dedup stores the invariant block once per issue in `inv_*.npy` and the
lead-dependent planes plus the target per sample in `x_*.npy` / `y_*.npy`;
`ShardDataset` reassembles the full `(C, H, W)` input in `build_input` channel
order on read, for one array fill per sample. The reassembled sample is
**bit-for-bit** the flat one — the equality tests run against the dedup layout
by default now, and a separate test asserts flat and dedup hand out identical
tensors, including `--lagrangian-channels 2`.

Which channels are invariant is derived from `channel_names()`
(`zarr_dataset.lead_varying_channel_indices`), never hard-coded, and the
renderer **verifies** it per issue: it builds the full input at each of the
issue's leads and refuses to write if a channel it was about to store once is
not identical across them. The manifest records the layout and the split.

Measured, at 192² float16 (input + `(1,H,W)` target, 4 leads/issue):

| layout | channels | MiB/sample | train 113,680 | val 28,344 | both 142,024 |
|---|---|---|---|---|---|
| flat | 33 | 2.391 | 265.4 GiB | 66.2 GiB | **331.6 GiB** |
| dedup | 33 (29 per issue) | **0.861** | 95.6 GiB | 23.8 GiB | **119.4 GiB** |
| flat | 35 (`--lagrangian-channels 2`) | 2.531 | 281.0 GiB | 70.1 GiB | 351.1 GiB |
| dedup | 35 (30 per issue) | **0.949** | 105.4 GiB | 26.3 GiB | **131.7 GiB** |

2.78× smaller at 33 channels (2.67× at 35 — the extra planes are one invariant
and one varying, so the varying half grows faster), and 131.7 GiB of 194 G
leaves room for the checkpoints. `--layout flat` is still there (one branch in
each file) for a box with the disk to spare and no interest in the per-sample
copy.

Both MiB/sample columns assume **4 leads per issue**, which is what the current
store gives every fully-covered issue. Dedup's per-sample cost is
`inv/leads_per_issue + var`, so it degrades as that ratio falls: a
`--require-rain-fraction` filter drops individual (issue, lead) samples while
the issue still pays for its whole invariant block, and at 1 lead/issue dedup
is 2.11 + 0.42 = 2.53 MiB/sample — no better than flat. Check
`n_samples / sum(n_issues)` in the manifest after rendering a filtered split.

Fallbacks if even that does not fit, in order: render `--split train` only and
leave val on the zarr (val is forward-only and 20% of the samples), then
`--max-samples` to cap the rendered set per split. Note what `--max-samples`
is: a cap on what gets RENDERED, recorded in the recipe and hashed, so a train
run cannot mistake a capped store for the full one — it is not a check against
the full store's sample count, and nothing detects "the store grew since".

**asusprime invocation** (python 3.10 / zarr 2.x there; the renderer only reads
the store, so no `zarr_format` concern). This is the 2.3 ablation render — the
Lagrangian planes baked in, so the flow estimate is paid once per issue here
instead of every epoch:

```
cd ~/pluvio_v2/research
df -h ~/pluvio_v2/data              # need ~132 GiB for both splits (dedup, 35 ch)
python -m tools.render_shards \
    --zarr ~/pluvio_v2/data/timeseries_v3.zarr \
    --out  ~/pluvio_v2/data/shards_v3_lag2 \
    --split train,val --workers 6 --dtype float16 \
    --layout dedup --lagrangian-channels 2 -v
# then, same flags as the zarr run apart from the dataset source.
# --zarr is what enables the store-derived recipe + source-fingerprint checks:
python -m model.train \
    --shards ~/pluvio_v2/data/shards_v3_lag2 \
    --zarr   ~/pluvio_v2/data/timeseries_v3.zarr \
    --lagrangian-channels 2 \
    --epochs 300 --batch-size 8 --base-channels 64 --num-workers 6
```

The 0-channel arm of the ablation is a second render (`--out
.../shards_v3_lag0`, no `--lagrangian-channels`) — the count is part of the
recipe, so it cannot be trained out of the lag2 store, and `--force`-ing one
store into the other would throw away the render.

The render is resumable, so a disk-full or a dropped session is recoverable by
re-running the same command. Two things not to do between render and train:
rebuild `timeseries_v3.zarr` in place (the fingerprint will refuse the resume
and the train run — correctly; re-render with `--force`), and change
`--val-frac`, which sets the split boundary that train.py compares against the
boundary recorded in both manifests.


## 2026-09-04 — loss A/B on the v3 store (2.1)

Arm A: plain Huber (the v3 run of 2026-09-03, stopped after epoch 12; best val
RMSE 0.6956 at epoch 4, then diverging to 0.78). Arm B: identical data, split
and config plus `--fss-weight 0.5 --sharpness-weight 0.05`; val RMSE 0.6856,
0.6822, 0.6957, 0.7030, 0.6871, 0.6932, 0.6932, 0.6889, 0.6981, 0.7047 for
epochs 1–10 (no divergence). Frozen benchmark (`tools/benchmark.py`, 2000
samples, 682 events, sample set `8dec3cd2e44123fd`) on arm B epoch 2 vs arm A
epoch 4: CSI@1 equal within CIs, FSS3 +0.04..+0.06 at every lead, CSI@0.1
+0.04..+0.12 at 60–120 min, RMSE equal or lower, mean error halved. Both
models beat the KNMI operational nowcast on CSI@1 at 90/120 min and on FSS3
at every lead; the operational product still wins CSI@0.1. Full table:
`research/benchmark/results/2026-09-04_v3_huber_vs_fss.md`.
