# Lightning multimodal v2 — corrected experiment runbook

Prereqs (all ✅ as of 2026-06-17): builder `.h5`/timestamp fix deployed; OPERA 2026
truth gap backfilling (wait for `opera-h5-gapfill` to finish — RATE); regularized
`train_seamless.py` (`--augment --weight-decay --cosine --patience --aux-channels`)
+ `eval_nowcast.py --aux-channels` validated on the 2060; asusprime env ready
(torch cu124, scipy/pysteps/cv2/**rasterio** — rasterio was missing and is needed by
eval via `model.geo`; installed 2026-06-17 — code synced, /home 194 G).

The smoke run (build→v2→relay→train mm 7ch + ro 6ch→eval) passed end-to-end, so
this is the same with the full window + more epochs.

## 0. Gate
Proceed only when `systemctl is-active opera-h5-gapfill` is `inactive` (done) and
`find /mnt/storagebox/opera/RATE/2026/{01..06} -name '*.tiff' | wc -l` is full
(~96/day per month). GII fill + cloud + ACRR + radklim can still be running.

## 1. Build on hetz1 (full LI annual cycle, complete truth)
```
docker run --rm -e PLUVIO_GRID_N=256 -v /mnt/storagebox:/mnt/storagebox -v /opt/pluvio/stage:/stage \
  -v /opt/pluvio/stage/code/model:/app/model -v /opt/pluvio/stage/code/tools:/app/tools \
  --entrypoint python seamless-builder:latest /app/tools/build_seamless_zarr.py \
  --out /stage/nowcast_mm_full.zarr --storage /mnt/storagebox --no-aifs \
  --cadence-min 15 --leads 0,10,20,30,40,50,60,70,80,90,100,110,120 \
  --aux-vars li_flash --start 2025-06-01 --end 2026-06-18
# then v2 convert:
docker run --rm -v /opt/pluvio/stage:/stage -v /opt/pluvio/stage/code/tools:/app/tools \
  --entrypoint python seamless-builder:latest /app/tools/zarr_v3_to_v2.py \
  /stage/nowcast_mm_full.zarr /stage/nowcast_mm_full_v2.zarr
```
VERIFY before training: probe `li_flash` finite-issue count > 0 AND `opera_rate`
finite across 2025-06→2026-06 incl. the formerly-missing 2026-01→05 months.

## 2. Relay hetz1 → local → asusprime
`tar` on hetz1 → `ansible fetch` → `scp` to asusprime → untar to
`/home/jeroentrappers/pluvio/nowcast_mm_full_v2.zarr`. (Bigger than smoke — full
window ~24k issues × 256² × 2ch ≈ a few GB; fine.)

## 3. Train on asusprime 2060 (regularized) — mm + ro, identical except aux
```
cd /home/jeroentrappers/pluvio
# multimodal (7ch, lightning):
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.train_seamless --zarr nowcast_mm_full_v2.zarr \
  --history-steps 6 --epochs 30 --batch-size 12 --workers 4 \
  --augment --weight-decay 1e-4 --cosine --patience 4 --out checkpoints/nowcast_mm_v2.pt
# radar-only (6ch, SAME zarr, lightning channel ablated):
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.train_seamless --zarr nowcast_mm_full_v2.zarr \
  --history-steps 6 --epochs 30 --batch-size 12 --workers 4 \
  --augment --weight-decay 1e-4 --cosine --patience 4 --aux-channels none --out checkpoints/nowcast_ro_v2.pt
```
Note 2060 = 6 GB: batch 12 should fit (smoke ran 8 comfortably); drop to 8 on OOM.
At cadence-15 full-window this is a multi-hour/overnight run — patience-4 early-stop
manages it (the unregularized v1 bottomed at epoch 3; regularization should push the
optimum later). If epoch time is impractical, rebuild at `--cadence-min 30`
(halves samples; history becomes 3 h) and note the change.

## 4. Eval both vs pysteps (asusprime, GPU + pysteps)
```
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.eval_nowcast --zarr nowcast_mm_full_v2.zarr \
  --ckpt checkpoints/nowcast_mm_v2.pt --samples 4000 > ~/eval_mm_v2.log 2>&1
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.eval_nowcast --zarr nowcast_mm_full_v2.zarr \
  --ckpt checkpoints/nowcast_ro_v2.pt --aux-channels none --samples 4000 > ~/eval_ro_v2.log 2>&1
```

## 5. Delta + write-up
Per-lead mm−ro on CSI@1 (heavy), CSI@0.1, MAE (same parser as
`gpu_results/mm_ro_20260617/DELTA.md`). This is the FIRST real test of the lightning
hypothesis (v1 was confounded by the dead channel). Update `paper_draft.md §4.6` +
memory `seamless-model` with the verified verdict. Caveats now removed: real
lightning, all-season, regularized. Remaining: lightning-only (GII fast-follow once
its backfill completes), 256² grid, 2060-scale training.
```
```
