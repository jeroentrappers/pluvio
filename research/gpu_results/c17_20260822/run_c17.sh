#!/bin/bash
# c17 nowcast — static terrain channels + the exceedance-pooled structure loss.
#
# Chain of SINGLE-variable arms, so every delta is attributable:
#   A  = c16_full + 3 static channels          → do terrain channels help?
#   C  = A, loss pools wet-mask not intensity  → does that close FSS@9/15km?
#   B  = C without the advection prior         → does dropping advection lift it further?
#
# Baseline/control is the existing c16_full run (NOT retrained). Note c16 used a
# 14-month zarr; c17 uses 24 months (2024-08-14 → 2026-08-19, 68,361 issues), so
# A-vs-c16_full mixes "static channels" with "more data". The clean static-only
# read is A vs a c16_full re-eval on THIS zarr with --no-static (cheap, ~15 min) —
# step 0 below does exactly that.
#
# Channels: 6 history + li_flash + oflow_rate + rate_tendency + 3 static = 12
#           (B drops oflow+tendency → 10)
#
# ~31 h per arm at this window size (c16 measured 95 min/epoch on 39k issues;
# 68k issues → ~165 min/epoch, patience-4 typically stopping ~epoch 12-17).
# Each arm is independent — safe to stop after any of them.
set -uo pipefail
cd /home/jeroentrappers/pluvio
export PLUVIO_GRID_N=256
PY=/home/pv/bin/python
Z=nowcast_mm_c17_v2.zarr
LEADS=0,10,20,30,40,50,60,70,80,90,100,110,120
C="--zarr $Z --history-steps 6 --leads $LEADS --batch-size 16 --workers 4 --augment \
   --weight-decay 1e-4 --cosine --patience 4 --epochs 25 --aux-channels li_flash"
EVAL="--zarr $Z --samples 4000"     # channel recipe now read from the checkpoint

# Step 0: derived channels must exist (advection prior + growth/decay tendency).
# Idempotent guard — skips if already present.
if ! $PY -c "import zarr,sys; k=set(zarr.open_group('$Z','r').array_keys()); sys.exit(0 if {'oflow_rate','rate_tendency'} <= k else 1)"; then
  echo "=== $(date -u) PRECOMPUTE advection+tendency (~65 min at 68k issues) ==="
  $PY -m tools.add_nowcast_channels --zarr $Z
fi

# Step 0b: re-evaluate the c16 champion on THIS zarr with static disabled, so the
# c17 numbers are comparable on the same validation window. Without this, every
# c17-vs-c16 delta silently includes the 10 extra months of data.
# --advection is passed explicitly here: c16_full predates the checkpoint-recipe
# change, so it carries no `advection` key and the eval would default it to False,
# assemble 7 channels against a 9-channel net, and (correctly) refuse to run.
echo "=== $(date -u) RE-EVAL c16_full on the c17 zarr (--no-static --advection, 9ch) ==="
$PY -m model.eval_nowcast $EVAL --no-static --advection \
    --aux-channels li_flash --ckpt checkpoints/nowcast_mm_c16_full.pt \
    > /home/jeroentrappers/eval_c17_c16base.log 2>&1

echo "=== $(date -u) TRAIN c17-A (static + advection + rate loss) ==="
$PY -m model.train_seamless $C --advection --fss-weight 0.3 --fss-mode rate \
    --out checkpoints/nowcast_mm_c17_a_static.pt
echo "=== $(date -u) EVAL c17-A ==="
$PY -m model.eval_nowcast $EVAL --ckpt checkpoints/nowcast_mm_c17_a_static.pt \
    > /home/jeroentrappers/eval_c17_a_static.log 2>&1

echo "=== $(date -u) TRAIN c17-C (static + advection + EXCEEDANCE loss) ==="
$PY -m model.train_seamless $C --advection --fss-weight 0.3 --fss-mode exceedance \
    --out checkpoints/nowcast_mm_c17_c_exceed.pt
echo "=== $(date -u) EVAL c17-C ==="
$PY -m model.eval_nowcast $EVAL --ckpt checkpoints/nowcast_mm_c17_c_exceed.pt \
    > /home/jeroentrappers/eval_c17_c_exceed.log 2>&1

echo "=== $(date -u) TRAIN c17-B (static + EXCEEDANCE loss, NO advection) ==="
$PY -m model.train_seamless $C --fss-weight 0.3 --fss-mode exceedance \
    --out checkpoints/nowcast_mm_c17_b_noadv.pt
echo "=== $(date -u) EVAL c17-B ==="
$PY -m model.eval_nowcast $EVAL --ckpt checkpoints/nowcast_mm_c17_b_noadv.pt \
    > /home/jeroentrappers/eval_c17_b_noadv.log 2>&1

echo "=== $(date -u) ALL DONE ==="
