#!/bin/bash
# c16 nowcast — stacked improvements over c15_0724 (advection prior + growth/decay
# tendency + multi-scale FSS loss), nowcast leads 0-120 min only. Three runs:
#   full  = advection + tendency + FSS loss   (the deliverable)
#   noadv = tendency + FSS loss, no advection  (ablation: is advection the FSS driver?)
#   nofss = advection + tendency, per-pixel loss (ablation: does the FSS term sharpen?)
# Baseline/control is the existing c15_0724 MM run — not retrained here.
set -uo pipefail
cd /home/jeroentrappers/pluvio
export PLUVIO_GRID_N=256
PY=/home/pv/bin/python
Z=nowcast_mm_c15_0724_v2.zarr
LEADS=0,10,20,30,40,50,60,70,80,90,100,110,120
C="--zarr $Z --history-steps 6 --leads $LEADS --batch-size 16 --workers 4 --augment \
   --weight-decay 1e-4 --cosine --patience 4 --epochs 25 --aux-channels li_flash"
EVAL="--zarr $Z --aux-channels li_flash --samples 4000"

# Step 0: ensure the advection/tendency channels exist in the zarr (idempotent guard).
if ! $PY -c "import zarr,sys; k=set(zarr.open_group('$Z','r').array_keys()); sys.exit(0 if {'oflow_rate','rate_tendency'}<=k else 1)"; then
  echo "=== $(date -u) PRECOMPUTE advection+tendency channels ==="
  $PY -m tools.add_nowcast_channels --zarr $Z
fi

echo "=== $(date -u) TRAIN c16 FULL (advection + tendency + FSS loss) ==="
$PY -m model.train_seamless $C --advection --fss-weight 0.3 --out checkpoints/nowcast_mm_c16_full.pt
echo "=== $(date -u) TRAIN c16 NO-ADV (tendency + FSS loss) ==="
$PY -m model.train_seamless $C --fss-weight 0.3 --out checkpoints/nowcast_mm_c16_noadv.pt
echo "=== $(date -u) TRAIN c16 NO-FSS (advection + tendency, per-pixel loss) ==="
$PY -m model.train_seamless $C --advection --out checkpoints/nowcast_mm_c16_nofss.pt

echo "=== $(date -u) EVAL FULL ==="
$PY -m model.eval_nowcast $EVAL --advection --ckpt checkpoints/nowcast_mm_c16_full.pt \
    > /home/jeroentrappers/eval_c16_full.log 2>&1
echo "=== $(date -u) EVAL NO-ADV ==="
$PY -m model.eval_nowcast $EVAL --ckpt checkpoints/nowcast_mm_c16_noadv.pt \
    > /home/jeroentrappers/eval_c16_noadv.log 2>&1
echo "=== $(date -u) EVAL NO-FSS ==="
$PY -m model.eval_nowcast $EVAL --advection --ckpt checkpoints/nowcast_mm_c16_nofss.pt \
    > /home/jeroentrappers/eval_c16_nofss.log 2>&1
echo "=== $(date -u) ALL DONE ==="
