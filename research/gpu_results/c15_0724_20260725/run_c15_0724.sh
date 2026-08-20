#!/bin/bash
set -uo pipefail
cd /home/jeroentrappers/pluvio
export PLUVIO_GRID_N=256
PY=/home/pv/bin/python
Z=nowcast_mm_c15_0724_v2.zarr
C="--zarr $Z --history-steps 6 --epochs 25 --batch-size 16 --workers 4 --augment --weight-decay 1e-4 --cosine --patience 4"
echo "=== $(date -u) TRAIN MM c15_0724 (7ch, +lightning) ==="
$PY -m model.train_seamless $C --out checkpoints/nowcast_mm_c15_0724.pt
echo "=== $(date -u) TRAIN RO c15_0724 (6ch, radar-only) ==="
$PY -m model.train_seamless $C --aux-channels none --out checkpoints/nowcast_ro_c15_0724.pt
echo "=== $(date -u) EVAL MM ==="
$PY -m model.eval_nowcast --zarr $Z --ckpt checkpoints/nowcast_mm_c15_0724.pt --samples 4000 > /home/jeroentrappers/eval_mm_c15_0724.log 2>&1
echo "=== $(date -u) EVAL RO ==="
$PY -m model.eval_nowcast --zarr $Z --ckpt checkpoints/nowcast_ro_c15_0724.pt --aux-channels none --samples 4000 > /home/jeroentrappers/eval_ro_c15_0724.log 2>&1
echo "=== $(date -u) ALL DONE ==="
