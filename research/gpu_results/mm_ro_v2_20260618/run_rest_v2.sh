#!/bin/bash
cd /home/jeroentrappers/pluvio
export PLUVIO_GRID_N=256
PY=/home/pv/bin/python
C="--zarr nowcast_mm_full_v2.zarr --history-steps 6 --epochs 20 --batch-size 8 --workers 4 --augment --weight-decay 1e-4 --cosine --patience 4"
# wait for the running mm training to finish
while pgrep -f "train_seamless.*nowcast_mm_v2.pt" >/dev/null; do sleep 60; done
if [ ! -f checkpoints/nowcast_mm_v2.pt ]; then echo "=== MM FAILED (no checkpoint) ==="; exit 1; fi
echo "=== $(date -u) MM DONE ==="; grep "done. best" /home/jeroentrappers/exp_v2.log | tail -1
echo "=== $(date -u) TRAIN RO (6ch, radar-only) ==="
$PY -m model.train_seamless $C --aux-channels none --out checkpoints/nowcast_ro_v2.pt
echo "=== $(date -u) EVAL MM ==="
$PY -m model.eval_nowcast --zarr nowcast_mm_full_v2.zarr --ckpt checkpoints/nowcast_mm_v2.pt --samples 4000 > /home/jeroentrappers/eval_mm_v2.log 2>&1
echo "=== $(date -u) EVAL RO ==="
$PY -m model.eval_nowcast --zarr nowcast_mm_full_v2.zarr --ckpt checkpoints/nowcast_ro_v2.pt --aux-channels none --samples 4000 > /home/jeroentrappers/eval_ro_v2.log 2>&1
echo "=== $(date -u) ALL DONE ==="
