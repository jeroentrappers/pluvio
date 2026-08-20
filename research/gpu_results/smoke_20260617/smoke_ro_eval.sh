#!/bin/bash
cd /home/jeroentrappers/pluvio
# wait for mm train to finish
while pgrep -f "train_seamless.*smoke_mm" >/dev/null; do sleep 5; done
echo "=== mm done; final mm log ==="; tail -2 /home/jeroentrappers/smoke_mm.log
echo "=== ro smoke-train (--aux-channels none -> 6ch) ==="
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.train_seamless --zarr nowcast_smoke_v2.zarr \
  --history-steps 6 --epochs 3 --batch-size 8 --workers 2 --augment --weight-decay 1e-4 \
  --cosine --patience 2 --aux-channels none --out checkpoints/smoke_ro.pt 2>&1 | grep -E "channels|optim|epoch|done" | tail -6
echo "=== smoke-eval mm (7ch) ==="
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.eval_nowcast --zarr nowcast_smoke_v2.zarr \
  --ckpt checkpoints/smoke_mm.pt --samples 150 2>&1 | grep -iE "deliverable|engine|pysteps|HEADLINE|^ *[0-9]+ " | head -8
echo "=== smoke-eval ro (--aux-channels none, 6ch) ==="
PLUVIO_GRID_N=256 /home/pv/bin/python -m model.eval_nowcast --zarr nowcast_smoke_v2.zarr \
  --ckpt checkpoints/smoke_ro.pt --aux-channels none --samples 150 2>&1 | grep -iE "deliverable|val=|HEADLINE" | head -4
echo "=== SMOKE PIPELINE OK ==="
