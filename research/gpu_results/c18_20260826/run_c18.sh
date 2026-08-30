#!/bin/bash
# c18 — follow-ups to the c17-C gate pass. Two single-variable arms off c17-C
# (12ch, advection + tendency + li_flash + 3 static, exceedance loss, weight 0.3):
#
#   D = C without the static channels (9ch)   → do terrain channels matter NOW?
#   E = C with --fss-weight 1.0                → is 0.3 (inherited from c16) too low?
#
# Why D is not a repeat of c17-A: A tested static under the RATE-pooled loss, where
# blurring cost almost nothing (measured: a blurred field cost 0.071x a 4-px
# displacement). Under that objective a placement signal from terrain could be
# masked entirely. D tests static against the objective that actually rewards
# correct placement, so a null here is a far stronger negative than A's.
#
# ~25-35 h per arm (c17 arms ran 8-19 epochs at ~163 min). Independent — safe to
# stop after either. c17-C is the control and is NOT retrained.
set -uo pipefail
cd /home/jeroentrappers/pluvio
export PLUVIO_GRID_N=256
PY=/home/pv/bin/python
Z=nowcast_mm_c17_v2.zarr
LEADS=0,10,20,30,40,50,60,70,80,90,100,110,120
C="--zarr $Z --history-steps 6 --leads $LEADS --batch-size 16 --workers 4 --augment \
   --weight-decay 1e-4 --cosine --patience 4 --epochs 25 --aux-channels li_flash \
   --advection --fss-mode exceedance"
EVAL="--zarr $Z --samples 4000"   # channel recipe read from the checkpoint

echo "=== $(date -u) TRAIN c18-D (no static, 9ch, exceedance w=0.3) ==="
$PY -m model.train_seamless $C --no-static --fss-weight 0.3 \
    --out checkpoints/nowcast_mm_c18_d_nostatic.pt
echo "=== $(date -u) EVAL c18-D ==="
$PY -m model.eval_nowcast $EVAL --ckpt checkpoints/nowcast_mm_c18_d_nostatic.pt \
    > /home/jeroentrappers/eval_c18_d_nostatic.log 2>&1

echo "=== $(date -u) TRAIN c18-E (static, 12ch, exceedance w=1.0) ==="
$PY -m model.train_seamless $C --fss-weight 1.0 \
    --out checkpoints/nowcast_mm_c18_e_w1.pt
echo "=== $(date -u) EVAL c18-E ==="
$PY -m model.eval_nowcast $EVAL --ckpt checkpoints/nowcast_mm_c18_e_w1.pt \
    > /home/jeroentrappers/eval_c18_e_w1.log 2>&1

echo "=== $(date -u) ALL DONE ==="
