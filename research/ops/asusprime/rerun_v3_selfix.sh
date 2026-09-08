#!/bin/bash
# Both v3 arms, selection on the objective (val_loss), sequential on one GPU.
# Resumable: the trainer writes checkpoints/<name>.state.pt every epoch and on
# SIGTERM, and --resume auto picks that up, so `pluvio-train pause` before a
# reboot costs at most the unfinished epoch. An arm whose log already says
# "Training done" is skipped rather than restarted.
cd /home/jeroentrappers/pluvio_v2 || exit 1
export PYTHONPATH=/home/jeroentrappers/pluvio_v2
for arm in lagr fss; do
  log=full_train_v3_${arm}_sel.log
  if grep -q "Training done" "$log" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) $arm already finished — skipping" >> rerun_v3_selfix.log
    continue
  fi
  if [ -f TRAIN_PAUSED ]; then
    echo "$(date -u +%FT%TZ) paused marker present — not starting $arm" >> rerun_v3_selfix.log
    exit 0
  fi
  if [ "$arm" = "lagr" ]; then
    extra=(--shards /home/jeroentrappers/pluvio_v2/data/shards_v3 --lagrangian-channels 2)
  else
    extra=()
  fi
  echo "$(date -u +%FT%TZ) launching $arm" >> rerun_v3_selfix.log
  venv/bin/python -m model.train \
    --zarr /home/jeroentrappers/pluvio_v2/data/timeseries_v3.zarr \
    "${extra[@]}" \
    --epochs 300 --batch-size 8 --num-workers 6 --patience 30 \
    --base-channels 64 --fss-weight 0.5 --sharpness-weight 0.05 \
    --select-on val_loss --resume auto \
    --checkpoint checkpoints/v3_192_${arm}_sel.pt -v >> "$log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $arm exited rc=$rc ($(grep -c Epoch "$log") epochs)" >> rerun_v3_selfix.log
  # A pause (clean stop, no "Training done") must not fall through to the next arm.
  if [ -f TRAIN_PAUSED ] || ! grep -q "Training done" "$log"; then
    echo "$(date -u +%FT%TZ) $arm not finished — stopping the queue here" >> rerun_v3_selfix.log
    exit 0
  fi
done
echo "$(date -u +%FT%TZ) queue done" >> rerun_v3_selfix.log
