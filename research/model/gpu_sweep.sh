#!/usr/bin/env bash
# On-GPU tuning sweep. Trains several configs on the staged store, evaluates each
# head-to-head vs the operational nowcast, and writes per-config checkpoints +
# eval reports + a summary. Run from /workspace (must contain timeseries.zarr,
# model/, notebooks/_lib.py, and this script). Outputs land in checkpoints/ so
# the orchestrator can scp them back.
#
# Grid rationale (from run 1: base-32 overfit after epoch 2, val_rmse 0.2537):
#   - bigger models (base 48/64 ≈ the ~1M-param design),
#   - more data (lower/zero rain-fraction filter — run 1 dropped 52k samples),
#   - gentler lr (1e-4) to curb the fast overfit,
#   - a bias-penalty variant (CSI vs bias trade-off).
set -uo pipefail
cd /workspace
export PYTHONPATH=/workspace
CKDIR=/workspace/checkpoints; mkdir -p "$CKDIR"
ZARR=/workspace/timeseries.zarr

CONFIGS=(
  "base32_lr1e4|--base-channels 32 --lr 1e-4 --require-rain-fraction 0.01"
  "base48|--base-channels 48 --lr 2e-4 --require-rain-fraction 0.01"
  "base48_allrain|--base-channels 48 --lr 2e-4 --require-rain-fraction 0.0"
  "base64|--base-channels 64 --lr 2e-4 --require-rain-fraction 0.01"
  "base64_lr1e4_allrain|--base-channels 64 --lr 1e-4 --require-rain-fraction 0.0"
  "base48_bias02|--base-channels 48 --lr 2e-4 --require-rain-fraction 0.01 --bias-penalty 0.2"
)

echo "RUNNING $(date -u)" > "$CKDIR/SWEEP_STATUS"
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"; args="${cfg#*|}"
  ck="$CKDIR/unet_${name}.pt"
  echo "=== $(date -u) :: $name :: $args ==="
  if python -m model.train --zarr "$ZARR" --device cuda --epochs 40 \
        --batch-size 32 --num-workers 8 $args --checkpoint "$ck" \
        > "$CKDIR/train_${name}.log" 2>&1; then
    python -m model.evaluate --zarr "$ZARR" --device cuda --checkpoint "$ck" \
        > "$CKDIR/eval_${name}.txt" 2>&1 || echo "  EVAL FAILED $name"
    echo "  $(grep -m1 'Best val_rmse' "$CKDIR/train_${name}.log" || echo 'no best line')"
  else
    echo "  TRAIN FAILED $name (see train_${name}.log)"
  fi
done

# Compact summary across configs.
{
  echo "=== SWEEP SUMMARY $(date -u) ==="
  for f in "$CKDIR"/eval_*.txt; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .txt | sed 's/^eval_//')
    beat=$(grep -i 'beats operational' "$f" | head -1)
    rmse=$(grep -m1 'Best val_rmse' "$CKDIR/train_${n}.log" 2>/dev/null)
    printf "%-26s | %s | %s\n" "$n" "${rmse:-?}" "${beat:-no eval line}"
  done
} | tee "$CKDIR/SWEEP_SUMMARY.txt"
echo "DONE $(date -u)" > "$CKDIR/SWEEP_STATUS"
echo "=== SWEEP COMPLETE ==="
