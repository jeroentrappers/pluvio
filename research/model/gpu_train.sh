#!/usr/bin/env bash
# On-pod training + evaluation, run from a Runcrate GPU instance with the
# persistent volume mounted at /workspace. The volume must already contain
# (populated once via tools/runcrate.py populate):
#     /workspace/timeseries.zarr     the unified 20-channel store
#     /workspace/model/              the model package (incl. this script)
#     /workspace/notebooks/_lib.py   helper imported by model/dataset.py
#
# Checkpoints + the eval report are written back to the volume so they survive
# the instance being stopped. Tunables come from env vars (set via the
# instance's env_vars / launch_script), with sensible defaults.
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace

echo "=== $(date -u) pluvio GPU train starting ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Deps. Most Runcrate GPU templates ship CUDA PyTorch; only install if missing.
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
python -c "import zarr,numpy,pandas,h5py" 2>/dev/null \
  || pip install --quiet zarr numpy pandas h5py

mkdir -p /workspace/checkpoints
echo "RUNNING" > /workspace/checkpoints/STATUS

# Training. Defaults target a ~1M-param model (base-32) on the full store.
python -m model.train --zarr /workspace/timeseries.zarr \
    --device cuda \
    --base-channels "${BASE_CHANNELS:-32}" \
    --epochs "${EPOCHS:-40}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --lr "${LR:-2e-4}" \
    --bias-penalty "${BIAS_PENALTY:-0.5}" \
    --require-rain-fraction "${RAIN_FRAC:-0.02}" \
    --checkpoint /workspace/checkpoints/pluvio_unet.pt \
    2>&1 | tee /workspace/checkpoints/train.log

# Evaluation → MAE/RMSE/bias + CSI/POD/FAR by lead, saved to the volume.
python -m model.evaluate --zarr /workspace/timeseries.zarr \
    --checkpoint /workspace/checkpoints/pluvio_unet.pt --device cuda \
    2>&1 | tee /workspace/checkpoints/eval_report.txt || true

echo "DONE" > /workspace/checkpoints/STATUS
echo "=== $(date -u) pluvio GPU train finished ==="
