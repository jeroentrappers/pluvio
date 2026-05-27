# CorrDiff on a rented GPU — turnkey recipe

The radar-only UNet (`model/unet.py`) trains on CPU and is the baseline.
CorrDiff is the state-of-the-art step up: a conditional **diffusion** model
that downscales coarse NWP (AIFS / ALARO) to high-resolution precipitation,
conditioned on radar + satellite + surface obs. It needs a GPU — there is
no honest way to train it on CPU.

This directory is the bridge: it documents exactly how to take the data we
already collect and train CorrDiff on a rented box, so the only thing
standing between us and a trained model is ~€100 of GPU time.

## Why not on this machine

CorrDiff's reference config (NVIDIA `physicsnemo`) trains for ~2 GPU-days on
an A100 80GB. Diffusion sampling is iterative (25–100 denoising steps per
inference). On a 14-core CPU, one training epoch would take days and the
full run would never realistically finish. The UNet baseline exists
precisely so we validate the data pipeline *before* spending on GPU.

## Hardware

| Option | $/hr | Notes |
|---|---|---|
| Lambda Cloud A100 40GB | ~$1.10 | cheapest reliable |
| RunPod A100 80GB (spot) | ~$1.20 | spot can be reclaimed |
| Vast.ai RTX 4090 24GB | ~$0.40 | works for the small config; slower |
| Hetzner GPU (dedicated) | ~€200/mo | only if training recurrently |

A full training run is ~2 GPU-days on an A100 ⇒ ~€55. Budget €100 with
restarts and evaluation.

## Step 1 — assemble the training data locally

Run the collectors for ~12 months (see `model/README.md`). Then convert the
per-file HDF5 / GeoTIFF into the zarr layout CorrDiff expects:

```bash
# (to be written) — pre-bakes radar + AIFS + Meteosat + AWS onto the common
# 1-km grid as a single zarr store, train/valid/test splits by month.
python -m model.corrdiff.build_zarr --data ../data --out ../data/corrdiff.zarr
```

The zarr store is ~250 GB for 12 months. rsync it to the GPU box once.

## Step 2 — provision the GPU box

```bash
# On the rented instance (Ubuntu 22.04 + CUDA 12.x):
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo/examples/generative/corrdiff
pip install nvidia-physicsnemo
pip install -r requirements.txt
```

## Step 3 — point CorrDiff at our data

CorrDiff is configured via Hydra YAML. Copy `conf/config_training_pluvio.yaml`
(in this directory) into `physicsnemo/examples/generative/corrdiff/conf/`
and adjust:

- `dataset.data_path` → the rsynced `corrdiff.zarr`
- `dataset.in_channels` → our conditioning channels (radar history + AIFS +
  Meteosat + AWS = ~24)
- `dataset.out_channels` → 1 (precipitation)
- `model.img_resolution` → 128 (pad our 100×100 to 128 for the UNet's
  power-of-two downsampling)

## Step 4 — train

```bash
torchrun --nproc_per_node=1 train.py \
    --config-name=config_training_pluvio.yaml
```

Watch the loss in `outputs/`. CorrDiff trains a *regression* UNet first,
then a *diffusion* residual on top — two phases, both in the config.

## Step 5 — evaluate against KMI/KNMI

Export the checkpoint, run inference over the held-out test month, then feed
the output through the same verification we use for the UNet:

```bash
python -m model.evaluate --data ../data/knmi --checkpoint corrdiff_export.pt
```

The win condition is unchanged: beat the operational nowcast on CSI/RMSE,
especially heavy rain at +30 min and beyond.

## What's in this directory

- `README.md` — this recipe.
- `conf/config_training_pluvio.yaml` — starting Hydra config (will need
  tuning on the GPU; the channel counts and data path are the bits to set).
- `build_zarr.py` — **TODO**: the zarr pre-baking step. Stubbed until we
  commit to the 12-month pull.

## Honest status

This is a recipe, not a tested run. The exact Hydra hyperparameters
(batch size, learning rate, diffusion steps, channel normalisation) need
iteration on the actual GPU against our actual data — that's the nature of
diffusion training. What's *not* uncertain: the data pipeline (proven by
the UNet), the verification harness (proven on real KNMI data), and the
input/output contract (radar+aux → corrected precip). CorrDiff slots into
that contract; it doesn't change it.
