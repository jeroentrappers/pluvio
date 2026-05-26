# Pluvio correction model

A small UNet that takes the KNMI operational nowcast plus auxiliary
signals (Meteosat IR + instability indices, AWS surface state, ALARO NWP)
and outputs a corrected precipitation field.

See `docs/model_architecture.md` for the design rationale.

## Files

- `unet.py` — the architecture (~1 M parameters).
- `dataset.py` — the joining layer that assembles input tensors from the
  collectors' on-disk outputs. **The `_build_index` method is the last
  piece to wire up against your local data layout** — see the TODO.
- `train.py` — training loop with weighted Huber loss, mixed precision,
  early stopping.

## End-to-end workflow

```bash
# 0. install the training extras (PyTorch + rasterio for GeoTIFF parsing)
pip install ".[ml]"

# 1. collect ~3 months of training data (multi-hour, ~60 GB)
python collectors/fetch_knmi_archive.py --start 2026-02-01 --end 2026-05-01 \
    --dataset radar_forecast
python collectors/fetch_knmi_archive.py --start 2026-02-01 --end 2026-05-01 \
    --dataset rtcor
python collectors/fetch_kmi_aws.py     --start 2026-02-01 --end 2026-05-01
python collectors/fetch_eumetsat_msg.py --start 2026-02-01 --end 2026-05-01 \
    --layers msg_fes:ir108,msg_fes:wv062,msg_fes:gii_kindex,msg_fes:gii_liftedindex,msg_fes:cth,msg_fes:rdt
python collectors/fetch_alaro_24h.py   --start 2026-02-01 --end 2026-05-01 \
    --layer Total_precipitation --layer Total_cloud_cover --layer ...

# 2. wire up PluvioCorrectionDataset._build_index() to point at the
#    actual on-disk paths (one-time job, ~50 LoC).

# 3. train
python -m model.train --data ../data --epochs 20 --batch-size 16

# 4. evaluate using the existing verification notebook with --model
python notebooks/01_verification.py --data data/knmi --model checkpoints/pluvio_unet.pt
```

## Compute budget

- Pre-processing / indexing: ~30 minutes once.
- Training: ~8 hours on a single RTX 4070 / 4080-class GPU (~1 M params,
  ~70 k training samples, 20 epochs with mixed precision).
- Inference at serve time: <50 ms for a full 24-lead-step forecast for
  one location.

## What we still need before training is meaningful

1. **Three months of data**. The collectors are ready; run them in the
   background and check disk fills sanely.
2. **`_build_index()` implementation**. About 50 lines of pathlib /
   `datetime` arithmetic to walk the on-disk layout and emit valid
   `(issue_time, lead_min, SamplePaths)` tuples. Marked with a clear
   `NotImplementedError` so you can't accidentally train on a broken
   index.
3. **Static fields** (terrain, landmask, distance-to-coast) baked into
   `assets/static/` as 100×100 numpy arrays. Source from SRTM / Natural
   Earth, one-off.
4. **GeoTIFF reprojection** via rasterio to land each Meteosat / ALARO
   tile on the KNMI radar grid. The current `_load_geotiff` stub returns
   zeros — fill it in.

These four pieces are the minimum to put real numbers through. Each is a
small, independent task.
