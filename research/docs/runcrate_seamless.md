# Seamless model — GPU run launch kit

End-to-end commands to build the training store and run the first GPU training
(the OPERA **nowcast baseline** while MTG/AIFS accumulate). Staging is from
hetz1's fast local NVMe (`/opt/pluvio/stage`, populated by the `pluvio-stage`
unit) — not the slow CIFS box.

## 0. Prereqs on the GPU node
A CUDA box (Runcrate 4090-class) with this repo + the research venv
(`pip install -e .` → torch+CUDA, rasterio, pyproj, zarr, numpy).

## 1. Uplink the staged data (hetz1 → GPU node)
```bash
# on the GPU node — pull the staged crops over the fast link
rsync -a --info=progress2 ansible@<hetz1>:/opt/pluvio/stage/ ./stage/
#   ./stage/opera/RATE/...  ./stage/aifs/...  ./stage/mtg_li/...  ./stage/mtg_l2/...
```

## 2. Build the seamless zarr (from the staged copy)
```bash
python -m tools.build_seamless_zarr \
    --storage ./stage --out ./seamless.zarr --cadence-min 15
#   ~1–2 h for 22 months of OPERA (47k reprojects). --limit N to smoke-test first.
#   OPERA truth is full; MTG/AIFS aux are NaN until they accumulate (forward-only).
```

## 3. Train (the nowcast baseline)
```bash
python -m model.train_seamless \
    --zarr ./seamless.zarr --epochs 30 --batch-size 16 \
    --out checkpoints/pluvio_seamless.pt
#   AMP auto-enables on CUDA. ~hours on a 4090, ~€5–10. Best val checkpoint saved.
```

## 4. Verify (head-to-head, against baselines that matter)
```bash
python -m model.eval_seamless --zarr ./seamless.zarr \
    --ckpt checkpoints/pluvio_seamless.pt --samples 4000
```
Scores the checkpoint **and three baselines** per lead band on the held-out
(most-recent) window: **persistence** (floor), **optical-flow** (pysteps-style
advection — the real 0–2 h bar) and **raw-AIFS** (the real outlook bar). Reports
MAE/RMSE, CSI at τ=0.1/1, scale-aware **FSS**, and **CRPS** (= MAE for a point
forecast; from quantiles for a probabilistic ckpt), and prints the grid
resolution so CSI isn't misread as a 1-km score. The verdict line is "did the
model beat the *reference baseline for that regime*" — beating persistence is not
a result. Promote only on the champion/challenger gate (`docs/plan_overview.md §5`).

## Notes
- First run is **nowcast-only-meaningful**: with NaN→0 AIFS, the outlook head has
  no NWP signal — expected until AIFS/MTG build up (weeks). Re-build + re-train
  once the obs channels have depth for the full seamless/multi-day result.
- Re-run is cheap; champion/challenger keeps unattended re-trains safe.
