# Training on Runcrate (RTX-4090) — launch kit

Persistent-volume pattern: upload the 12 GB store **once**, reuse across runs.
The GPU only bills during training, not the upload.

## Cost (live rates, Jun 2026)
- Single **RTX-4090** `focused-neumann-rtx4090-3615` (Oslo, 70 GB): **$0.66/hr**
- Cheapest **CPU** (for populate) `…cpu_small…` : **$0.39/hr**
- A full run ≈ 6–10 h → **~$4–7**. Populate ≈ 40 min → **~$0.25**.
- **20 credits ≈ ~30 GPU-hours ≈ 3–4 full runs.** Top up for a large sweep.
- Volume storage: a few $/month; persists between runs.

## Prereqs
- `RUNCRATE_API_KEY` in `research/.env` (done) + credits loaded (done).
- An SSH key (`ssh-keygen -t ed25519` if needed).
- `tools/runcrate.py gpus` works (verified).

## One-time setup

```bash
cd research
. .venv/bin/activate

# 1. See current GPU/CPU ids + rates
python -m tools.runcrate gpus

# 2. Register your SSH key
python -m tools.runcrate ssh-key --name me --pub ~/.ssh/id_ed25519.pub      # → ssh_key_id

# 3. Create a persistent volume (≥ store size + checkpoints; ~40 GB is plenty).
#    Pick a region close to the GPU region (Oslo) for fast mounts.
python -m tools.runcrate volume-create --name pluvio-data --size 40 --region <region_id>   # → volume_id

# 4. Build the upload bundle (zarr + the code the trainer imports)
tar -C data -cf /tmp/pluvio_bundle.tar timeseries.zarr
tar -C . -rf /tmp/pluvio_bundle.tar model notebooks/_lib.py        # append code

# 5. Boot a cheap CPU box with the volume attached, scp the bundle in, untar, stop.
python -m tools.runcrate populate --volume <volume_id> --ssh-key <ssh_key_id> \
       --type sleepy-archimedes-cpu_small-de28
#    → it prints the instance id; `status <id>` to get the SSH host, then:
#        scp /tmp/pluvio_bundle.tar <user>@<host>:/workspace/
#        ssh  <user>@<host> 'cd /workspace && tar -xf pluvio_bundle.tar && rm pluvio_bundle.tar'
#    then: python -m tools.runcrate stop <cpu_instance_id>
#    (~40 min at this server's ~42 Mbps uplink; one time only.)
```

## Each training run

```bash
python -m tools.runcrate train --volume <volume_id> --ssh-key <ssh_key_id> \
       --gpu focused-neumann-rtx4090-3615 \
       --epochs 40 --base-channels 32 --batch-size 32 --rain-frac 0.02
# → boots the 4090, runs model/gpu_train.sh on the mounted volume:
#     train.py --zarr → checkpoints/pluvio_unet.pt   (+ train.log)
#     evaluate.py     → checkpoints/eval_report.txt  (MAE/RMSE/CSI by lead)
#     checkpoints/STATUS flips RUNNING → DONE

python -m tools.runcrate status <instance_id>          # poll until STATUS=DONE
# pull results (small):
scp <user>@<host>:/workspace/checkpoints/{pluvio_unet.pt,eval_report.txt} ./checkpoints/
python -m tools.runcrate stop <instance_id>            # volume + checkpoints persist
```

`base-channels 32` ≈ the ~1M-param design size; bump to 48/64 if VRAM allows.
Leads are {30,60,90,120} (the store's 30-min cadence — see model/zarr_dataset.py).

## First-run shakedown (expect ~10 min of fiddling, a few cents)
Things only verifiable by actually provisioning once:
- **SSH host/user** — read from `status <id>` output (the API response carries it).
- **Region co-location** — volume region vs GPU region; if a mount fails, recreate
  the volume in the GPU's region.
- **CUDA/PyTorch template** — `gpu_train.sh` pip-installs torch only if the image
  lacks it; if the base image already has CUDA torch it's skipped.
Do this interactively the first time; subsequent runs are one `train` command.

## Manual fallback (no API)
Deploy a 4090 + attach the volume from the Runcrate web UI, `scp` the bundle to
`/workspace`, and run `bash model/gpu_train.sh` over SSH. Same result.
