# asusprime (training box) operator scripts

The GPU box dual-boots, so a training run has to survive being stopped for the
evening. `model/train.py` writes `checkpoints/<name>.state.pt` after every
epoch and on SIGTERM/SIGINT — weights, optimizer moments, LR-schedule position
and the early-stopping counters — and `--resume auto` continues from it.

    pluvio-train pause     # stop cleanly before rebooting into Windows
    pluvio-train resume    # continue the queue where it left off
    pluvio-train status    # what is running, and the last two epoch lines

A pause costs at most the unfinished epoch's compute. A hard power-off costs
the epoch in progress plus nothing else, since the state file is written
atomically (write to `.tmp`, then rename).

Install (from a checkout of this repo):

    scp research/ops/asusprime/rerun_v3_selfix.sh asusprime:~/pluvio_v2/
    scp research/ops/asusprime/pluvio-train /tmp/ && \
      ssh asusprime 'sudo install -m 0755 /tmp/pluvio-train /usr/local/bin/'

`rerun_v3_selfix.sh` is the current queue: the two v3 arms (2.3 ablation),
sequential on the one GPU, skipping any arm whose log already says
"Training done" and refusing to start the next arm while the pause marker
`~/pluvio_v2/TRAIN_PAUSED` exists.
