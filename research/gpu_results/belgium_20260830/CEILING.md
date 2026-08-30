# Why the production model cannot beat OPERA over Belgium

Not a tuning problem. An architectural one, and it is visible in the data contract
rather than in any metric.

## The model's training label IS OPERA

From `model/seamless_dataset.py`:

    opera_rate  (n, H, W)  mm/h   OPERA analysis at each issue-time (truth)
    target = opera_rate at issue+lead

The deployed c17-C checkpoint confirms the recipe it was trained under:

    in_ch 12 | aux ['li_flash'] | static [distance_to_coast, elevation, landmask] | adv True

A model regressed onto OPERA reproduces OPERA, including OPERA's misses. It cannot
systematically detect rain that is absent from its own labels — there is no gradient
signal for a storm the target never contained.

## How much that costs over Belgium — measured

OPERA against KMI gauges, 29 days, 19,488 gauge-times, 719 wet:

    POD 0.529 -> OPERA misses 339 of 719 wet gauge-times

Those 339 are **unlearnable** by anything trained on OPERA. They set a hard ceiling on
the production nowcast over Belgium, independent of architecture, loss function, channel
recipe or training length.

## Raw radar does see what OPERA misses

This is the part that makes the ceiling worth breaking rather than accepting. Detection
of wet gauges, NL, 770 gauge-times, by rain intensity:

| gauge intensity | n | our radar composite | OPERA |
|---|---|---|---|
| light 0.1-1 mm/h | 83 | **71%** | 13% |
| moderate 1-3 mm/h | 25 | **92%** | 20% |
| heavy >3 mm/h | 25 | **88%** | 36% |

And on 2026-08-30 over Belgium the composite detected 6 of 13 wet gauge-times where
OPERA detected 0. The signal exists in the polar volumes and is being discarded when
OPERA is used as truth.

## What this means for the goal

"Better than OPERA over Belgium" is not reachable by improving the nowcast model while
OPERA remains its target. The ordering is forced:

1. Accumulate Belgian polar volumes (capture runs every 5 min; the source is a 24-h
   rolling cache, so this is accumulate-only and cannot be backfilled).
2. Build the composite QPE well enough over Belgium to serve as truth — which needs the
   quality-weighted merge that the 12-radar test showed is the missing mechanism, not a
   longer radar list.
3. Retrain on that truth instead of `opera_rate`.

Step 3 is what actually lifts the ceiling. Steps 1 and 2 are prerequisites, and step 1
is a wall-clock constraint no amount of work removes today.

⚠️ This argument rests on the training contract and on OPERA's measured 0.529 POD, not on
a direct model-vs-OPERA run over Belgium. That run was attempted and is blocked: no zarr
on the production host carries the `oflow_rate`/`rate_tendency` channels the deployed
recipe requires (they were generated on the training box), so driving c17-C over historical
Belgian dates needs those channels rebuilt first. The prediction to test when they exist
is that the model scores at or BELOW OPERA on Belgian gauge detection, never above.
