# c17-A — static terrain channels — 2026-08-22

**Negative result: elevation / landmask / distance-to-coast do not improve the
nowcast.** Win-counts against real pysteps are *identical* to the baseline on
every metric, and the per-lead deltas are third-decimal noise.

## Setup
`nowcast_mm_c17_a_static.pt` — 12ch = 6 history + `li_flash` + `oflow_rate` +
`rate_tendency` + **3 static** (`elevation_m`, `landmask`,
`distance_to_coast_km`). Loss `precip_loss + 0.3·multiscale_loss(mode=rate)`,
i.e. c16_full's objective unchanged. Zarr `nowcast_mm_c17_v2.zarr`: 68,361 issues,
**2024-08-14 → 2026-08-19** (24 months, +73% over c16's 14), cadence-15, 13 leads,
256² (~3 km). AdamW wd 1e-4, dihedral augment, cosine, patience-4, batch-16.
15 epochs / ~41 h on the free RTX-2060; best val 0.0196 at epoch 10.

## Wins vs real pysteps (leads 10–120, 12 leads)
| variant | MAE | CSI@0.1 | CSI@1 | FSS@3km | FSS@9km | FSS@15km |
|---|---|---|---|---|---|---|
| c16_full re-eval (9ch) — **the baseline** | 12/12 | 10/12 | 12/12 | 10/12 | 0/12 | 0/12 |
| **c17-A + static (12ch)** | 12/12 | 10/12 | 12/12 | 10/12 | 0/12 | 0/12 |

⚠️ The baseline row is `nowcast_mm_c16_full.pt` **re-scored on this zarr** with
`--no-static` (step 0b of `run_c17.sh`), not the published c16 table. That matters:
c17 changed two things at once — static channels *and* the window growing from 14
to 24 months — so only a same-window baseline isolates anything. Notably the
re-scored baseline reproduced its 14-month win-counts **exactly** (val 98,899 →
171,985 samples), which is itself a useful result: the c16 numbers were not a
small-sample artifact, and **the extra 10 months of data changed nothing either**.

## Per-lead delta (c17-A − baseline; positive = static helped)
| lead | ΔCSI@1 | ΔCSI@0.1 | ΔFSS@3km | ΔFSS@9km | ΔFSS@15km |
|--:|--:|--:|--:|--:|--:|
| 10 | +0.0110 | -0.0070 | -0.0050 | -0.0050 | -0.0040 |
| 20 | +0.0110 | +0.0010 | +0.0010 | +0.0020 | +0.0020 |
| 30 | +0.0060 | -0.0020 | -0.0020 | -0.0030 | -0.0030 |
| 40 | +0.0120 | -0.0040 | -0.0040 | -0.0040 | -0.0050 |
| 50 | +0.0100 | -0.0040 | -0.0040 | -0.0040 | -0.0040 |
| 60 | +0.0110 | -0.0040 | -0.0040 | -0.0050 | -0.0050 |
| 70 | +0.0010 | -0.0010 | +0.0000 | +0.0000 | -0.0010 |
| 80 | +0.0020 | -0.0040 | -0.0030 | -0.0040 | -0.0030 |
| 90 | +0.0030 | -0.0020 | -0.0020 | -0.0020 | -0.0020 |
| 100 | +0.0040 | -0.0040 | -0.0050 | -0.0050 | -0.0060 |
| 110 | -0.0060 | +0.0010 | +0.0020 | +0.0030 | +0.0040 |
| 120 | -0.0180 | +0.0010 | +0.0000 | +0.0020 | +0.0030 |
| **mean** | **+0.0039** | **-0.0024** | **-0.0022** | **-0.0021** | **-0.0020** |

Read: **CSI@1 gains ~+0.011 at 10–60 min but reverses at long leads** (−0.006 at
110, −0.018 at 120); everything else is ~−0.002, i.e. marginally *worse*. All of
this is within seed noise for a single-seed run. No metric changed a win-count.

## The gate is untouched
| lead | ΔFSS@9km | ΔFSS@15km |
|--:|--:|--:|
| 10 | -0.024 | -0.028 |
| 20 | -0.022 | -0.025 |
| 30 | -0.022 | -0.035 |
| 40 | -0.022 | -0.044 |
| 50 | -0.042 | -0.065 |
| 60 | -0.027 | -0.053 |
| 70 | -0.037 | -0.068 |
| 80 | -0.033 | -0.061 |
| 90 | -0.033 | -0.066 |
| 100 | -0.014 | -0.046 |
| 110 | -0.029 | -0.061 |
| 120 | -0.024 | -0.053 |

FSS@9km and FSS@15km still lose at **all 12 leads**, −0.014 to −0.068, worst at
50–90 min — essentially unchanged from c16. The static channels did not touch the
one deficit that blocks promotion.

## Interpretation
Terrain is a *stationary* field; the nowcast head is being asked to correct
advection over 0–2 h, where the useful signal is motion and growth/decay, not
where the hills are. Any orographic climatology the channels carry is probably
already implicit in the 6-frame radar history. This does not rule terrain out for
the **outlook/downscaling** head (where the ERA5→OPERA downscaler already shows
+33–43%) — only for the nowcast.

With static ruled out **and** data volume ruled out, the FSS@9/15km deficit must
come from the objective or the advection prior — which is exactly what arms C
(exceedance-pooled loss) and B (drop advection) test.

## Caveats
Single seed. Static entered as 3 raw normalised channels concatenated at the
input; a different injection (FiLM conditioning, or terrain-modulated skip
connections) is not excluded by this result. Because A moved neither metric, the
static-vs-window confound is moot here, but a `--no-static` eval of this same
checkpoint would separate them for ~15 min of GPU if it ever matters.
