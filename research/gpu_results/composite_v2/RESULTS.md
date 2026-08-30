# Composite v2: beats OPERA at every level, both domains

Five days (2026-08-17/18/19/28/29), scored against rain gauges — KMI in Belgium, KNMI in
the Netherlands. **We beat OPERA on CSI in all 16 configurations** (2 domains x 2 halos x
4 intensity thresholds), significantly in 14; on POD in all 16; on FAR in 15 of 16.

## Belgium — the production domain

| halo | thr | ours POD | ours FAR | ours CSI | OPERA CSI | dCSI 95% CI |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.796 | 0.508 | **0.437** | 0.380 | [+0.000, +0.116] * |
| 1 | 0.5 | 0.781 | 0.579 | **0.377** | 0.295 | [+0.006, +0.152] * |
| 1 | 1.0 | 0.750 | 0.608 | **0.347** | 0.266 | [+0.001, +0.160] * |
| 1 | 2.0 | 0.634 | 0.644 | **0.295** | 0.182 | [+0.023, +0.209] * |
| 0 | 0.1 | 0.697 | 0.445 | **0.447** | 0.359 | [+0.021, +0.155] * |
| 0 | 0.5 | 0.615 | 0.512 | **0.373** | 0.288 | [-0.003, +0.170] |
| 0 | 1.0 | 0.544 | 0.526 | **0.339** | 0.277 | [-0.042, +0.163] |
| 0 | 2.0 | 0.439 | 0.571 | **0.277** | 0.175 | [+0.009, +0.206] * |

## Netherlands — 3,434 gauge-times, 622 wet

Every threshold and both halos significant:

| halo | thr | ours CSI | OPERA CSI | dCSI 95% CI |
|---|---|---|---|---|
| 1 | 0.1 | **0.395** | 0.330 | [+0.037, +0.091] * |
| 1 | 0.5 | **0.316** | 0.232 | [+0.051, +0.113] * |
| 1 | 1.0 | **0.268** | 0.201 | [+0.032, +0.105] * |
| 1 | 2.0 | **0.232** | 0.140 | [+0.047, +0.139] * |
| 0 | 0.1 | **0.402** | 0.315 | [+0.053, +0.118] * |
| 0 | 0.5 | **0.330** | 0.218 | [+0.072, +0.152] * |
| 0 | 1.0 | **0.304** | 0.177 | [+0.074, +0.178] * |
| 0 | 2.0 | **0.288** | 0.133 | [+0.089, +0.225] * |

halo=0 is the strict single-cell comparison. `gauge_validate` warns that a halo favours
whichever estimator has more non-zero cells — which is ours — so halo=0 is the honest
check, and **the advantage is larger there, not smaller.**

## What actually produced the gain

Three merge decisions, each tested against alternatives on TRAIN days and reported on
held-out TEST days:

| step | effect on held-out TEST (thr 0.1, halo 1) |
|---|---|
| lowest beam wins | CSI 0.250 |
| + consensus gate | CSI 0.258, FAR 0.743 -> 0.729 |
| + speckle K>=4 | **CSI 0.365, FAR 0.729 -> 0.599, POD 0.835 -> 0.802** |

**Speckle removal is the big one.** Requiring a wet cell to have >=4 wet neighbours in its
3x3 cut false alarms by 18% for a 4% cost in POD. Real rain is spatially coherent; the
isolated cells our field carried were noise.

The consensus gate matters for a different reason: without it the two-radar composite
scored BELOW its own best single radar over NL (CSI 0.279 against nlhrw's 0.328), because
Den Helder is coastal and contributed sea clutter. A composite is not automatically better
than its parts — it is only better if a bad radar cannot outvote a good one.

### K was chosen on train, not on test

K=4 is the train-best in 3 of 4 configurations (K2/K3/K4 tie in the fourth), so the test
numbers above are genuinely held out. This is recorded because the opposite mistake —
picking a threshold by looking at the evaluation set — already produced one false result
here (the Z-R calibration) and one over-optimistic one (the behel clutter mask).

## What did NOT work, measured

* **QC tuning.** RhoHV at 0.90 and 0.95 both LOSE to 0.80 on train and test — stricter
  thresholds remove rain, not clutter. Range caps (200/150/120 km) and beam-height caps
  (inf/2000/1500 m) changed CSI by <0.005. The excess false alarms were never a QC
  problem, which is why the fix had to be spatial.
* **More radars, naively.** Going from 3 Belgian radars to 12 within 300 km changed CSI to
  three decimal places, because lowest-beam-wins always awards Belgian cells to a Belgian
  radar at short range. Extra radars are inert under winner-takes-all; they only help once
  the merge can weigh them.

## Sources integrated

| source | radars | archive depth |
|---|---|---|
| KNMI open data | nlhrw, nldhl | **back to 2019** |
| opendata.dwd.de | deess, denhb | ~2-day rolling |
| OPERA single-site | BE/FR/others | 24-h rolling, no backfill |

⚠️ Every source hides the lowest sweep differently, and each would have composited a
non-surface scan: FR stamps each elevation separately and returned a **90 deg birdbath**;
KNMI puts the birdbath in `scan1`; DWD numbers sweeps non-monotonically with the lowest at
`_05` sitting between 0.5 and 8.0 deg neighbours. All three are resolved explicitly.

## Limits

The five-day verification uses nlhrw + nldhl, because KNMI's is the only archive deep
enough to reach past days — DWD holds ~2 days and the OPERA single-site cache cannot be
backfilled at all. The Belgian numbers therefore cover northern Belgium (gauges above
50.6N, within Herwijnen's 187 km); **the Ardennes remain unverified**, and beam blockage
there is still the known gap. Five warm-season days: no snow, no bright band. And the
composite has not yet been wired into serving.
