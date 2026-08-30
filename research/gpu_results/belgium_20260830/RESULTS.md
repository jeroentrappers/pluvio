# Belgium: the Dutch result does NOT transfer

The composite beats OPERA ~3x on CSI over the Netherlands (770 gauge-times, 133 wet).
Belgium is what pluvio.appmire.be actually serves, so it was scored separately against
KMI gauges. **It does not reproduce.**

388 BE gauge-times, 13 wet, 2026-08-30, composite of behel + bejab + bewid:

| | POD | FAR | CSI | corr |
|---|---|---|---|---|
| BE composite | 0.462 | 0.906 | 0.085 | 0.047 |
| bejab alone | 0.231 | 0.812 | 0.115 | 0.024 |
| OPERA | 0.000 | 1.000 | 0.000 | -0.016 |

Nothing here works. OPERA detects **none** of the 13 wet gauge-times; ours detects 6 but
fires on 58 dry ones. Correlations are ~0 for all three.

**13 wet points cannot support a conclusion about detection**, so the honest statement is
that Belgian performance is UNPROVEN, not that it is bad. What follows uses only the
well-sampled half of the problem: 379 dry station-times.

## Ruled out: the gauge-quantum artifact

KMI reports `precip_quantity` in 0.1 mm/10min steps, so its quantum is 0.6 mm/h. Drizzle
below that reads as exactly 0 at the gauge while a radar sees it, which would penalise the
more sensitive estimator purely by construction. Re-scoring at 0.3, 0.6 and 1.0 mm/h did
not rescue any estimator — CSI stays under 0.05 throughout. Not an artifact.

## The well-sampled finding: behel is the problem

Per-radar false alarms on the 379 dry station-times:

| radar | FA / dry | rate |
|---|---|---|
| **behel** | 50 / 379 | **13.2%** |
| bejab | 13 / 379 | 3.4% |
| bewid | 7 / 379 | 1.8% |

The composite's 15% FA rate is essentially behel leaking through. At an identical 0.3 deg
lowest elevation, behel reports echo in 18.6% of raw bins against 4.9% (bejab) and 3.7%
(bewid), and still 12.3% after declutter against ~3%.

⚠️ bewid was EXPECTED to be the bad one — it sits in the Ardennes and beam blockage there
is the known gap. It is the cleanest of the three. The expectation was wrong.

### Not clear-air echo, and not fixable with a threshold

Widespread weak summer echo suggests insects, but the false alarms are STRONGER than the
hits: FA median 0.364 mm/h against 0.263 for hits, FA max 4.35 against 1.65. Every rate
floor therefore removes hits faster than false alarms:

    floor 0.3 mm/h -> keeps 30/50 false alarms but only 2/5 hits

### It is ground clutter

The false alarms recur at fixed locations — three stations produce 36 of the 50:

| station | FA slots (of 28) | range from behel |
|---|---|---|
| 51.221N 5.027E | 15 | ~31 km |
| 50.916N 5.450E | 11 | ~17 km |
| 50.511N 6.073E | 10 | ~78 km, high ground near the Eifel |

Two at close range and one on terrain. That is the signature of stationary ground clutter,
and it is what `tools/clutter_map.py` targets.

## Partial fix, honestly labelled

A persistence mask (drop cells lit in >30% of slots) takes behel from 13.2% to 10.0% FA
with all 5 hits intact. It removes a quarter of the problem, not the problem.

**The threshold is fitted in-sample.** No cell is lit >50% of the time and a >40% mask
removes almost nothing, so the operating point sits on a cliff one day of data cannot
locate. The "no hits lost" claim rests on 5 hits and cannot detect a genuine loss of light
rain over cluttered ground. Both numbers are optimistic until re-tested on other days.

## What would actually settle Belgium

Days with real Belgian rain. Volume capture began 2026-08-30 and holds one day; the source
is a 24-h rolling cache, so history cannot be recovered retrospectively — only accumulated.
Disk allows ~16 more days before the MIN_FREE_GB guard stops collection.

---

# Four attempts to close the Belgian gap — all measured, none sufficient

The Belgian bar is OPERA's 29-day CSI **0.210** (see BASELINES.md). Our BE composite sits
at ~0.10. Each hypothesis below was a real candidate, and each was tested rather than
argued:

| # | hypothesis | result | CSI |
|---|---|---|---|
| 1 | merge rule: nearest radar -> lowest beam | no change | 0.096 |
| 2 | behel ground clutter is the problem | MAE 0.151->0.056, corr 0.057->0.279 | 0.096 |
| 3 | evaluating only :00/:30 wastes sample | 28 slots is all the volumes there are | 0.102 |
| 4 | too few radars: 3 BE -> 12 within 300 km | **identical, to three decimals** | 0.102 |

Attempt 4 is the informative failure. Adding nine radars including frave — the CLOSEST
radar to Belgium's centre at 71.9 km — changed nothing, because lowest-beam-wins always
awards Belgian cells to a Belgian radar at 0.3 deg and short range. Neighbours never win a
cell, so under a winner-takes-all merge extra radars are inert over the interior. Getting
value from them needs a weighted or quality-informed combination, not another entry in
the radar list.

## Why this cannot be settled today

13 wet gauge-times, from 28 slots on one partial day. Nothing distinguishes a 0.10 from a
0.21 at that sample size, and four consecutive negative results on the same 13 points is
the signature of a data limit rather than a modelling one. Continuing to tune against
them would be fitting noise — the same mistake the Z-R calibration already made once here.

The composite is also structurally simpler than what it is being compared to: one lowest
sweep with Marshall-Palmer, against OPERA's multi-elevation, quality-controlled,
gauge-adjusted product. That gap is real work, not a parameter.

**Blocked on data, not ideas.** Volume capture runs every 5 minutes; the source is a 24-h
rolling cache so Belgian rain days accumulate but cannot be recovered retrospectively.
