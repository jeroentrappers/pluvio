# The Ardennes: a geometry problem, not a processing one

We beat OPERA across the Netherlands and northern Belgium at every intensity level. In
the Ardennes we do not, and it is worth being precise about why, because three plausible
explanations turned out to be wrong.

**119 gauge-times, 19 wet** (2026-08-28/29/30, southern Belgian KMI gauges below 50.6N),
composite of denhb + deess + deoft + defld + nlhrw + nldhl:

| halo | thr | ours CSI | OPERA CSI | dCSI 95% CI |
|---|---|---|---|---|
| 1 | 0.1 | 0.222 | **0.239** | [-0.132, +0.095] |
| 1 | 0.5 | 0.146 | **0.192** | [-0.199, +0.090] |
| 0 | 0.1 | **0.261** | 0.242 | [-0.126, +0.156] |
| 0 | 0.5 | 0.074 | **0.190** | [-0.300, +0.051] |

Nothing is significant on 19 wet points, but the point estimates favour OPERA and our FAR
is clearly worse (0.768 against 0.711 at thr 0.1).

## Wrong explanation 1: beam blockage

The obvious suspect, and the term ODYSSEY weights by that we lacked. Implemented properly
(`tools/beam_blockage.py`, `tools/build_dem.py`) it is **not** the cause:

| radar | mean quality | cells partly blocked (Q<0.7) |
|---|---|---|
| nlhrw | 0.955 | 0.00% |
| nldhl | 0.999 | 0.00% |
| denhb | 0.973 | 0.11% |
| deess | 0.954 | 5.98% |

Blockage is real but small — these sites were chosen to avoid it.

Two bugs had to be fixed before those numbers meant anything, and both produced confident
nonsense first:
* **DEM resolution.** Against the analysis grid's own 3 km `elevation_m`, blockage came out
  at 0% everywhere. 3 km terrain is too smooth to clip a beam. `tools/build_dem.py`
  mosaics the open Copernicus 30 m COGs (no auth, on S3) to 500 m — 68 tiles, 1576x2418.
* **Antenna height.** KNMI's HDF5 records no antenna height anywhere, so it defaulted to
  0 m, putting the beam at sea level where any terrain clips it. That gave Herwijnen — in
  the flattest part of the Netherlands — **65% blocked**. The real heights were read from
  the ODIM `/where/height` published for the same radars through the OPERA feed
  (nlhrw 25 m, nldhl 55 m), after which Herwijnen correctly shows 0%.

## Wrong explanation 2: the merge rule

ODYSSEY is documented as a weighted average over contributing radars, weighted by quality
index, distance, and inverse beam altitude. Implemented (geometric terms) and tested on
held-out days, it **loses to our winner-takes-all rule**:

| TEST thr 0.1 | POD | FAR | CSI |
|---|---|---|---|
| ours (lowest beam + consensus + speckle) | 0.802 | 0.599 | **0.365** |
| odyssey-weighted + speckle | 0.854 | 0.629 | 0.349 |
| OPERA | 0.599 | 0.610 | 0.309 |
| odyssey-weighted alone | 0.892 | 0.745 | 0.248 |

⚠️ This is NOT a fair reproduction of ODYSSEY: only its geometric half is implemented, and
the real quality index carries clutter and blockage terms. What it does show is that
weighted averaging by itself smears rather than helps — the speckle filter is doing the
work in both variants.

## The actual reason: every archived radar overshoots the Ardennes

Beam height above ground over the five southern Belgian gauges (terrain 350-600 m):

| radar | height above ground | usable? |
|---|---|---|
| **bewid** (Wideumont, 585 m) | **240-1050 m** | yes |
| denhb | 640-2450 m | marginal |
| deess | 1450-4500 m | no |
| nlhrw | 1400-4400 m | no — above the melting layer |
| nldhl | 5000-9700 m | no |

Only Wideumont samples the Ardennes near the ground. Everything in our multi-day archive
looks at ice and melting-layer echo 1.4-4.4 km up, which is precisely why our FAR there is
worse: we detect aloft precipitation that never reaches the gauge.

**OPERA includes bewid. Our archive does not**, because bewid reaches us only through the
OPERA single-site 24-h rolling cache, which cannot be backfilled — unlike KNMI's radars,
which have an open archive to 2019. No amount of QC, merging or blockage correction fixes
a radar that is not there.

The route is capture: bewid is being collected every 5 minutes, so this becomes answerable
as wet Ardennes days accumulate. Nothing else will do it.
