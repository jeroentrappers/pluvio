# We do NOT beat best-in-class

We beat OPERA everywhere we have tested. That is a weaker claim than it sounds: OPERA is
the pan-European composite and is not the strongest product over any single country.
National services run their own with their own QC, and those are what a user actually
compares against — KNMI's RTCOR is what sits behind Buienradar.

Benchmarked against RTCOR on the same five days, same gauges, same halos:
**RTCOR beats us in 15 of 16 configurations, significantly.**

## Netherlands (3,291 gauge-times inside the RTCOR footprint)

| halo | thr | ours CSI | **RTCOR CSI** | OPERA CSI | d(ours-RTCOR) 95% CI |
|---|---|---|---|---|---|
| 1 | 0.1 | 0.395 | **0.482** | 0.333 | [-0.110, -0.065] |
| 1 | 0.5 | 0.315 | **0.524** | 0.235 | [-0.240, -0.179] |
| 1 | 1.0 | 0.269 | **0.495** | 0.201 | [-0.265, -0.189] |
| 1 | 2.0 | 0.234 | **0.464** | 0.142 | [-0.281, -0.183] |
| 0 | 0.1 | 0.405 | **0.584** | 0.318 | [-0.210, -0.147] |
| 0 | 0.5 | 0.335 | **0.599** | 0.221 | [-0.310, -0.217] |
| 0 | 1.0 | 0.304 | **0.612** | 0.175 | [-0.369, -0.250] |
| 0 | 2.0 | 0.292 | **0.544** | 0.134 | [-0.333, -0.174] |

At halo 0, thr 1.0, RTCOR scores **twice** our CSI (0.612 against 0.304).

## Belgium (597 gauge-times inside the RTCOR footprint)

| halo | thr | ours CSI | **RTCOR CSI** | OPERA CSI |
|---|---|---|---|---|
| 1 | 0.1 | 0.442 | **0.498** | 0.383 |
| 1 | 2.0 | 0.299 | **0.431** | 0.182 |
| 0 | 0.1 | 0.457 | **0.520** | 0.362 |
| 0 | 2.0 | 0.277 | **0.412** | 0.175 |

RTCOR wins in Belgium too — outside its own country, on a domain it was not built for.

**The ordering is consistent: RTCOR > ours > OPERA.**

## The gap is false alarms, not detection

At NL halo 0, thr 1.0: our FAR is **0.553** against RTCOR's **0.204**. RTCOR also has the
higher POD (0.726 against 0.488), so this is not a threshold trade — they are better on
both axes at once.

Our speckle filter already cut FAR from 0.729 to 0.599 and was the single biggest gain we
found. RTCOR is another 0.35 below that. This is not a tuning gap; it is the difference
between a hand-rolled Marshall-Palmer conversion off one low sweep and an operational
chain with:

* 8 radars rather than our 2
* real clutter and anaprop removal
* vertical profile of reflectivity correction (we have none — this is likely the largest
  single missing piece, since we sample one low sweep and assume it represents the surface)
* dual-pol hydrometeor classification
* their own Z-R handling per precipitation type

## What this means

**Do not treat "beats OPERA" as success.** The honest position:

* against OPERA — a genuine, significant win at every level, both domains
* against the national best-in-class — we lose, by roughly 0.09 to 0.31 CSI

Closing it means implementing what operational chains do, in rough order of expected
value: VPR correction, proper clutter/anaprop QC, more radars per pixel, and only then
gauge adjustment. `baltrad/rave` is the open-source implementation of most of this and is
literally the toolbox EUMETNET uses to QC input to OPERA, so it should be read before any
more of it is derived by hand here.
