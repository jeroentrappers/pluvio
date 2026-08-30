# What OPERA actually scores, on large samples

Every earlier comparison here rested on ONE day (2026-08-30) — the only day with polar
volumes. That is the day our composite is measured on, so if it happened to be a bad day
for OPERA, our advantage would be an artifact of day selection rather than a real result.
These baselines test that directly, using only OPERA and gauges (no volumes needed), so
they can run over months.

## Belgium — 29 days, 19,488 gauge-times, 719 wet (KMI)

| | POD | FAR | CSI | corr |
|---|---|---|---|---|
| OPERA, Aug 1-29 | 0.529 | 0.742 | **0.210** | 0.085 |

OPERA detects 380 of 719 wet gauge-times. Under-reads intensity: gauge mean at wet
stations 2.114 mm/h against OPERA's 1.045.

⚠️ **This corrects an earlier claim in RESULTS.md.** On 2026-08-30 OPERA scored CSI 0.000
over Belgium and I wrote that it "detects nothing at all in Belgium". That was 13 wet
points on one day. Over 29 days OPERA scores 0.210, and 2026-08-30 is genuinely
exceptional for it — Belgian rain that day averaged 0.974 mm/h at wet stations against
2.114 for the month, and OPERA misses light rain.

**The Belgian bar is CSI 0.210, not 0.000.** Our BE composite scored 0.096 there, so over
Belgium we are BELOW OPERA, not above it.

## Netherlands — 13 days with >=10 wet gauge-times (KNMI)

| OPERA over NL | POD | CSI |
|---|---|---|
| median across 13 days | 0.494 | **0.156** |
| on 2026-08-30 | 0.175 | 0.130 |

OPERA's **CSI** on the benchmark day (0.130) is close to its 13-day median (0.156), so the
CSI comparison is representative and the composite's 0.430 really does beat OPERA's
typical performance by ~2.8x.

⚠️ **Its POD that day (0.175) was far below its median (0.494).** The earlier claim of "~4x
on POD" (0.782 against 0.188) is therefore an artifact of day selection. The honest POD
advantage is ~1.6x. The CSI claim survives; the POD claim does not.

## OPERA is not misregistered by our pipeline

Worth ruling out, since a projection error on our side would manufacture exactly this
result — and a warp bug already did once (the nodata erosion). Detection of wet gauges
against a grid shift, halo 1, 366 wet points:

    shift (0,0)   58.2%      shift (0,3)   59.3%      shift (3,3)   61.2%
    shift (-3,0)  56.8%      shift (0,-3)  54.9%      shift (-6,0)  51.4%

No sharp peak away from the origin: the spread is noise, not displacement. A real
misregistration would show a clear off-centre maximum. OPERA's low scores are OPERA's.

## Halo choice moves everything, so it must be stated

Same 366 wet points, OPERA detection against neighbourhood radius:

    halo 0 (0 km)  47.3%    halo 2 (6 km)  64.8%    halo 5 (15 km) 79.2%
    halo 1 (3 km)  58.2%    halo 3 (9 km)  69.7%    halo 8 (24 km) 85.0%

Any "X% detected" figure is meaningless without its halo. All composite comparisons here
use halo 1, applied identically to every estimator.
