# History-mode temporal consistency: the full investigation (2026-08-31)

The user report: "volumes blinking in and out of existence — physically impossible."
Correct on both counts. Chain of findings, each measured before acted on:

## Fixed

1. **10-min stepping** → 5-min (every feed is 5-min native; at 1 km pixels a 10-min
   step moves cells 3–6 km — reads as jumping).
2. **Compute-time incompleteness** — the big one. Frames were composited once, when
   the timer first saw the stamp; Belgian files arrive ~12 min late, so newest frames
   were built from 5–6 radars and never revisited: whole radar footprints appeared and
   vanished between neighbouring frames. Now: frames carry their radar count, runs
   process chronologically, and any frame younger than 60 min built incomplete is
   recomputed — every frame converges to the deterministic full-set composite.
3. **Writer interleaving** — a manual rebuild and the 5-min timer ran concurrently
   (journal: 18 timer events during one rebuild) while the VPR EMA made frames
   history-dependent: served wet area ballooned +59% in one step (20.3k → 32.2k cells)
   while live RTCOR conserved area on the same pair. Per-radar fields without EMA were
   stable across that exact pair (nlhrw 6.35 → 6.29% wet), i.e. the smoothing's target
   flicker did not exist. EMA removed; every producer run takes an exclusive flock.
4. **Inter-radar seams** — measured calibration offsets in overlap regions: bejab
   −5.8 dB, bewid −6.5 dB, deess +5.3 dB vs the gauge-anchored nlhrw. 6 dB is ×2.3 in
   rain rate: exactly the visible edges between sources. Harmonisation + feathered
   height-window blending replaced the hard winner-takes-all switch; a same-day gauge
   guard showed improvement, not regression (CSI up at all thresholds, wet bias
   −0.86 → +0.02).
5. **Sub-8 km² trace clusters** — the 3×3 speckle filter spans 81 km² at 3 km but only
   9 km² at 1 km. Area-based despeckle (drop components < 8 km² that nowhere reach
   1 mm/h) for the display product; archive unfiltered.
6. **Trace opacity ramp** (display only): 0.1–0.5 mm/h fades in instead of rendering a
   solid swatch, so threshold-crossing scan noise reads as marginal drizzle rather
   than blinking cells. Colours unchanged; the legend stays true.

## Where it now stands, honestly

Churn arbiter against live RTCOR, same 5-min pairs, same metric: ours **1.53×** RTCOR's
gain-component count (median 638 vs 427). Ablation pinpoints the source: removing the
Belgian radars collapses churn to 355 (below RTCOR); removing fuzzy QC or the German
radars changes nothing. The Belgian radars are the only feed without dual-polarimetric
moments — no polarimetric QC is possible on them — and their +6 dB harmonisation scales
scan-to-scan noise by ×2.3.

## Negative results, kept

- Post-calibration noise floors on the QC-less radars (8/11 dBZ): churn 642–796, far
  from the 355 floor — the flicker is in moderate echo, not just trace.
- Quality penalty (×0.5) on QC-less radars in the blend: 763 — overlap regions only.
- **Temporal median-of-3: made it WORSE** (599 → 739 components) while collapsing
  physical maxima (214 → 83 mm/h). Rain advects 2–3 pixels per 5-min frame at 1 km, so
  a stationary-pixel median re-fragments every moving edge. Any temporal treatment must
  be advection-aware (Lagrangian, as RTCOR's own accumulation is).

## Interpolated sub-frames (the Buienradar ingredient) — implemented

The reference animations are interpolated video, not raw scans. Three attempts, two
rejected on measurement:

1. Advection-corrected ACCUMULATION per frame (RTCOR Eq. 5): painting the motion track
   fragments the wet contour (gain components 600 → 797) and halves peaks — right for
   gauge comparison, wrong for animation.
2. Eq.-4 CROSS-FADED interpolants: every interpolant is a weighted union of both wet
   masks, so moved cells leave 33–67 %-weight ghosts that the next exact scan deletes —
   a sawtooth of 5.06 pp median wet-delta per 100 s (raw scans: 1.36 per 300 s).
3. **Semi-Lagrangian nearest-scan advection** (accepted): below half-way warp the
   earlier scan forward along the flow, past half-way warp the later scan backward; the
   handover sits exactly where the two advected scans align best. Robust flow: Gaussian
   pre-blur σ2.5, 21-px flow smoothing, magnitude clamp at 8 px/5 min (~100 km/h).
   Result: 109 served frames at ~100 s cadence, median wet-delta 1.43 pp/step (p90
   2.38), peaks preserved. Scans themselves stay exact; store and archive untouched.

## The Flanders blinking: a reader bug masquerading as radar noise

User report: cells blinking in and out over East/West Flanders — bejab's home footprint.
Measurement chain:

1. bejab's own wet mask flipped **93–98% of cells between consecutive scans at every
   intensity** (nlhrw, dual-pol QC'd, same day: 62–82%). Real rain cannot decorrelate
   in 5 minutes at 2 mm/h — so not noise, something structural.
2. Root cause: the multi-sweep ODIM and DWD readers rolled rays by −a1gate. ODIM rows
   are already north-aligned; a1gate records where the antenna happened to START.
   bejab's a1gate varies per scan (136 → 298), so every scan was rotated by a different
   random angle: inter-scan correlation 0.471 unrolled vs **−0.101 rolled**. deess's
   constant a1gate=100 rotated German rain to the wrong geography (corr vs QC'd
   neighbours +0.157 unrolled, −0.511 rolled). The original gauge-validated
   lowest-sweep reader never rolled — which is why the core 5-day KNMI-only results
   stand; every arm that composited BE/DWD radars through the multi-sweep readers is
   tainted and flagged.
3. Cascade: the first calibration fit compared rotated fields — its ±5–6.5 dB offsets
   measured the rotation, not the electronics. Re-fit on straight fields: −2.15…+1.11
   dB, the normal inter-network range, on ~10× larger overlap samples.
4. After both fixes: bejab flips 63/77/86% at 0.1/0.5/1.0 mm/h (trace now at the QC'd
   radar's level), Flanders composite flip 91% → 75% (NL reference 53%). The residual
   Flanders-vs-NL gap is the genuine no-dual-pol handicap plus a convective day.

Also tried before finding the real bug: two-scan persistence confirmation for QC-less
radars — mathematically saturated at bejab's then-apparent 7% noise density (an 8 px
dilated confirm mask covers everything) and moved Flanders only 91 → 82. Kept in the
chain, where it now operates on correlated fields as a mild safety net.

## What would actually reach parity

1. Advection-aware display interpolation (RTCOR Eq. 4–5, Farnebäck flow) — needs an
   optical-flow dependency; also what makes Buienradar's animation look smooth.
2. Per-radar statistical clutter maps for the Belgian radars, fitted over accumulating
   archive days (the qpe archive at 5-min makes this possible within a week or two).
3. Re-fit the calibration offsets over multiple wet days — the current ones come from
   one morning (8 stamps) and carry melting-layer risk; the +6 dB values are first
   estimates.
4. KMI's own QC'd Belgian data, if it ever becomes openly available with dual-pol
   moments — the structural fix.

## Continental-grid regressions and fixes (2026-08-31 evening)

Going Belgium/1.2 km -> continental/4 km silently broke two display-layer tunings:

1. **Pixel-denominated flow constants.** Farneback blur/window/smoothing/clamp were
   tuned in px on the 1.2-km grid; at 4 km they meant a 140-km analysis window and a
   32-km-per-5-min clamp. Measured over NL: 48-62% wet-cell flips on the second
   interpolant of every gap — user-visible blinking. Fix: all scales in km,
   converted per grid (3-km blur, 42-km window, 25-km smoothing, 10-km/5-min clamp).
2. **Nearest-scan handover.** With km-scaled flow the mid-gap source switch still
   flipped 45.7% of wet cells (I1->I2) vs 29-32% for scan steps. Fix: motion-aligned
   morph — advect BOTH scans to the intermediate time, blend the aligned fields
   (weights continuous in f, no handover anywhere). Mid-gap flips: 45.7 -> 19.9%.

Residual boundary flips (~30-38%) are coarse-grid discretization: at 4 km a cell's
size is comparable to its per-frame motion, so mask-level churn has a floor no
interpolation scheme can beat. That resolution argument (plus "intensity reads low"
from 4-km averaging of 1-km sources) drove two product changes the same evening:
the Met Office eight-band colour scale (four flat WMO bands hid the structure), and
**viewport-tiled 1-km serving** — hi-res cube as a memmap npy, 256-px tile sprites
fetched per viewport, block-mean overview npz for low zoom. 2-km frame build
measured at ~40-45 s (niced, shared CPU); 1-km expected 2-3 min/frame — final
resolution choice pends a clean-box measurement.

**Morph completion (same evening):** the linear blend of imperfectly aligned warps
smears support (+45% area) at diluted intensity — perceived as regional "pulsing".
Weak-flow median infill alone did not move it; global gain/trim moved it globally
but not regionally. Final scheme: motion-aligned morph + per-24-cell-block support
trim and wet-mean gain to the f-interpolated endpoints. Measured end state: area
ratio 1.03-1.04, wet-mean ratio 1.01-1.03, interpolant flips below scan flips in
both test regions. Display interpolation is now strictly smoother than raw playback.

## Oresund pulsation (2026-09-01 morning): alternating scan programs at the fill seam

Measured east of Copenhagen: consecutive 5-min SCANS alternated wet area
0.32→2.48→0.65→2.59→0.65→2.05% (median scan-to-scan area change 111% of the level)
while interpolants smoothly bridged — so not an interpolation defect. Root cause:
DMI alternates radar scan programs every 5 minutes; DK volumes exist at every stamp
but their coverage of the box flips 76%→99%→75%→100%, so a band of pixels alternates
own-composite ↔ OPERA fill each frame, and the two sources disagreed in LEVEL.
Fix: gain-match the fill to the own composite in their wet overlap (median ratio,
clipped 0.5–2) before gap-filling — the handover becomes invisible in intensity.
If residual pattern-level pulsing remains, the next lever is coverage hysteresis
(only downgrade a pixel to fill after two consecutive own-NaN scans).
