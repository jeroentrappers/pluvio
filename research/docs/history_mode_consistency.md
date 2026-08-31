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
