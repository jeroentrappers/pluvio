# c18 — negative result, and a noise floor

Two arms, both trained to early-stop convergence on the 24-month window:

| arm | channels | loss | leads beating optical flow |
|---|---|---|---|
| **c18-D** | 9 (no static) | exceedance, w=0.3 | **12 / 12** |
| c18-E | 12 (static) | exceedance, w=1.0 | 11 / 12 (fails at lead 20) |

## Neither arm beats the deployed c17-C

| lead | c17-C CSI@0.1 | c18-D CSI@0.1 | c17-C FSS3 | c18-D FSS3 |
|---|---|---|---|---|
| 0 | 0.900 | 0.922 | 0.987 | 0.991 |
| 30 | 0.591 | 0.588 | 0.845 | 0.842 |
| 60 | 0.444 | 0.437 | 0.723 | 0.718 |
| 90 | 0.390 | 0.387 | 0.667 | 0.664 |
| 120 | 0.345 | 0.340 | 0.613 | 0.652 |

c18-D shares c17-C's recipe, so this is close to a reproduction, and it lands
marginally BELOW it at every lead past 0. **c17-C stays deployed; nothing to promote.**

## The useful part: a measured noise floor

Because c18-D is a re-run of the winning recipe, the gap between the two runs is
run-to-run variance, not signal: **~0.005 CSI and ~0.005 FSS3**. Any future arm that
wins by less than about 0.01 CSI has not demonstrably won. This is worth more than the
arms themselves — several earlier decisions were argued over differences of that size.

## Static channels lost again

c17-A already found elevation/landmask/distance-to-coast to be a negative result. c18-E
carried them with a stronger structure term (w=1.0 against 0.3) in case the earlier
arm simply under-weighted the term that would exploit them. It still lost, and lost the
lead-20 gate that the no-static arm passes. Two independent runs now agree: **the static
terrain channels do not help this architecture.** Stop re-testing them without a new
mechanism — a third arm of the same shape would not be informative.

⚠️ w and static were varied TOGETHER between D and E, so strictly this does not separate
"static hurts" from "w=1.0 hurts". The c17-A result is what makes static the likelier
cause; if that mattered for a future decision it would need w held fixed.
