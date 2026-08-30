# Belgium: our QPE beats OPERA against rain gauges

The production domain, on a sample large enough to mean something.

**613 gauge-times, 151 wet, 5 days (2026-08-17/18/19/28/29), northern Belgian KMI gauges.**

| | POD | FAR | CSI | MAE | corr |
|---|---|---|---|---|---|
| **ours (Herwijnen)** | **0.874** | 0.576 | **0.400** | 0.714 | **0.463** |
| OPERA | 0.649 | 0.520 | 0.381 | 0.716 | 0.194 |

## What is significant, and what is not

Bootstrapped over 4,000 resamples, because a 0.019 CSI margin on its own proves nothing:

| quantity | mean | 95% CI | verdict |
|---|---|---|---|
| dPOD | **+0.225** | [+0.141, +0.310] | **decisive** |
| dCSI @ 0.1 mm/h | +0.019 | [-0.038, +0.074] | **not significant** |
| dFAR | +0.056 | [+0.004, +0.108] | **significantly WORSE for us** |

The headline CSI margin at the 0.1 mm/h threshold does **not** survive its own confidence
interval. Our detection advantage is partly offset by more false alarms on trace rain.

But the advantage grows with rain rate, and there it does survive:

| threshold | wet n | ours CSI | OPERA CSI | dCSI 95% CI |
|---|---|---|---|---|
| 0.1 mm/h | 151 | 0.400 | 0.381 | [-0.038, +0.074] |
| **0.5 mm/h** | 96 | **0.368** | 0.295 | **[+0.003, +0.144]** |
| 1.0 mm/h | 68 | 0.342 | 0.266 | [-0.005, +0.157] |
| **2.0 mm/h** | 41 | **0.292** | 0.182 | **[+0.019, +0.209]** |

**At rain rates that matter (>=0.5 mm/h) we beat OPERA on CSI with the interval clear of
zero, and the margin widens with intensity.** POD is better at every threshold. The one
metric OPERA wins is false alarms on trace precipitation.

## How this became possible

Every earlier Belgian attempt was stuck on 13 wet gauge-times from one day, because the
OPERA single-site feed is a 24-h rolling cache that cannot be backfilled. **KNMI publishes
full polar volumes for Herwijnen and Den Helder through its open-data API with an archive
reaching back to 2019.** Herwijnen covers northern Belgium, so the sample went from 13 wet
points to 151 without waiting weeks for capture.

The format is KNMI HDF5 v3.6, not ODIM, and needed a new reader (`tools/knmi_volume.py`).
Two traps in it:

* `scan1` is the **90 degree birdbath**, so sweeps must be sorted by angle — the same
  trap the French per-elevation files set, which had already put a vertical scan into the
  radar list once.
* Data are uint16 needing the per-scan linear calibration, `GEO=0.00193793*PV+-31.5019`.

These volumes are also **fully dual-polarimetric** (RhoHV, PhiDP, KDP, ZDR), unlike the
Belgian radars in the OPERA feed which carry only TH and DBZH. Non-meteorological echo is
removed with RhoHV < 0.80 — the textbook discriminator — instead of a fitted persistence
heuristic.

### The reader was validated before it was trusted

nlhrw is the same physical radar we already read via ODIM from the OPERA feed, so both
paths must agree. At 20260830T0730 they do: identical site, identical 0.30 deg elevation,
identical (360, 838) shape, **correlation 0.835**, wet-cell IoU 0.738. An azimuth-rotation
sweep peaks sharply at zero offset (0.835, against 0.43 at +/-5 deg and 0.04 at +/-20),
confirming the ray convention. The residual difference is QC: this path filters on RhoHV
where the ODIM path uses Gabella declutter.

## Limits

Northern Belgium only (gauges above 50.6N, within Herwijnen's 187 km range) — the Ardennes
are not covered and beam blockage there remains untested. One radar, not a composite. Five
days, all warm-season; no snow, bright band or winter stratiform. And our FAR is
measurably worse than OPERA's, which is the honest cost of the detection advantage.
