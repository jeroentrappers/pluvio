# The honest near-term deliverable: the 0–2 h nowcast

**Status: this is what we can defend *now*.** The multimodal + multi-day
"seamless" thesis (docs/seamless_model_plan.md) is the ambition, but it needs
months of accumulated MTG/AIFS depth across enough convective regimes before any
claim about it is honest. The **0–2 h nowcast does not** — 22 months of OPERA are
already deep. So we scope the first real, self-contained result to the nowcast
and hold ourselves to a defensible bar (recommendation #2).

## The claim we are allowed to make

> On the held-out (most-recent) OPERA window, does the model beat **optical-flow
> advection** — the operational-grade nowcast baseline — at 0–2 h, and if so at
> which lead × scale × intensity?

What we are **not** allowed to claim:
- "beats persistence" — persistence is the zero-skill floor; optical-flow beats
  it at every lead (see `eval_nowcast`), so beating persistence proves nothing.
- a fine-resolution heavy-rain win read off **CSI alone** — the analysis grid is
  ~7 km (`model.geo.grid_resolution_km()`), which inflates categorical scores;
  the scale-aware **FSS** must agree before any heavy-rain claim stands.
- anything about the 6 h–10 d outlook from this deliverable — that is a separate
  result gated on data depth and verified against **raw AIFS**, not radar.

## How it's measured

`python -m model.eval_nowcast --zarr ./seamless.zarr --ckpt <ckpt>`

Per lead (10-min steps), for **model vs optical-flow vs persistence**:
- MAE and **CRPS** (= MAE for a point forecast; from quantiles for a
  probabilistic checkpoint, rec #3),
- **CSI** at τ = 0.1 / 1 mm/h,
- **FSS** at neighbourhood scales (px → km printed),
- a headline naming the lead(s) where the model overtakes optical-flow on
  CSI@0.1 — or stating plainly that it does not yet (the honest default).

Absolute CSI must sit in the INCA-class public range (≈0.45–0.55 at +30 min,
τ=0.1, at comparable resolution); far above ⇒ leakage or grid inflation, far
below ⇒ a pairing bug (`docs/verification.md`).

## Promotion

A nowcast checkpoint ships to the product (`--producer model` in
`produce_forecast.py`) **only** when it beats optical-flow on CSI **and** FSS at
the leads we serve, on the champion/challenger gate (`plan_overview.md §5`).
Until then the product runs the classical optical-flow⊕AIFS baseline
(`model/classical.py`, rec #4) — which is itself a perfectly respectable nowcast.
The research model is an upgrade, not a dependency.

## Why this is the right scope

It is the regime where (a) the data is already deep, (b) the convective-
initiation channels (lightning/GII/cloud-top cooling) can plausibly add skill
that radar extrapolation lacks, and (c) the win is cleanly verifiable against a
real baseline. A demonstrated 0–2 h CSI/FSS win over optical-flow *driven by
those channels* is a genuine, publishable contribution on its own — and it
doesn't wait on anything.
