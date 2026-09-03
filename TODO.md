# Pluvio — Development Roadmap & Work Breakdown

Goal: the best open and free alternative to Buienradar and the reference in the
market — trained from raw radar and open data, verified in public.

Source of truth for priorities. Status markers: `[ ]` open · `[~]` in progress
· `[x]` done · `[!]` blocked. Owner lanes: `agent` (parallel worker + review),
`ops` (touches servers — one person at a time), `research` (needs GPU/data).

Ground rules for every item
- Acceptance criteria are tests or measurements, never "looks right".
- Nothing ships to serving without the frozen benchmark (3.2) once it exists.
- No geometry without a `Grid` contract (1.1) once it exists.
- Commits are plain human authorship, no tool attribution.

---

## Epic 1 — Rigor & data contracts (days 1–30)

Why: three defects a human caught by eye in five minutes each (reversed
advection, a 50 km projection stretch, cross-fade instead of motion) lived
below every metric. Make eyes unnecessary.

- [~] **1.1 `Grid` data contract** — `research/model/grid.py`: CRS, bounds,
      shape, row order; serialised into zarr attrs by every store builder and
      read (never assumed) by every consumer (`build_store_v3`, `infer_latest`,
      `zarr_dataset`, composite producer, backend cache). Acceptance: a store
      missing/inconsistent attrs fails loudly; all producers write them.
      Lane: agent → ops for deploy. Depends: —
- [~] **1.2 Geometry & warp unit tests** — `research/tests/`: grid_latlon
      south edge equals the 700/765 trim of `_lib._resample`; `morph._warp`
      sign convention (content moves by +D); `morph_pair` centroid and
      intensity preservation; `zarr_dataset` derives grid from store.
      Acceptance: pytest green locally and in CI. Lane: agent.
- [~] **1.3 Fiducial round-trip test** — synthetic delta at known lat/lon
      through store → `build_input` → reprojection → render must land within
      one cell. Acceptance: test in CI; would have caught trim + aux + sign.
      Lane: agent. Depends: 1.5
- [~] **1.4 Store contract checks as a library** — factor `qc_inputs` /
      `qc_watchdog` into `research/tools/qc/` with one JSON verdict format;
      calibrate range checks to the store's normalised units (alaro 0–255,
      msg_ir108, aws_*); synthetic fixtures in CI. Acceptance: hourly run
      has zero false warnings for 7 days while still flagging the dead sst.
      Lane: agent + ops.
- [~] **1.5 Research test infrastructure** — `research/requirements-dev.txt`,
      `uv` venv recipe, `pytest` layout, `.github/workflows/research-tests.yml`.
      Acceptance: CI runs research tests on every push. Lane: agent.
- [~] **1.6 Input unit fixes** — `alaro_precip` scale (0–255 → mm/h?),
      `msg_ir108` units documented; `sst`: feed alive (OSTIA D+2 06:00 lag),
      channel was NaN store-wide because the 36 h max-age preceded publication
      → max-age 96 h deployed (3e82e0d); one-off SST backfill for existing
      issues still needed. Acceptance: QC range checks pass with documented
      units; affected channels corrected in the store. Lane: ops + research.
- [x] **1.7 Archive retention audit** — audited 2026-09-03: `qpe_archive.py
      --prune` deletes only raw volumes/caches (coverage-guarded); day-zarrs are
      kept forever by design. The QPE archive is simply young (started
      2026-08-31) — composite truth exists for days, RAC carries the history.
      Follow-up: write the retention classes into a manifest (raw 3 d, OPERA
      7 d, composite/QPE/forecast archives forever). Lane: ops.
- [ ] **1.8 Operations as declared state** — every recurring job a systemd
      unit generated from one manifest (append+infer cron, producers, QC,
      archives); asusprime training under a persistent job runner (supervisor
      loop is the prototype); laptop is a client, never an orchestrator.
      Acceptance: `systemctl list-timers` is the complete schedule; no crontab.
      Lane: ops.
- [ ] **1.9 Serving box → full Benelux** — once v3 converges: `infer_latest`
      bounds/shape from the store `Grid`, backend `DEFAULT_GRID`/bounds, web
      forecast domain, Lagrangian blend crop. NOTE (review of 1.1): the
      backend nowcast path never reads the npz `bounds` and hard-crashes on
      any rates shape ≠ DEFAULT_GRID_SHAPE — backend must land before or with
      the `infer_latest` switch; painters use EDGE bounds, Grid.bounds are
      cell-centre bounds → use Grid.edge_bounds(). Acceptance: forecast
      overlay covers NL; fiducial round-trip passes on the new box. Lane: ops.
      Depends: v3 convergence, 1.1

- [ ] **1.10 `geo.bbox()` over-claims the stereographic domain** — the legacy
      analysis grid is not a lat/lon rectangle (south row varies 0.475° W→E);
      `bbox()` returns the corner envelope. Audit every caller (WMS GetMap,
      overlay bounds, Flutter `Env.radarBounds*`) — same blast radius as the
      trim bug. Lane: agent. Found by review of 1.2.
- [ ] **1.11 Env-latched geometry** — `geo.GRID` resolves `PLUVIO_GRID_N` at
      import and `grid_latlon` caches the bias env inside `lru_cache`; later
      changes are silently ignored (mechanism of the 192² incident). Read env
      into named constants at module scope, log resolved values once, or pass
      the bias explicitly. Lane: agent.
- [ ] **1.12 Store builders fail loudly** — `build_store_v3` dispatches the
      trimmed/untrimmed mapping by ndim + one name (any new 3-D array silently
      gets the aux extent) → explicit name→extent table with a hard raise;
      `zarr_dataset._discover` silently drops mis-shaped statics and admits any
      n-length 3-D array as input → raise on shape-contract mismatch, log the
      resolved channel list, allow an expected-channel-count assert; assert
      `issue_time` units (seconds) at store creation. Lane: agent.
- [ ] **1.13 Backend pixel conventions** — `cache.GridSpec.latlon_to_cell`
      (centre, `*(h-1)`, `round`) vs `history.py` and `colormap.draw_fiducials`
      (edge, `*h`, `int`) disagree by up to a cell; unify on Grid semantics
      (centre bounds from the store, `edge_bounds()` for painters). Depends 1.1.
      Lane: agent + ops.
- [ ] **1.14 Dead code / small debts** — `morph.py` unused `gy`; `zarr_dataset`
      unused `src`; `torch` missing from `research/pyproject.toml` deps;
      `research/tests` needs a pytest path config. Lane: agent.

## Epic 2 — Model objective & inputs (days 31–60)

Why: a deterministic Huber loss rewards hedging; fields blur with lead and
CSI decays. The objective is the biggest lever we own.

- [~] **2.1 Loss upgrade** — `model/losses.py`: exceedance-FSS (port from
      `train_seamless`), gradient/spectral sharpness; `train.py` flags
      `--fss-weight --fss-thresholds --sharpness-weight` (defaults 0 = today's
      behaviour). Acceptance: unit tests on synthetic fields; an A/B epoch on
      the v3 store shows higher CSI@1 at equal RMSE. Lane: agent → research.
- [ ] **2.2 Probabilistic head** — quantile (0.1/0.5/0.9) or small ensemble
      with CRPS; serving carries P(rain>thr) per lead. Acceptance: reliability
      diagram in benchmark; sharper median than deterministic baseline.
      Lane: research. Depends: 2.1, 3.2
- [ ] **2.3 Lagrangian input channels** — advected latest observation at each
      lead as model input (port `add_nowcast_channels` to the v3 store, using
      `morph` flow or pysteps). Acceptance: ablation on benchmark. Lane: agent
      → research.
- [ ] **2.4 5-minute issue densification** — `build_store_v3` at 5-min issues
      (×12 samples) from RAC 5-min frames. Acceptance: store contract passes;
      learning curve vs 30-min store. Lane: research.
- [ ] **2.5 Temporal encoder** — ConvGRU / axial-attention over the history
      stack + UNet decoder, AMP + checkpointing at 192². Acceptance: benchmark
      win over 2.1 model. Lane: research (rented GPU).
- [ ] **2.6 Pre-rendered training shards** — one-time render of samples to
      sharded tensors; streaming loader. Acceptance: ≥3× epoch speedup, same
      loss curve. Lane: agent.

## Epic 3 — Evaluation institution (days 1–60)

Why: every skill claim this week had to be re-derived because runs were scored
on different rulers. Transparency is the only credible route to "reference".

- [~] **3.1 Frozen benchmark definition** — `research/benchmark/benchmark.yaml`:
      one validation season + curated convective/frontal days; fixed
      thresholds (0.1/0.5/1/2/5 mm/h) and leads. Lane: agent.
- [~] **3.2 Benchmark scorer** — `research/tools/benchmark.py`: CSI/POD/FAR,
      FSS (neighbourhoods), RMSE, bias, reliability (probabilistic) per
      lead/threshold for any list of (name, checkpoint|baseline); baselines
      persistence + advection; JSON + markdown table. Acceptance: unit tests on
      synthetic fields; reproduces this week's tables. Lane: agent.
- [~] **3.3 External baselines** — Buienradar point forecasts (raintext API)
      sampled at stations, UKMO nowcast over the UK box, OPERA where
      obtainable; archived alongside our runs. Acceptance: nightly rows in the
      scoreboard. Lane: agent + ops.
- [ ] **3.4 Public scoreboard page** — nightly job over the Verify archive →
      static page (per lead, per model, yesterday vs what fell). Lane: agent.
- [ ] **3.5 Deployment gate** — checkpoint swap requires benchmark win + a
      canary hour where old and new fields are both archived and diffed.
      Lane: ops. Depends: 3.2

## Epic 4 — Domain & latency (days 61–90)

- [ ] **4.1 Low-latency inference path** — infer every 5 min on radar-complete
      issues with aux carried forward; QC reports issue-age budget.
      Acceptance: median issue age < 10 min. Lane: ops + research.
- [ ] **4.2 Patch-based continental training** — random 192² patches over the
      wide composite; radar + motion + statics inputs, aux masked where absent.
      Lane: research. Depends: 1.1, wide archive depth.
- [ ] **4.3 Tiled continental inference + serving** — same weights over the
      whole composite domain. Lane: ops + research. Depends: 4.2

## Epic 5 — Product: parity with Buienradar, then past it (days 61–90)

- [ ] **5.1 Point-forecast precompute + push alerts** ("rain at your location
      in 20 min"). Lane: agent + ops.
- [ ] **5.2 Per-location narrative** with honest confidence across bands.
- [ ] **5.3 Precipitation type (snow/hail) and lightning** (runbook exists).
- [ ] **5.4 Public API, embeddable widget, share links.**

---

## Currently running
- v3 training (full Benelux 192², RAC truth, healed aux) on asusprime under
  `train_supervisor.sh`; ~47 min/epoch, patience 30.
- Hourly `qc_inputs` + `pluvio-qc` timers on hetz1.

## Agent workflow
Workers (Sonnet) implement one WBS item each in an isolated worktree branch,
run its tests, and report diff + results. A reviewer (Opus) reviews the diff
against the acceptance criteria before merge. Nothing agent-authored touches
servers; deploys are a separate `ops` step.
