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

- [x] **1.1 `Grid` data contract** (merged 2026-09-03; adoption in zarr_dataset/train/backend tracked under 1.9/1.12) — `research/model/grid.py`: CRS, bounds,
      shape, row order; serialised into zarr attrs by every store builder and
      read (never assumed) by every consumer (`build_store_v3`, `infer_latest`,
      `zarr_dataset`, composite producer, backend cache). Acceptance: a store
      missing/inconsistent attrs fails loudly; all producers write them.
      Lane: agent → ops for deploy. Depends: —
- [x] **1.2 Geometry & warp unit tests** — `research/tests/`: grid_latlon
      south edge equals the 700/765 trim of `_lib._resample`; `morph._warp`
      sign convention (content moves by +D); `morph_pair` centroid and
      intensity preservation; `zarr_dataset` derives grid from store.
      Merged 2a43652; mutation-tested (aux trim, fy/fx swap, NaN target
      exclusion each fail exactly one test). Lane: agent.
- [x] **1.3 Fiducial round-trip test** — synthetic delta at known lat/lon
      through store → `build_input` → reprojection → render must land within
      one cell. Merged 2a43652; pick cell moved to the south/east edge so the
      painter's half-cell (edge vs centre) bug is not hidden by truncation.
      Lane: agent.
- [x] **1.4 Store contract checks as a library** — `research/tools/qc/`
      (`verdict`, `checks`, `thresholds`) wrapping both CLIs; AWS bands
      derived from `build_aux.AWS_CHANNELS`, percentile (p0.1/p99.9) range
      check, additive `verdict` key in the JSON. Merged 1495442, deployed to
      hetz1 2026-09-03 (whole package). First live verdict: `warn` — sst 93 %
      NaN (1.6), alaro_precip signed corr 0.03, msg_ir108 −0.065 (both under
      the 0.05 alignment floor; investigate in 1.6). Acceptance still open:
      zero false warnings for 7 days. Lane: agent + ops.
- [x] **1.5 Research test infrastructure** — `research/requirements-dev.txt`
      (torch from the CPU index), `research/.venv` via `uv`, fixture conftest
      + `_store_spec`, `.github/workflows/research-tests.yml` (pytest + ruff).
      Merged 2a43652; 102 tests green from main. Lane: agent.
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
- [x] **1.11 Env-latched geometry** — `grid_latlon(bias=...)` explicit with a
      per-call env fallback; memoisation keyed on the resolved (shape, bias);
      `log_resolved_geometry()` called by the CLIs after logging is configured.
      Bit-identical to the previous output (sha256-checked, default and
      non-zero bias). Merged 2026-09-03 (branch 1cc2999). Lane: agent.
- [x] **1.12 Store builders fail loudly** — `build_store_v3` uses an explicit
      name→extent table (validated at `--create` too); `zarr_dataset._discover`
      raises on any shape-contract mismatch naming array and shapes, logs the
      channel list once, optional `expected_channels`; `issue_time` must be
      epoch seconds (ms → raise, a low outlier slot → warning, so an
      interrupted append cannot take down inference). Proven safe against the
      live legacy store layout and the v3 layout (byte-identical channel lists
      through train and infer paths). Merged 2026-09-03. Lane: agent.
- [ ] **1.13 Backend pixel conventions** — `cache.GridSpec.latlon_to_cell`
      (centre, `*(h-1)`, `round`) vs `history.py` and `colormap.draw_fiducials`
      (edge, `*h`, `int`) disagree by up to a cell; unify on Grid semantics
      (centre bounds from the store, `edge_bounds()` for painters). Depends 1.1.
      Lane: agent + ops.
- [x] **1.14 Dead code / small debts** — `morph.py` unused `gy` (with 2.7),
      `zarr_dataset` unused `src`, `torch` in `research/pyproject.toml`,
      `[tool.pytest.ini_options]` replaces the conftest sys.path hack; `cv2`
      import made lazy in `build_store_v3`. Merged 2026-09-03. Lane: agent.

## Epic 2 — Model objective & inputs (days 31–60)

Why: a deterministic Huber loss rewards hedging; fields blur with lead and
CSI decays. The objective is the biggest lever we own.

- [~] **2.1 Loss upgrade** (code merged 2026-09-03, A/B epoch pending) — `model/losses.py`: exceedance-FSS (port from
      `train_seamless`), gradient/spectral sharpness; `train.py` flags
      `--fss-weight --fss-thresholds --sharpness-weight` (defaults 0 = today's
      behaviour). Acceptance: unit tests on synthetic fields; an A/B epoch on
      the v3 store shows higher CSI@1 at equal RMSE. Lane: agent → research.
- [ ] **2.1b Loss follow-ups (review notes)** — cap `--sharpness-weight`
      (noise injection is locally rewarded up to parity; crossover moves right
      above ~0.3 — re-measure before any higher weight); replace the Python
      branch on `target_e <= dry_floor` with a tensor mask (host sync, blocks
      graph capture on the non-default path); fix docstring literals (canvas
      size unstated); drop the dead re-export shim in train.py. Lane: agent.
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

- [x] **2.7 Better motion estimator** — `research/model/motion.py` is the one
      NCC block-flow estimator (mean-subtracted, std-normalised over wet
      cells, grid-derived search radius, parabolic sub-pixel refinement);
      `tools/_advection.py` wraps it and the backend `morph.py` carries a
      lockstep copy (asserted equal by test; backend image ships only
      backend/src). Measured on realistic synthetic scenes: true-motion error
      7.17 → 1.28 px, morph RMSE 0.54 → 0.06 mm/h; the old scorer saturated at
      the ±7 px radius on low-contrast tails. Merged b03b2a6. Open: backend
      image rebuild + deploy to put it on the serving path (ops). Lane: agent.

## Epic 3 — Evaluation institution (days 1–60)

Why: every skill claim this week had to be re-derived because runs were scored
on different rulers. Transparency is the only credible route to "reference".

- [x] **3.1 Frozen benchmark definition** — `research/benchmark/benchmark.yaml`:
      val window from 2026-04-02T21:30Z (after the store's own 80/20 split;
      the scorer refuses earlier starts unless `allow_train_overlap`), case
      days scored in full as their own stratum, thresholds 0.1/0.5/1/2/5 mm/h.
      Merged 19cccf0. Lane: agent.
- [x] **3.2 Benchmark scorer** — `research/tools/benchmark.py` +
      `_advection.py`: CSI/POD/FAR/freq-bias/FSS/RMSE/MAE/mean_error/CRPS per
      lead/threshold, persistence + advection (NCC block flow, radius from the
      finer grid axis) + operational baselines, one validity mask per sample
      shared by every entry, `sample_set_hash`/config/git provenance. Merged
      19cccf0 + follow-ups ece22c9. Open: `reliability` slot is `None` until
      2.2; full 2000-sample 192² run needs ~7 GB RAM; run it against
      operational/legacy/v2/v3 once v3 converges. Lane: agent.
- [~] **3.3 External baselines** — Buienradar raintext point forecasts at 20
      BE/NL stations: `research/tools/external_baselines.py` merged 3a87a96
      (DST-safe rollover, t0-relative 5-min leads, self-healing JSONL index,
      dtype-agnostic NaN skip; 34 tests). Running on hetz1 since 2026-09-03
      12:05 as `pluvio-external-baselines.timer` (every 5 min, :30 offset),
      archive `/mnt/storagebox/external_baselines/buienradar/YYYY/MM/DD.jsonl`.
      Open: wire `score_against_truth` to the composite truth in the nightly
      run (3.4); UKMO nowcast over the UK box and OPERA where obtainable.
      Acceptance: nightly rows in the scoreboard. Lane: agent + ops.
- [ ] **3.4 Public scoreboard page** — nightly job over the Verify archive →
      static page (per lead, per model, yesterday vs what fell). Lane: agent.
- [x] **3.6 Benchmark statistics** — `research/tools/_stats.py`: per-sample
      sufficient statistics (contingency counts, sum/sum-abs/sum-sq error,
      FSS numerator/denominator per threshold/scale), tagged with issue_time
      so the block bootstrap resamples 6 h blocks of stats rather than
      re-scoring — memory footprint drops, it no longer retains per-sample
      pointwise arrays or FSS field stacks at all. `ci_lo`/`ci_hi` merged
      onto every metric plus a paired-difference CI vs `bootstrap.
      reference_model` (default persistence), for both the pooled and
      case-day strata; adequacy counted in issue-time events (a scored
      target truth field's domain max > `adequacy.threshold_mm_h`, default
      5 mm/h; `adequate: false` below `min_events`, default 30) reported in metadata
      and the markdown table; per-lead stratified sampling was already in
      `_select_samples` (verified with a test); sidecar `<out>.samples.jsonl`
      manifest, `sample_set_hash` now hashes that manifest. Lane: agent.
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
