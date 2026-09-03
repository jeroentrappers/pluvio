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
      cell-centre bounds → use Grid.edge_bounds(). The backend half is done
      (1.13): the nowcast path reads the npz `bounds`/shape and serves a 192²
      artifact on its own footprint. Remaining for 1.9:
      (a) `cache.DEFAULT_BOUNDS`/`DEFAULT_GRID_SHAPE` → the full-Benelux box,
      so point shards, the sprite and `/v1/forecast`'s location check cover it
      (until then `inference_worker` logs and excludes an off-grid band from
      the shards/sprite, and a location outside the legacy box still 400s);
      (b) the web client treats the API's `bounds` as pixel EDGES —
      `web/src/api.ts` `DEFAULT_BOUNDS` (a hardcoded copy of the legacy box),
      `RadarMap.tsx` `maxBounds`/overlay box/`visibleTiles()` tile split — while
      the API serves CENTRE bounds, so it must either inflate by half a cell
      itself or the API must publish an explicit edge-bounds field;
      (c) the same for Flutter `Env.radarBounds*`;
      (d) the mixed-grid transition is not safe to ship as-is: grid.json holds
      ONE footprint, so with a 192² nowcast and a 100² short band the short
      band's overlays are mislabelled by the 192² bounds (~35 km W, ~111 km N)
      — and because point shards/the sprite only cover bands on the cache
      grid, `/v1/forecast` loses leads 0-120 (or 503s outright when the
      nowcast is the off-grid band). Either widen the cache grid in the same
      change as the `infer_latest` switch, or make grid.json/overlay URLs
      per-band. Acceptance: forecast
      overlay covers NL; fiducial round-trip passes on the new box. Lane: ops.
      Depends: v3 convergence, 1.1, 1.13

- [x] **1.10 `geo.bbox()` over-claims the stereographic domain** — the legacy
      analysis grid is not a lat/lon rectangle (south row varies 0.475° W→E);
      `bbox()` returns the corner envelope. Audit every caller (WMS GetMap,
      overlay bounds, Flutter `Env.radarBounds*`) — same blast radius as the
      trim bug. Lane: agent. Found by review of 1.2.
      DONE 2026-09-03: `bbox()` split into `envelope()` (superset, the old
      behaviour, kept as `bbox()`) and `inner_rectangle()` (guaranteed subset),
      both on `Grid` and `model.geo`, all three taking an explicit `bias` like
      `grid_latlon()`; all 24 consumers audited in
      `research/docs/geometry_audit.md`. Most callers use the box correctly, as
      a fetch/coverage superset. The one real curvature defect is
      `radar_single_site.py --compare-opera` (#10): it scores a naive regular
      lat/lon raster over the envelope against OPERA warped onto the TRUE
      curved grid, so the two sides are misregistered domain-wide — median
      37.1 km, max 120.4 km, 30.7 km at the domain centre, i.e. 4–9 cells
      everywhere (43.8–69.3 km median inside the scored 100 km discs), which
      makes every historical correlation/bias/MAE from that tool
      uninformative, not merely edge-biased. `verify_radar.py` is NOT affected
      (it reprojects OPERA onto the same naive box it bins onto, so it is
      self-consistent). West edge is exactly meridian-aligned by construction
      (lon_0=0); the east extremum is the NE corner, the deficit is at the SE.
      Also fixed here: `radar_single_site.py` set `PLUVIO_GRID_N` after its
      first `model.geo` import, making `--grid-n` a silent no-op. Documented,
      not fixed: `infer_latest` writes cell-CENTRE bounds that the backend
      overlay/GridSpec treat as edges → uniform ~2 km half-cell shift, the only
      live user-visible georeferencing error (belongs to 1.13). Open follow-up:
      rebin `polar_to_grid` onto the true curved grid (nearest-neighbour /
      cKDTree against the real cell centres) so `--compare-opera` measures
      something, then re-run the single-site validation.
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
      Follow-up found 2026-09-03: the live store's `issue_time` is not strictly
      monotonic (35754 issues, no zero slots) — audit for duplicates/out-of-
      order appends (newest-first backfill?) and add a monotonicity check to
      qc_inputs. Also: the hetz1 research checkout is not git — the first
      deploy of the repo's `build_zarr.py` exposed a latent NameError
      (fixed e32148a); diff remote files before overwriting (see 1.8).
- [x] **1.13 Backend pixel conventions** (merged 2026-09-03) — the backend
      now says, in one place, which of the TWO conventions each array carries
      (table in `cache.GridSpec`): forecast npz / zarr store attrs /
      grid.json / `GridSpec.bounds` are CELL-CENTRE envelopes; the observed
      cube (produce_observed's rasterio `from_bounds` raster) and the QPE day
      stores (`bounds_convention="outer_edges"`) are PIXEL EDGES.
      `cache.edge_bounds()`/`GridSpec.edge_bounds()` inflate a centre
      envelope by half a cell; `GridSpec.latlon_to_cell` floors the EDGE-based
      index (same as `Grid.cell_of`'s rounded centre index, except exactly on
      a cell boundary — floor-on-edge is south/east-inclusive) and accepts the
      half-cell margin; `GridSpec.cell_center_latlon` inverts it. Painters:
      `colormap.draw_fiducials` documents that it takes EDGE bounds, and
      `cache._fiducial_bounds` (overlays + sprite) converts — `history.py`'s
      observed-cube painting and point lookups are already edge-referenced and
      stay as they were, with the reason spelled out. `model._lagrangian_blend`
      inflates only its forecast-grid side and now CLAMPS the crop window to
      the observed raster instead of rejecting it: with the target box equal
      to the observed box (every live configuration) the old guard saw c0=-1
      and silently disabled the seam anchor. `model.model_band` returns the
      grid it served on — the npz's own `bounds` (centre; `infer_latest`
      writes `Grid.bounds`, or the BE_* constants on its legacy branch) plus
      its rates shape, falling back to `DEFAULT_BOUNDS` for a legacy npz with
      no `bounds` — and `inference_worker` writes the band, its overlays and
      grid.json on that grid. Tests: `backend/tests/test_gridspec.py`,
      `test_history_points.py`, plus blend-active and 192² cases in
      `test_model_cube.py`, `test_inference_worker.py`, `test_api.py`.
      `verify.py`'s QPE crop is left to the scoreboard branch (attrs-driven
      edge bounds + NaN-aware regrid). Web client untouched — see 1.9.
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
- [x] **2.1b Loss follow-ups (review notes)** — cap `--sharpness-weight`
      (noise injection is locally rewarded up to parity; crossover moves right
      above ~0.3 — re-measure before any higher weight); replace the Python
      branch on `target_e <= dry_floor` with a tensor mask (host sync, blocks
      graph capture on the non-default path); fix docstring literals (canvas
      size unstated); drop the dead re-export shim in train.py. Lane: agent.
- [ ] **2.2 Probabilistic head** — quantile (0.1/0.5/0.9) or small ensemble
      with CRPS; serving carries P(rain>thr) per lead. Acceptance: reliability
      diagram in benchmark; sharper median than deterministic baseline.
      Lane: research. Depends: 2.1, 3.2
- [~] **2.3 Lagrangian input channels** — advected latest observation at each
      lead as model input. Landed (agent half): computed ON THE FLY in
      `ZarrCorrectionDataset`, no store rewrite — the legacy
      `add_nowcast_channels` store layout is NOT reused.
      * `lagrangian_channels=0|1|2` on the dataset, default 0 → `n_channels`
        and every existing channel index are bit-identical to before
        (regression-tested), and the new planes are APPENDED after the
        statics so turning them on never renumbers anything.
      * plane 1 `lagrangian_rate`: the newest analysis advected to the
        sample's lead by `model.motion.block_flow` on the two newest history
        frames, scaled linearly (`lead/history_step`) and clamped to the grid
        by `warp`'s coordinate clip; mm/h, same units as the history planes.
        Search radius from the store's own `bounds`/`grid_n`
        (`motion.km_per_px_from_bounds`, now the single implementation — the
        benchmark's `_km_per_px` delegates to it) so the channel and the
        advection baseline it must beat see the same motion. `subpixel=False`
        for the same reason, and because the parabolic fit adds up to 0.5 px
        of spurious offset on an exact match, which the lead scaling
        multiplies into visible drift.
      * plane 2 `lagrangian_flow_mag` (only at 2): per-step displacement
        magnitude / search radius, ~[0,1], deliberately lead-independent — a
        per-issue "how far was this prior transported / did the estimator
        find motion at all" signal.
      * NaN: filled before the flow estimate; after the warp the NaN mask is
        warped by the same displacement and restored, so a NaN wake is "no
        observation advected here" rather than a fabricated dry cell.
        `build_input`'s own final `nan_to_num` is what turns it into the 0.0
        the net sees — same convention as every other channel.
      * checkpoints carry a `channel_recipe` (history steps/step_min, aux
        list, static list, lagrangian count, total) and `infer_latest`
        rebuilds the input FROM IT (`dataset_for_checkpoint`), cross-checked
        against `in_channels`; a pre-recipe checkpoint resolves to exactly
        today's behaviour. Flags: `train.py --lagrangian-channels {0,1,2}`
        (rejected on the legacy HDF5 dataset, which never goes through
        `build_input`) and `infer_latest.py --lagrangian-channels` as an
        override.
      * cost, measured at 192² (`build_input`, 4 leads/issue, 15 channels,
        CPU): off 6.9 ms/sample; on 51.9 ms/sample walking the index in order
        (one 165 ms flow estimate amortised over an issue's 4 leads);
        7.9 ms/sample with the flow already cached; 172 ms/sample if the
        flow is re-estimated per sample. The flow is cached per issue as the
        4x4 BLOCK field (128 B/issue, so a whole split fits and nothing is
        evicted) — but the cache is per DataLoader worker, so under
        `shuffle=True` with W workers an issue's leads usually land in
        different workers and the epoch cost tends toward the 172 ms figure.
        If that shows up in the profile: group an issue's leads in one batch
        (sampler) or fold the flow into 2.6's pre-rendered shards, where it
        is computed once per issue for good.
      * merged: dataset/train/infer + 21 tests
        (`research/tests/test_lagrangian_channels.py`: channel count,
        bit-identical off-path, known synthetic motion at each lead, zero
        flow == persistence, one-frame history == zero flow, flow estimated
        once per issue, NaN wake, recipe round-trip through
        `dataset_for_checkpoint`).
      Acceptance still OPEN — needs the GPU ablation, not the agent lane:
      train three runs on the frozen benchmark store, identical seed/loss/
      schedule, `--lagrangian-channels 0` vs `1` vs `2`, and score all three
      with `tools/benchmark.py` (same config hash + manifest hash). Report
      per-lead CSI/FSS at 0.5/1/2/5 mm/h and RMSE against the run's own
      `advection` baseline: the channel earns its place only if it beats the
      0-channel run at the longer leads (60-120 min), where the net currently
      has to learn advection implicitly, AND the 0-channel run does not
      already match the advection baseline there. If 2 ≈ 1, ship 1 (one fewer
      channel, and the magnitude plane is the speculative half).
      Lane: research (GPU). Depends: 3.2
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
- [~] **3.4 Public scoreboard page** — `research/tools/scoreboard.py`: nightly
      job over the forecast archive (`forecast`/`nowcast` kinds) vs the QPE
      composite truth, reusing `_stats.SampleStats`/`block_bootstrap` for
      CSI@0.1/1 mm/h, FSS, RMSE, mean_error with CIs; Buienradar station rows
      scored against composite-at-station truth alongside our own forecast at
      those same stations/times (one shared truth sample feeds both, so the
      comparison is like-for-like); one JSON record/day appended to
      `<out_root>/YYYY/MM/DD.json`, static self-contained HTML with a table
      per lead, an events-yesterday adequacy line, and a 30-day trend table.
      33 tests. Review fixes on top of the first cut: the truth georeference
      is read from the day-zarr's own `bounds` attr and NOTHING else — it was
      hardcoded to `produce_forecast.BE_BOUNDS`, the 100² serving box, which
      read Brussels truth 237 km off and squashed the whole 768² composite
      onto the serving box. Deriving it from this repo's `model.geo`/
      `model.grid` is not a safe fallback either: the archiver runs from the
      `/opt/pluvio/radarproc` checkout, whose `model/geo.py` predates the
      700/765 trim and the registration bias, so the stores are on
      `(0.0, 48.8953, 10.8565, 55.9736)` while a fresh derivation returns
      `(0.07, 49.4387, 10.9265, 55.9736)` — 60 km out at the south edge. The
      attr is therefore mandatory (archiver patched in 1b6f023, existing
      day-stores backfilled); a store without it raises `QpeGeometryError`
      naming the path, and `--qpe-bounds` remains only as a loudly-logged
      explicit override. The composite is area-averaged onto each run's grid
      (float64 integral image — a float32 running total over 768² drifts
      ~6e-3 mm/h near the far corner) with insufficiently-covered target cells
      left NaN instead of `nan_to_num`'d to observed-dry. The bootstrap groups
      kinds by issue-time sequence (`forecast` and `nowcast` come from
      separate producers, so one paired draw over both raised `ValueError`)
      and surfaces `ci`/`ci_vs_reference` per row exactly as
      `tools/benchmark.py` does — `ci_vs_reference` only for a kind actually
      drawn together with the reference, `None` otherwise, since an unpaired
      difference interval would not mean what the column says. The point join
      loads the previous UTC day's issues (Buienradar's archive is keyed by
      valid time); truth frames are memoised by (day, slot) and the truth
      lookup is a dict, not a linear scan (~12 min/day of re-reads); slot
      rounding wraps into the next day. Same wrong bounds tuple, same
      mandatory-attr rule and the same regrid fixed in
      `backend/src/pluvio_backend/verify.py` (+12 tests there; its `scores()`
      now counts only observed cells).
      Open: the `pluvio-scoreboard` nightly timer is declared in
      `research/docs/ops_schedule.md` (02:30 UTC, previous day, host paths)
      but NOT yet installed on hetz1 — ops step, nothing left in the tool.
      Lane: ops.
      First real record (2026-09-02, light rain, max 2 mm/h): forecast lead-0
      CSI@0.1 0.13, nowcast lead-0 CSI 0.115 with FAR 0.79; lead-0 correlation
      between the KNMI-derived nowcast field and our composite only 0.07 (no
      integer shift explains it; forecast kind 0.26). Open: validate on a wet
      day, cross-check against qc_inputs' registration fit, and confirm the
      composite slot/units before quoting numbers. Timer installed 02:30 UTC.
      Buienradar point rows start 2026-09-03 (n_matched 0 for 09-02 is expected).
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
- v3 training STOPPED by hand 2026-09-03 16:50 after epoch 12: best val RMSE
  0.6956 at epoch 4 (`~/pluvio_v2/checkpoints/v3_192.pt`, 33 ch), then
  0.71/0.72/…/0.78 while train loss kept falling; LR already halved by the
  plateau scheduler. Diagnose before the next run (val split, dead `sst`
  channel in the v3 store, weak alaro/msg alignment per QC). GPU is free for
  the 2.1 loss A/B and the benchmark of v3@4 vs v2 vs operational.
- Hourly `qc_inputs` + `pluvio-qc` timers on hetz1.

## Agent workflow
Workers (Sonnet) implement one WBS item each in an isolated worktree branch,
run its tests, and report diff + results. A reviewer (Opus) reviews the diff
against the acceptance criteria before merge. Nothing agent-authored touches
servers; deploys are a separate `ops` step.
