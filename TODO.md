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
      Deployed to hetz1 2026-09-03 21:08 UTC (GHCR image from 0888519 after the
      backend CI gate was made green; api/worker recreated, overlay + point
      endpoints 200, seam blend active on the live 100² grid).
- [x] **1.14 Dead code / small debts** — `morph.py` unused `gy` (with 2.7),
      `zarr_dataset` unused `src`, `torch` in `research/pyproject.toml`,
      `[tool.pytest.ini_options]` replaces the conftest sys.path hack; `cv2`
      import made lazy in `build_store_v3`. Merged 2026-09-03. Lane: agent.

## Epic 2 — Model objective & inputs (days 31–60)

Why: a deterministic Huber loss rewards hedging; fields blur with lead and
CSI decays. The objective is the biggest lever we own.

- [x] **2.1 Loss upgrade** — `model/losses.py` exceedance-FSS + sharpness terms
      behind zero-default flags (code 2026-09-03), A/B measured 2026-09-04 on
      the frozen benchmark (2000 samples, 682 events, same manifest): arm B
      (`--fss-weight 0.5 --sharpness-weight 0.05`, epoch 2, val RMSE 0.6822)
      vs the plain-Huber v3 (epoch 4, 0.6956). CSI@1 mm/h equal within the
      90 % CIs at every lead (0.457/0.341/0.257/0.194 vs 0.446/0.334/0.264/
      0.199); FSS 3 px higher at every lead (+0.04..+0.06: 0.766/0.650/0.558/
      0.459 vs 0.729/0.591/0.495/0.398); CSI@0.1 much higher at 60–120 min
      (0.421/0.354/0.321 vs 0.377/0.313/0.202); RMSE equal or lower; wet bias
      halved. Against the KNMI operational nowcast: CSI@1 ties at 30/60 and
      wins at 90/120; FSS3 wins at every lead; CSI@0.1 still loses (0.54 vs
      0.61 at 30 min). Baseline training also diverged after epoch 4 while
      arm B held 0.68–0.70 through epoch 10. Verdict: adopt the structure
      loss; the run continues under patience 30 and gets re-benchmarked at its
      best epoch. Results: `research/benchmark/results/2026-09-04_v3_huber_vs_fss.md`.
      Lane: agent → research.
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
      * downstream-edge caveat (found in review, fixed here): `block_flow`
        skips any candidate offset whose window would leave the array, so a
        block on the grid edge can only search INWARD — it is not flagged
        invalid, it just reports ~zero. Measured on 192² with a uniform
        (+3,+3): 7 of 16 blocks wrong, the upsampled field decaying to zero
        from row 168 on (~60 % of the area under-advected), and plane 2
        decaying with it. `zarr_dataset.repair_edge_flow` now replaces every
        block that could not measure the motion — fenced in by the edge
        (`free_blocks`) or too dry (`valid=False`) — with the per-component
        median of the blocks that could. With BLOCKS=4 and a search radius
        near the block size that is the 2x2 interior, so the field is close
        to uniform: the edge bands trade spatial detail they never had for
        the right displacement. `motion.py` keeps its estimator untouched —
        the only change there is the extraction of `km_per_px_from_bounds`
        (the benchmark's `_km_per_px` now delegates to it), and the file is
        still hand-synced with the backend copy. If the estimator ever
        learns to search outward, delete `repair_edge_flow`;
        `test_raw_block_flow_still_needs_the_repair` fails when that day
        comes. NOTE: the benchmark's own `advection`
        baseline still has the unrepaired behaviour, so the ablation compares
        the channel against a slightly weaker advection than it carries.
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
      * 2.6 integration (the intended path): `render_shards
        --lagrangian-channels {0,1,2}` bakes the planes into the shards, so
        the flow is estimated once per issue for good; the count is part of
        the shard RECIPE (`shard_dataset.RECIPE_KEYS`), so `--shards
        --lagrangian-channels 2` against shards rendered without them is a
        named mismatch instead of a run that silently trains on 33 channels.
        A shard-trained checkpoint carries the same `channel_recipe` as a
        zarr-trained one — true since the 2.6 follow-ups, where train.py
        stopped hand-building a `channel_names`-less subset on the shard
        path and reads the manifest's own copy instead (asserted by
        `test_shard_and_zarr_trained_checkpoints_carry_the_same_channel_recipe`).
        `render_shards`' `_channel_recipe` is gone —
        `ZarrCorrectionDataset.channel_recipe()` (now including
        `channel_names`) is the single source, in the manifest and the
        checkpoint alike. Adding the recipe key invalidates any shard store
        rendered before this change: it fails loudly naming
        `lagrangian_channels`, and needs a re-render.
      * cost, measured at 192² (`build_input`, 4 leads/issue, 15 channels,
        CPU): off 5.8 ms/sample; on 50.2 ms/sample walking the index in order
        (one 166 ms flow estimate amortised over an issue's 4 leads);
        7.3 ms/sample with the flow already cached; 166 ms/sample if the
        flow is re-estimated per sample. The flow is cached per FRAME PAIR
        as the 4x4 BLOCK field (128 B, so a whole split fits and nothing is
        evicted) — but the cache is per DataLoader worker, so under
        `shuffle=True` an issue's four leads usually land in different
        workers and almost every sample pays. Projected added loader time per
        epoch on the real store (113k samples / ~28k issues, 6 workers,
        166 ms/flow):
          - shards rendered with the planes: ZERO per epoch (~77 min of
            render CPU once, parallel over `--workers`) — do this;
          - `--zarr` with an issue-grouped sampler (an issue's leads
            consecutive in one worker): ~13 min/epoch;
          - `--zarr` as-is (shuffled, no `persistent_workers`): ~52 min/epoch,
            i.e. it roughly doubles the 47 min/epoch the zarr path already
            costs. (The review's own arithmetic put these at ~11 and ~34 min
            on slightly different assumptions; same ordering, same
            conclusion. Neither is measured on the box yet.)
      * merged: dataset/train/infer/render_shards + 33 tests in
        `research/tests/test_lagrangian_channels.py` (channel count,
        bit-identical off-path, channel names, known synthetic motion at each
        lead, all 16 blocks correct for (+3,+3)/(−3,+3)/(0,+4) after the edge
        repair, no stationary downstream band at lead 90, the unrepaired
        estimator's failure documented, `free_blocks`/`repair_edge_flow`
        units, zero flow == persistence, one-frame history == zero flow, flow
        estimated once per frame pair, cache keyed on the pair, NaN wake,
        recipe round-trip and renamed-static rejection through
        `dataset_for_checkpoint`) + 5 in `test_shard_dataset.py` and 2 in
        `test_train_cli.py`.
      Acceptance still OPEN — needs the GPU ablation, not the agent lane:
      train three runs on the frozen benchmark store, identical seed/loss/
      schedule, `--lagrangian-channels 0` vs `1` vs `2`, and score all three
      with `tools/benchmark.py` (same config hash + manifest hash). Report
      per-lead CSI/FSS at 0.5/1/2/5 mm/h and RMSE against the run's own
      `advection` baseline: the channel earns its place only if it beats the
      0-channel run at the longer leads (60-120 min), where the net currently
      has to learn advection implicitly, AND the 0-channel run does not
      already match the advection baseline there. Render the shards once with
      `--lagrangian-channels 2` and train the 1-channel arm from a second
      render rather than paying the flow per epoch. Decide 1 vs 2 on measured
      skill only: both planes come out of the same flow estimate, so plane 2
      costs nothing at render time — the old "ship 1 to save the compute"
      argument does not apply. Ship 1 if 2 shows no gain (one fewer channel,
      and the magnitude plane is the speculative half); ship 2 if it does.
      Lane: research (GPU). Depends: 3.2, 2.6
- [ ] **2.4 5-minute issue densification** — `build_store_v3` at 5-min issues
      (×12 samples) from RAC 5-min frames. Acceptance: store contract passes;
      learning curve vs 30-min store. Lane: research.
- [ ] **2.5 Temporal encoder** — ConvGRU / axial-attention over the history
      stack + UNet decoder, AMP + checkpointing at 192². Acceptance: benchmark
      win over 2.1 model. Lane: research (rented GPU).
- [x] **2.6 Pre-rendered training shards** — `research/tools/render_shards.py`
      renders the `ZarrCorrectionDataset` sample set once into `.npy` memmaps
      (issue-aligned shards, float16 by default, resumable, manifest carries
      the layout + channel recipe + grid + sample count + source-store hash +
      per-shard sha256); `research/model/shard_dataset.py` (`ShardDataset`)
      streams them; `model/train.py --shards <dir>` swaps the dataset. Same
      index, same chronological split boundary, same order, same targets →
      same loss curve; the float16 cast happens once at render time
      (`cast_for_shard`) and the tests assert bit-for-bit equality with that
      same cast applied to the zarr side (`--dtype float32` is bit-equal
      un-quantised). Measured on a real-shape synthetic store (192², 33 ch,
      CPU, 1 worker): 41.5 → 14,737 samples/s. Since 2.3,
      `--lagrangian-channels {0,1,2}` bakes the advected-observation planes in
      too (the recipe records the count, so a mismatched train run is refused)
      — that is where the per-issue flow estimate stops being a per-epoch cost.
      Review follow-ups now merged:
      * **per-issue dedup is the primary layout** (`--layout dedup`, default).
        Flat did not fit the box: asusprime's `/home` was 194 G at the c15
        relay against 332 GiB for both splits. 29 of the 33 channels are a
        function of the ISSUE — the history stack, the aux planes, the statics
        (and `lagrangian_flow_mag`, which is deliberately lead-independent);
        only `nowcast_at_lead` / `lead_over_120` / `tod_sin` / `tod_cos` (and
        `lagrangian_rate`) vary per lead. Storing the invariant block once per
        issue: **2.391 → 0.861 MiB/sample, 331.6 → 119.4 GiB** for
        113,680+28,344 samples at 192²×33 ch (35 ch with
        `--lagrangian-channels 2`: 2.531 → 0.949 MiB/sample, 351.1 → 131.7
        GiB; 2.78x at 33 ch, 2.67x at 35). Both figures assume 4 leads/issue —
        dedup's per-sample cost is `inv/leads_per_issue + var`, so a
        `--require-rain-fraction` split that drops individual leads pays more
        per sample (at 1 lead/issue it is no better than flat). And the win is
        FOOTPRINT, not read bandwidth: `train.py` shuffles, so each sample
        pulls its own per-issue block and reads about what flat reads — an
        issue-grouped sampler (the one 2.3 also wants for the `--zarr` flow
        cache) is what would make it a bandwidth win.
        `ShardDataset` reassembles in `build_input` channel order and the
        sample is bit-for-bit the flat one — the equality tests now run against
        dedup by default, plus a flat-vs-dedup identity test. The split comes
        from `channel_names()` via
        `zarr_dataset.lead_varying_channel_indices`, never hard-coded, and the
        renderer VERIFIES it per issue (builds every lead, refuses to write if
        a store-once channel is not identical across them). `--layout flat`
        stays available; a manifest with no `layout` key reads as flat.
      * **resume validates the source store, not just the recipe.** The recipe
        cannot see a store rebuilt IN PLACE (same arrays, shapes, attrs,
        `issue_time`, different values): the resume re-rendered only the
        missing shards from the new numbers, `index.npy` was rewritten from the
        new index on top, and `ShardDataset` accepted the mixture silently.
        `source_store_hash` is now structural **+ sampled content** (hash of
        `radar[i,0]`, `truth[i]` and one aux at 64 evenly spaced issue indices,
        `hash_mode: "structural+sampled"` — seconds on the real store); a
        resume refuses on a difference and points at `--force`, and
        `train.py --shards --zarr` compares it too.
      * **resume refuses a re-cut shard plan.** `--samples-per-shard` is
        deliberately not in RECIPE_KEYS (it does not change what a sample
        means), and the kept-shard guard compared only `n_samples` — so a
        partial render at `--samples-per-shard 2` resumed at 3 kept shard 3
        (`first_sample` 6) at offset 12: 3 of 81 samples silently wrong,
        manifest `complete`, `--verify` clean. The guard now requires the
        `first_sample` OFFSET to match as well, and a changed
        `samples_per_shard` refuses the resume outright.
      * layout is inferred from the shard entries (`inv` present → dedup), not
        from the top-level key alone, so a dedup manifest that lost its
        `layout` no longer reads as flat and hand out `(n_var, H, W)` arrays
        against a 33-channel recipe; a manifest whose key contradicts its
        entries is refused, and every shard's channel counts are asserted on
        first open.
      * the source fingerprint hashes `radar[i]` — every lead, one chunk, no
        extra I/O — not just `radar[i, 0]`: probing the analysis alone left a
        rebuild of the `nowcast_at_lead` leads invisible.
      * an empty index refuses instead of writing a `complete` manifest with
        zero shards.
      * `index.npy` is REQUIRED by the loader (it used to fall back to zeros —
        a silently wrong stratification, and a wrong issue→row mapping under
        dedup) and is written tmp+rename; `--force` unlinks every
        `*.npy`/`*.npy.tmp` the new plan does not name; a bumped
        `NORMALISE_VERSION` is refused with or without `--zarr` (it is a
        constant of the build, so the guarantee no longer depends on the flag);
        the checkpoint records
        `{"shards": {"root", "recipe_hash", "source_store"}}`; the
        stale-manifest error lists the recipe keys missing/added vs
        `RECIPE_KEYS`, so a pre-2.3 store is recognisable as one.
      * docs corrected: the float16 rationale was wrong — `_normalise` runs
        float16 arithmetic, so the aux/SST/static channels and `lead/120` are
        float16-EXACT; only `tod_sin`/`tod_cos` (≤2.4e-4) and the Lagrangian
        planes quantise.
      61 tests in `research/tests/test_shard_dataset.py`. CLI, storage table
      and the asusprime `--lagrangian-channels 2` invocation in
      `research/docs/training_run_v2.md`. Open: run it on the real store to
      confirm the ≥3× epoch gate (needs the GPU box). Lane: agent.

- [x] **2.7 Better motion estimator** — `research/model/motion.py` is the one
      NCC block-flow estimator (mean-subtracted, std-normalised over wet
      cells, grid-derived search radius, parabolic sub-pixel refinement);
      `tools/_advection.py` wraps it and the backend `morph.py` carries a
      lockstep copy (asserted equal by test; backend image ships only
      backend/src). Measured on realistic synthetic scenes: true-motion error
      7.17 → 1.28 px, morph RMSE 0.54 → 0.06 mm/h; the old scorer saturated at
      the ±7 px radius on low-contrast tails. Merged b03b2a6. Open: backend
      image rebuild + deploy to put it on the serving path (ops). Lane: agent.

- [~] **2.8 Motion-consistent radar→NWP handoff** — measured 2026-09-03 on the
      09:00Z run: the nowcast band advects cells east (wet centroid col 83→96
      by 60 min; KNMI analysis history and the operational nowcast agree on the
      direction), but from 2 h on the rain "comes back from the east": the
      2–6 h blend in `classical.py` is a POINTWISE cross-fade
      `w·radar_extrapolation + (1−w)·NWP` (cosine taper), the NWP (AIFS) rain
      sits 10–20 columns west of and later than the radar cells, and the
      backend cross-fades (not morphs) the hourly short/medium keyframes with a
      122→180 min gap — so the blend weight shifting from the exiting radar
      cells to the western NWP field is rendered as cells travelling west.
      Fix: phase-correct the NWP field onto the radar-consistent frame (NCC
      flow, as `fallback-phasecorr` already does) and blend intensities in that
      moving frame; morph, not fade, between later keyframes (or end the
      animation at the last radar-consistent frame and show the NWP outlook as
      a separately labelled regime); benchmark the handoff on the 2–6 h leads
      (scoreboard forecast kind, CSI≈0 there today). Lane: research + agent.

      DONE 2026-09-04 (producer side): `classical.seamless_cube(phase_correct=True)`
      measures the radar-vs-NWP phase offset at the nowcast horizon (FFT phase
      correlation, refused when either field is dry or the shift exceeds a
      quarter grid), shifts the NWP onto the radar frame through the whole
      blend window and relaxes the shift linearly over the next 12 h of the
      outlook; the offset is recorded in the cube (`phase_offset_px`) and
      logged by the producer. Tests reproduce the reversal with the old fade
      and its absence with the correction (test_classical_handoff.py). Open:
      rebuild + deploy the producer image; backend still cross-fades the
      hourly short/medium keyframes (morph or hold instead); evaluate the 2-6 h
      leads on the scoreboard over the coming days.
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
- [~] **3.7 Buienradar EU composite + forecast archive** —
      `research/tools/buienradar_eu.py` (+ 95 offline tests): archives the
      Europe rain-radar composite as a continuous linear history and every
      forecast run separately, so our nowcasts can be scored against theirs
      on the same frames. Collected with Buienradar's written permission;
      named User-Agent, one fetch per frame ever, 0.2 s spacing, backoff.
      `collect` writes `composite/YYYY/MM/DD/<YYYYMMDDHHMM>Z.png`,
      `forecast/YYYY/MM/DD/<run>Z/<valid>Z.png`, hash-deduped metadata under
      `meta/<kind>/`, a sqlite frame index (sha256 + fetch time) and
      `forecast/runs.jsonl`; `cadence` summarises the run cadence, `verify`
      cross-checks index vs disk, `decode` renders mm/h. Verified live
      2026-09-03: metadata timestamps and the compact ids in the image URLs
      are **UTC** (`timeOffset` is the site's display offset, 2.0 in CEST),
      run cadence exactly 15 min (18:45/19:00/19:15 UTC) published ~30 min
      after the run id, composite 15-min steps ~40 min behind real time,
      30 forecast frames per run at +35..+180 min, and a run's earliest leads
      fall out of the metadata as it ages (27 left by +55 min) — hence the
      mandatory 5-min tick.
      Georeference is a sidecar (`georeference.json` + `frame.pgw`): EPSG:3857,
      766×652, corners 34-61 N / 13.5 W-35 E, ~7.05 km square pixels.
      **Deployed** on hetz1 since 2026-09-03 21:50 UTC+2:
      `pluvio-buienradar-eu.timer`, every 5 min (`*:00/5:45`), archive root
      `/mnt/storagebox/buienradar_eu`, retention forever. The unit carries
      `Environment=PLUVIO_BUIENRADAR_EU_INDEX=/opt/pluvio/state/buienradar_eu/index.sqlite`
      because the root is CIFS and SQLite cannot lock there ("database is
      locked" on an empty index); the plain `collect --root ...` invocation
      does not work in production and `verify` needs the same env var.
      First ticks: 41 frames, then 0 downloaded / 41 skipped. Measured frame
      sizes 38.9 KB (composite) / 42.7 KB (forecast, wet scene) → 18-46 GB/yr
      at 96 runs/day × 30 frames, forecast frames dominating.
      Open: (a) the colour→mm/h table is PROVISIONAL —
      Buienradar publishes only a 5-class legend (0-2/2-5/5-10/10-100/100+
      mm/h) while the PNGs carry a finer continuous ramp, so
      `png_to_rate` interpolates between the legend anchors and needs
      validating against our own composite over the same frames (use
      `png_to_class` until then); (b) confirm whether composite frames are
      ever revised in place — the collector re-checks the newest two frames
      every tick and would archive a `.r1` twin, none seen in the first hour,
      so this needs days of ticks to answer; (c) storage review once a month
      of real (mixed wet/dry) depth is on disk, against the 18-46 GB/yr
      estimate, before committing to "forever" for good.
      Acceptance: 24 h of unbroken 15-min composite history + every run
      archived, and a decoder validated against our composite.
      Lane: agent + ops. Depends: 3.2

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
- hetz1 timers: `pluvio-buienradar-eu` (every 5 min, EU composite + forecast
  runs), `pluvio-external-baselines` (every 5 min, station raintext),
  `pluvio-scoreboard` (02:30 UTC nightly), hourly `qc_inputs` + `pluvio-qc`,
  `pluvio-qpe-archive` (10 min, now self-describing), */5 append+infer cron.
- Backend: GHCR image from 0888519 (1.13 conventions, Verify on store attrs)
  live since 2026-09-03 21:08 UTC.
- asusprime: idle. v3 training stopped after epoch 12 (best epoch 4 kept).
  Next GPU jobs, in order: render dedup shards with `--lagrangian-channels 2`
  (~132 GiB, see training_run_v2.md), then the 2.1 loss A/B and the 2.3
  ablation on the frozen benchmark.

## Agent workflow
Workers (Sonnet) implement one WBS item each in an isolated worktree branch,
run its tests, and report diff + results. A reviewer (Opus) reviews the diff
against the acceptance criteria before merge. Nothing agent-authored touches
servers; deploys are a separate `ops` step.
