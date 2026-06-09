# Improvements backlog

What we know we want but isn't done yet. Each item should explain
**what**, **why**, **rough cost**, and **expected value** so future-us
can pick the next thing without re-deriving the rationale.

Sorted into roughly _land-soonest_ to _land-eventually_. Promote / demote
freely as priorities shift.

---

## P0 — fix-the-house items (do before scaling further)

### Re-pull deep-archive Meteosat at wide bbox
- **What:** rerun `fetch_eumetsat_msg --start 2020-09-01 --end 2024-08-14`
  for all 6 layers at 60-min cadence, now that the default bbox covers
  the full analysis grid.
- **Why:** we paused the 4-year deep archive to fix the bbox bug. The
  earlier narrow-bbox files were deleted as part of the bbox fix; the
  deep archive currently has zero data on disk.
- **Cost:** ~12 GB disk, ~12 hours wall-clock at 6 parallel streams.
- **Value:** pre-training signal + climatology context; mostly useful as
  an MSG-only auxiliary set since KNMI radar doesn't go back that far.

### Verify and document KNMI rate-limit pacing
- **What:** the rate-limit-aware fetcher works in principle (sleeps to
  the reset). Confirm under the full 31-hour pull that it actually
  recovers cleanly across the first reset, doesn't burn the quota on
  listing requests, and finishes the 22-month window without 403s.
- **Cost:** one careful read of `/tmp/pluvio-pull/knmi_radar.log` once
  the pull completes.
- **Value:** safety — if pacing is broken we'll keep losing chunks.

### Move the in-flight MSG/KNMI data to systemd-managed services
- **What:** the historical pulls are detached via `setsid nohup`. Wrap
  them as systemd-user services (`pluvio-historical-msg@<layer>`,
  `pluvio-historical-knmi`) so they auto-restart on crash and have
  proper journal logs.
- **Cost:** ~1 hour.
- **Value:** robustness; the current ad-hoc bash script can drop a
  stream silently if the venv hiccups.

---

## P1 — channel completeness

### Widen the analysis grid (Level 2 in the bbox discussion)
- **What:** the current 100×100 KNMI-stereographic grid is set by
  KNMI's native radar footprint (NL + edges). Going wider — say
  150×120 covering UK + N. France + W. Germany — gives the model
  more upstream context. Storms approaching from the Atlantic show
  up earlier; pressure tendency across the Channel matters.
- **Cost:** non-trivial.
  - Re-derive `model/geo.py` for the new grid (pick projection +
    extent + cell count).
  - KNMI radar is NaN outside its footprint. Either accept that
    (and let the convnet learn to ignore NaN), or mosaic KMI's
    Belgian radar + DWD's German radar in. Mosaicking is real work.
  - Re-pull every source against the new bbox.
- **Value:** medium-to-high. The verification showed long-lead skill
  drops fastest where radar can't see the precursor — that's the gap
  this fix targets. Worth doing once we have a baseline trained on
  the current grid.

### Wire the 5 remaining MSG layers + 9 ALARO layers into the zarr
- **What:** `build_zarr.py` currently bakes in only `msg_ir108`.
  Generalise to a per-layer config and add `msg_wv062`, `msg_gii_kindex`,
  `msg_gii_liftedindex`, `msg_cth`, `msg_rdt`, plus the 9 ALARO bands
  (precip, cloud cover, u/v wind, RH, CAPE, MSLP, T, dewpoint).
- **Cost:** ~2 hours. Mostly mechanical — the reproj + alignment
  scaffolding is in v1.
- **Value:** unlocks training with the full ~29-channel input the
  architecture doc designed.

### Incremental append on `build_zarr.py`
- **What:** v1 rebuilds the whole zarr from scratch on every run.
  As historical pulls finish and forward cron grows the dataset,
  we want an `--append` mode that only writes new `issue_time` slots.
- **Cost:** ~2 hours.
- **Value:** lets us run the zarr builder daily without spending 30+
  minutes redoing past work.

### Lightning (Blitzortung) collector
- **What:** Blitzortung community network publishes lightning strokes
  as a stream. Station operators get raw archives; the public WebSocket
  feed gives near-real-time data with ~5 km accuracy.
- **Cost:** signup as a station operator OR scrape the public feed
  carefully; collector + grid aggregator ~half a day.
- **Value:** direct convective confirmation — the single best
  "convection is happening now" signal we don't have. Especially
  useful for the long-lead heavy-rain skill the current model misses.

### Dual-pol radar moments
- **What:** KNMI offers polarimetric volume scans at Herwijnen (NL62)
  and Den Helder (NL61) as separate datasets — registered in
  `fetch_knmi_archive.py`'s DATASETS map but not auto-pulled.
  Volume data carries ZDR, RhoHV, KDP, Vrad in addition to reflectivity.
- **Cost:** high.
  - Volume files are 5–20 MB each, ~200 GB for 3 months across both
    radars.
  - Native format is polar (range × azimuth × elevation). Needs
    `wradlib` or `Pyart` to read, plus reprojection onto the 100×100
    Cartesian analysis grid.
  - Same KNMI 1000/h rate limit applies, shares quota with the main
    radar pull.
- **Value:** distinguishes hail/heavy rain from drizzle at the radar
  level — could unlock the convective heavy-rain skill the model
  currently misses.

### NWP ensemble: DWD ICON-D2 + (later) AROME
- **What:** complementary NWP from a different model improves the
  context channel and gives implicit forecast uncertainty (where
  ALARO and ICON-D2 disagree = the model knows it's a hard case).
  ICON-D2 is open, 2 km grid, covers Germany + BeNeLux edge.
- **Cost:** new collector ~1 day each.
- **Value:** medium; mostly for the 6–24 h lead window where ALARO
  alone is the only NWP signal.

### GNSS integrated water vapour
- **What:** EUMETNET E-GVAP and the EPN network publish 15-min
  precipitable-water retrievals from ~150 ground-based GPS receivers
  across Europe. Strong convection precursor.
- **Cost:** ~half day collector; access policy needs confirming
  (free for research per their docs).
- **Value:** medium-high; high IWV + cooling IR = thunderstorm in
  30–60 min, exactly the lead window the model targets.

### Radiosondes (Uccle, De Bilt, Beauvechain)
- **What:** twice-daily soundings give CAPE, CIN, 0-6 km bulk shear
  directly. Sparse temporally (12 h cadence) but high-signal.
- **Cost:** ~3 hours; text format, public archives.
- **Value:** low-medium; the ALARO `Surface_CAPE` channel already
  captures most of the signal at model cadence.

### Sea-surface temperature
- **What:** Copernicus Marine Sentinel-3 / OSI-SAF gridded daily SST.
  Quasi-static at the model timescale.
- **Cost:** account needed at marine.copernicus.eu; collector ~half day.
- **Value:** medium — Belgian summer convection is strongly modulated
  by North Sea SST. Could be modelled as a slowly-varying aux channel.

### Netatmo crowdsourced surface obs
- **What:** thousands of public personal weather stations across
  Benelux. Sub-km density of surface T/RH/pressure.
- **Cost:** OAuth signup (account + client_id/secret), rate-limited
  API. Collector ~1 day.
- **Value:** high if the noise is manageable — surface field density
  is ~20× what KMI AWS gives. Real papers show convective nowcast
  improvements.

---

## P2 — model + training

### Bigger model + GPU training run
- **What:** the CPU phase used a 119k-param UNet (base-16). The
  architecture doc designed for ~1M params (base-32 to -64). With a
  proper GPU run on the wider data we can hit the design size.
- **Cost:** ~€55–100 of A100 time per training run, per the findings
  doc. ~8 hours on a 4080-class GPU. Plus pipeline integration with
  whatever cloud GPU we use.
- **Value:** core deliverable. The CPU model already beats KMI on RMSE
  for 75 % of leads past 30 min on a 6-day window; the bigger model
  on more data should close the convective-detection gap too.

### CorrDiff diffusion architecture
- **What:** swap the deterministic UNet for a diffusion model that
  produces an *ensemble* of samples per inference. Gives free
  uncertainty + sharper convective structure.
- **Cost:** significant. Architecture rewrite, longer training. There
  are pre-trained CorrDiff weights for the EU domain (NVIDIA Modulus
  collab) that might be fine-tunable.
- **Value:** unlocks the "chance interval" UI we deferred + better
  heavy-rain skill. Probably the next-but-one architecture, not the
  next.

### Bias-penalty sweep on the new dataset
- **What:** the precision↔recall tradeoff via `bias_penalty` in
  `train.py` was tuned on a 6-day CPU window. Redo on the proper
  22-month dataset to find the right operating point.
- **Cost:** several training runs.
- **Value:** medium; useful only after the bigger-data run is solid.

---

## P3 — ops + nice-to-have

### Higher-res static elevation
- **What:** current `static.npz` uses ETOPO1 (~1.8 km). For the 100×100
  grid that's effectively 4 ETOPO cells per analysis cell — fine. If
  we ever go to a denser grid (Level 2), bump to SRTM 30m and aggregate
  ourselves.
- **Cost:** small. SRTM tiles, rasterio aggregation.
- **Value:** low at current resolution; high if/when we densify.

### ALARO historical via different endpoint
- **What:** opendata.meteo.be WMS exposes only forward-of-current
  ALARO. To get historical ALARO (paired with old KNMI radar), we'd
  need to find an archive endpoint or wait for cron to accumulate
  forward.
- **Cost:** unknown. Need to ask KMI Open Data team or grovel through
  their site.
- **Value:** medium — unlocks paired NWP training with the existing
  radar archive going back to 2024-08-14.

### Backup the zarr off the homeserver VM disk
- **What:** the zarr lives on `/home/jeroentrappers/...`, on the VM's
  286 GB disk. If the VM dies / disk corrupts we re-collect the lot
  (multi-day pulls). Periodic snapshot to a real NAS dataset would
  insure against that.
- **Cost:** ~1 hour setup (TrueNAS-side snapshot/clone, or rsync target).
- **Value:** medium — protects against losing weeks of accumulated
  forward-cron data.

### Disk-grow plan if we ever scale up
- **What:** current VM disk is 286 GB, 180 GB taken by Home Assistant
  data, leaving ~70 GB free after all pulls. If we want the deep
  archive AND room to grow + train artifacts, we need ~500 GB.
- **Cost:** TrueNAS UI → grow zvol; small.
- **Value:** unlocks scope expansion (deep archive, dual-pol, model
  checkpoints, eval artefacts).

### Monitoring + alerting on the forward cron
- **What:** if a forward timer starts silently failing — KNMI rotates
  an API token, a layer name changes, disk fills — we'd notice via
  staleness only when training fails weeks later. Want a basic check
  ("did each source land a file in the last N minutes?") + an alert.
- **Cost:** ~half day to wire to email or Home Assistant notifications.
- **Value:** medium; insurance against silent data rot.

### Pluvio backend: swap in the trained model
- **What:** once a real model checkpoint exists, the backend's
  `inference_worker.run_tick(infer=...)` swap from `stub_band` to the
  trained model is one line. Plus packaging the model into the Docker
  image (size considerations) and benchmarking inference latency on
  the CAX11's ARM CPU.
- **Cost:** small for the wiring; the model packaging + tuning is the
  real work.
- **Value:** ships the actual product.
