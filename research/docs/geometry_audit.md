# `geo.bbox()` over-claims the stereographic domain — consumer audit (1.10)

## The bug

The legacy KNMI stereographic analysis grid (`Grid.legacy_knmi_analysis`,
`geo.grid_latlon()`) is **not** a lat/lon rectangle: it is a regular grid in
*projected* (km) space, so its rows and columns curve when expressed in
lat/lon. `geo.bbox()` (now `geo.envelope()`) returns
`(lon.min(), lat.min(), lon.max(), lat.max())` — the axis-aligned corner
envelope of all 10 000 cell centres. That envelope is a real superset of the
grid's true footprint, but it is not the footprint itself.

Measured on the current 100x100 default grid (`research/model/grid.py`,
`research/model/geo.py`), at zero registration bias — every lat/lon literal
in the table below is `bias=(0.0, 0.0)`; the default `(0, 0.07)` bias shifts
all of them east by 0.07 deg and changes none of the gaps, which are
differences (the `envelope()`/`inner_rectangle()` literals quoted further
down DO include the default bias):

| edge  | row/col that actually reaches the envelope value | worst-case gap between the envelope edge and that row/col's own extremum |
|-------|---------------------------------------------------|----------------------------------------------------------------------------|
| north | 0 (north row) touches `lat.max()` at one column   | 0.585 deg lat ≈ **64.9 km** (other columns' own row-0 latitude falls short of the envelope's north edge) |
| south | last row touches `lat.min()` at one column        | 0.475 deg lat ≈ **52.8 km** |
| east  | `lon.max()` is reached at (row 0, col 99) — the **NE corner** (lat 55.389, lon 10.8565) | 1.715 deg lon ≈ **115.2 km flat-earth** (124.4 km geodesic — see below); the **SE corner** is where the deficit is: the south row's own east-most longitude (9.1417, at (row 99, col 99)) falls 1.715 deg short of the envelope's east edge |
| west  | the whole west column sits at lon = 0.0 exactly    | **0 km, exactly** — not approximate: `_LEGACY_PROJ4` has `lon_0=0` and both west corners (`_LEGACY_CORNERS_LONLAT` LL and UL) sit at lon 0.0, so the whole west column is at projected x = 0 by construction, not by coincidence |

The edge-by-edge framing above understates the actual defect for any
consumer that builds an *independent* regular lat/lon grid over the envelope
(rather than just using the envelope as a fetch region) and then treats that
new grid as if it were the true curved grid cell-for-cell — see
`radar_single_site.polar_to_grid` (table entries #9, #10 and #12-#17
below). That comparison is wrong everywhere, not just at the edges:
measuring the great-circle distance (geodesic, WGS84) between each
naive-grid cell centre and the true curved cell centre at the same
`(row, col)`, over the whole 100x100 grid — the naive grid built exactly the
way `polar_to_grid` does it (`col = (lon - w)/(e - w) * wd`, i.e. cell
centres at half-pixel offsets inside the envelope):

| stat | value |
|------|-------|
| min | 0.1 km |
| median | 37.1 km |
| mean | 39.4 km |
| max | 120.4 km |
| NW corner (0,0) | 5.0 km |
| NE corner (0,99) | 61.5 km |
| SW corner (99,0) | 49.4 km |
| SE corner (99,99) | 120.4 km (= the domain max) |
| centre | 30.7 km |

(Pinned in `research/tests/test_grid.py::test_documented_naive_regular_grid_displacement`
so this table cannot silently drift from the geometry.)

That is a **4-9 cell shift everywhere on the grid**, not a boundary effect —
the centre of the domain is already off by ~4 cells (30.7 km on a ~7-8
km/cell grid). Restricted to the single-radar `--max-range-km 100` disc that
`radar_single_site.py --compare-opera` actually scores against (site
coordinates from the ODIM volumes), the same naive-vs-true displacement is:

| radar | median (km) | max (km) |
|-------|--------------|----------|
| bejab | 43.8 | 55.7 |
| behel | 60.6 | 78.0 |
| bewid | 69.3 | 89.1 |

(Site lon/lat as read from the ODIM volumes, mask exactly as
`--compare-opera` computes it. The per-radar figures move a few km with the
site coordinates and the range-mask convention; the 40-90 km band — 6-12
cells — does not.)

**Every historical `--compare-opera` correlation, bias, and MAE produced by
`radar_single_site.py` is therefore uninformative** — not merely biased
toward the domain edges. The comparison is a systematic multi-cell
(4-9 cell) shift between the "our field" grid and the "OPERA reference"
grid at essentially every point in the scored disc, which is exactly the
kind of geometry error `verify_radar.py`'s own doc-comment says should drive
correlation toward zero or negative — the fact that historical runs of this
tool reported plausible-looking positive correlations at all is itself
evidence the shift was likely deflating them rather than being caught by
them. (`verify_radar.py` itself is unaffected — see entry #11.)

## Fix landed in this pass (`research/model/geo.py`, `research/model/grid.py`)

`bbox()` now has a docstring stating exactly this, and is joined by two
explicitly-named functions so a new caller can pick the right one on purpose
instead of reaching for `bbox()` by habit:

- `envelope()` (`grid.py: Grid.envelope()`, `geo.py: envelope()`) — identical
  to the old `bbox()`: the corner envelope, a guaranteed **superset** of the
  domain. Safe for "fetch/reproject a region that must cover the whole grid"
  uses.
- `inner_rectangle()` (`grid.py: Grid.inner_rectangle()`,
  `geo.py: inner_rectangle()`) — the largest lon/lat rectangle guaranteed to
  be covered by *every* row and *every* column, i.e. a guaranteed **subset**
  of the domain. Safe for "this box must actually sit inside the grid" uses.
  For the current 100x100 grid this is
  `(0.070, 49.914, 9.212, 55.389)` vs. the envelope's
  `(0.070, 49.439, 10.926, 55.974)`.

All three of `bbox()`, `envelope()` and `inner_rectangle()` take the same
optional `bias` argument as `grid_latlon()`, so a caller (or a test) can pin
the geometry instead of depending on `PLUVIO_GRID_LATLON_BIAS` (1.11).

Also fixed here, since it is a one-line ordering bug rather than a
reprojection change: `radar_single_site.py` set `PLUVIO_GRID_N` from
`--grid-n` *after* its first `from model.geo import ...`, which made the flag
a silent no-op (#24 below). The `os.environ.setdefault()` now sits right
after `parse_args()`/`basicConfig()`, matching `verify_radar.py`, and
`research/tests/test_geo.py` pins the line ordering for both tools.

`bbox()` itself is kept as a deprecated-in-spirit alias for `envelope()`
(not renamed) because most existing callers already use it correctly (see
table below) and a rename would touch every one of them for no behavioural
change.

## Consumer table

Every `bbox(`, `radarBounds`, `RADAR_BOUNDS`, WMS `bbox` string, and npz
`bounds` producer/consumer in the repo, audited for what it does with the
box and whether the corner-envelope-vs-true-domain gap above is material.

| # | Consumer | File:line | What it does with the box | Error magnitude | Verdict |
|---|----------|-----------|----------------------------|------------------|---------|
| 1 | `_frame_to_grid_index` | `research/model/build_aux_msg.py:39-50` | Fetches a WMS frame that is genuinely plate-carrée over `bbox()`, then samples it at the TRUE curved `grid_latlon()` points via linear pixel index into that frame. `bbox()` is a real superset, so every curved point is guaranteed inside the fetched frame. | None — the frame itself covers the full envelope, and the sampled points are the true curved lat/lon, not a regular re-grid. | **Correct** |
| 2 | `fetch_mask` WMS `bbox` param | `research/model/build_aux_msg.py:62-68` | Same fetch region as #1. | None | **Correct** |
| 3 | `fetch_geotiff` WMS GetMap, `--bbox` CLI default | `research/collectors/fetch_eumetsat_msg.py:66-90,122-131` | `--bbox` default is a literal `0.0,48.5,11.0,56.0` superset (docstring explicitly cites `geo.bbox() ≈ (0,48.9,10.9,55.97)` and says "so every grid cell gets MSG coverage"). Fetched raster is later reprojected onto the analysis grid via `nwp_regrid.reproject_to_analysis_grid`, which warps through the TRUE stereographic `analysis_grid_dst()` — the fetch box is only ever a coverage superset, never treated as the grid itself. | None (over-fetch only; downstream reprojection uses the real CRS) | **Correct / harmless** |
| 4 | `fetch_geotiff` WMS GetMap, `--bbox` CLI default | `research/collectors/fetch_alaro_24h.py:56-81,111-121` | Same pattern as #3 (ALARO instead of MSG). | None | **Correct / harmless** |
| 5 | `build()` DEM mosaic region | `research/tools/build_dem.py:85-88` | Builds a genuinely regular EPSG:4326 DEM raster covering `bbox()` (a superset) at ~500 m; the DEM's own `bounds` describe that raster truthfully (it is what it says it is — a real rectangle). | None (DEM raster is honestly regular; only the coverage region comes from `bbox()`, as a safe superset) | **Correct / harmless** |
| 6 | `_sample_dem` | `research/tools/beam_blockage.py:36-53` | Samples the DEM raster (#5) at arbitrary continuous (lat, lon) points (radar beam paths) using the DEM's own true regular bounds — never conflated with the curved analysis grid. | None | **Correct** |
| 7 | `LOG.info("grid %s, bbox=%s", ...)` | `research/model/build_static.py:131` | Logging only. | None | **Harmless** |
| 8 | `analysis_grid_dst()` reprojection target | `research/model/geo.py:122-139` | Uses `_legacy_trimmed_extent()` (the true projected extent in the native stereographic CRS), **not** `bbox()`. Listed here only because it's the correct pattern the broken consumers below should have used. | N/A | **Correct (reference)** |
| 9 | `polar_to_grid` / `_polar_geometry` binning | `research/tools/radar_single_site.py:287-411` (`row = ((n-lats)/(n-s)*h)`, `col = ((lons-w)/(e-w)*wd)`) | Bins polar radar bins onto a grid by **linearly interpolating lat/lon across `bounds`** — i.e. it builds its own independent *regular* lat/lon raster over whatever box it is given. When called with `bounds=bbox()` (every call site below), that raster is geometrically different from the true curved `geo.grid_latlon()` grid of the same shape, by up to the table-1 gaps. | Up to **~115 km** at the domain edges between this tool's own grid and the true analysis grid of the same `GRID` shape | **Must fix — flagged, not silently fixed** (see below) |
| 10 | `--compare-opera` scoring | `research/tools/radar_single_site.py:459-488` | Builds the single-radar field via #9 (naive regular grid over `bbox()`), then scores it against `nwp_regrid.reproject_to_analysis_grid(opera_tif)`, which warps OPERA onto the TRUE curved `analysis_grid_dst()` target. The two "grids" being compared cell-for-cell are **not the same grid** — this is the ONLY consumer in this audit that actually compares the naive-regular grid against the true curved grid. | Systematic multi-cell (4-9 cell / ~30-120 km) shift across the whole scored disc — see "The bug" above; every historical run's correlation/bias/MAE is uninformative | **Must fix — flagged, not silently fixed** (see below) |
| 11 | `verify_radar.opera_ref` / `field` | `research/tools/verify_radar.py:42-99,120-156` | **Not a defect.** `opera_ref` reprojects OPERA with `rasterio.warp.reproject(..., dst_transform=from_bounds(w, sth, e, n, shape[1], shape[0]), dst_crs=EPSG:4326)` — the exact same naive regular lat/lon box (`bounds`, default `bbox()`, overridable via `--bounds`) that `field()`/`polar_to_grid`/#9 bins the radar data onto. Both sides of the comparison are built on the identical regular box, so they are self-consistent with each other (unlike #10, which compares the naive box against the true curved `analysis_grid_dst()` grid). | ≈0 — both sides use the same regular grid | **Correct / harmless** |
| 12 | `radar_composite.build` (multi-radar composite + quality weighting) | `research/tools/radar_composite.py:62-190` | `beam_height_grid`/`site_grid_distance`/`radar_field` all key off `bounds=bbox()` fed into #9's binning; internally self-consistent (every radar in the composite is binned onto the *same* naive-regular grid), so radars don't misalign against each other. But if this composite is ever compared against, or written into a store next to, data that assumes the true curved grid (e.g. `analysis_grid_dst`-reprojected OPERA, or `geo.grid_latlon()`-keyed truth), the same ~115 km edge error applies. | Up to ~115 km vs. anything keyed to the true curved grid; 0 internally (self-consistent composite) | **Must fix — flagged** (self-consistent internally; external comparisons are the risk) |
| 13 | `qpe_archive.py` per-radar QPE accumulation | `research/tools/qpe_archive.py:86-96` | Same `bounds=bbox()` → #9 pattern, accumulated over a day. Per the recent v3 work (`build_store_v3.py` truth now comes from native 1-km RAC tars — commit `f67954d`), this archive is noted elsewhere as "pruned" from the current truth pipeline, but the tool itself is unchanged. | Up to ~115 km if ever compared/joined against the true curved grid | **Must fix — flagged** |
| 14 | `qpe_batch.py` batch QPE + gauge sampling | `research/tools/qpe_batch.py:35-54` | Same pattern; also uses `bounds` to sample gauge lat/lon into the naive grid via `sample()`. | Gauge (lat,lon) sampled into the WRONG cell by up to ~115 km worth of grid displacement near the domain edges (less near the domain interior, where the curvature is smaller) | **Must fix — flagged** |
| 15 | `gauge_validate.py` gauge sampling | `research/tools/gauge_validate.py:97-180` | Same `sample()`/`bounds=bbox()` pattern as #14. | Same as #14 | **Must fix — flagged** |
| 16 | `zr_calibrate.py` Z-R calibration sampling | `research/tools/zr_calibrate.py:39-67` | Same pattern. | Same as #14 | **Must fix — flagged** |
| 17 | `clutter_map.py` clutter frequency map | `research/tools/clutter_map.py:48-91` | `build_frequency(..., bounds, shape, ...)` → #9 pattern. | Up to ~115 km if compared against the true curved grid | **Must fix — flagged** |
| 18 | `backend/src/pluvio_backend/cache.py` `DEFAULT_BOUNDS` | `backend/src/pluvio_backend/cache.py:52-53` | A hardcoded, genuinely regular Belgium/Benelux lat/lon box `(1.5, 48.9, 7.5, 52.5)` — **not** derived from `geo.bbox()` at all (values don't match: `bbox()` is `(0.07, 49.44, 10.93, 55.97)`). Used as `GridSpec.bounds` for `latlon_to_cell` (regular-grid math, correct for a genuinely regular box). | None from the curvature bug (this box was never the curved grid); out of scope for this pass — see "Not modified" below | **Correct as a regular grid, but disconnected from `geo.bbox()`** — document only |
| 19 | `lib/core/config/env.dart` `Env.radarBounds*` | `lib/core/config/env.dart:27-46` | Same hardcoded Belgium box as #18 (`1.5, 7.5, 48.9, 52.5`), used to place the Flutter map's `LatLngBounds` (`lib/features/radar/presentation/widgets/radar_map.dart:23-56`). | Same as #18 | **Document only (out of scope: lib/)** |
| 20 | `web/src/*` overlay bounds (`RadarMap.tsx`, `api.ts`, `App.tsx`) | `web/src/map/RadarMap.tsx`, `web/src/api.ts:7`, `web/src/App.tsx` | Consume the backend's `grid.bounds`/npz `bounds` (the regular Belgium box, #18/#22) to place the overlay image via a regular-grid `LatLngBounds`-equivalent. | Same as #18 (regular box, not curved) | **Document only (out of scope: web/)** |
| 21 | `backend/src/pluvio_backend/colormap.py:draw_fiducials`, `history.py`, `verify.py`, `model.py`, `stubs.py`, `api.py` | `backend/src/pluvio_backend/*.py` | All consume the regular Belgium `bounds` (from `cache.GridSpec`/npz), never `geo.bbox()`/the curved grid. | None from this bug (see 1.13 for the separate centre-vs-edge pixel convention issue already tracked) | **Document only (out of scope: backend/)** |
| 22 | `infer_latest.py` npz `bounds` (legacy v2 fallback path) | `research/model/infer_latest.py:121-143` | The v3-store path emits `store_grid.bounds` (a genuinely regular `Grid.regular(...)` box — not the curved legacy grid). The legacy v2 fallback path explicitly **reprojects** (`scipy.interpolate.griddata`) the curved field from `glat/glon` (true `grid_latlon()` points) onto a hardcoded regular Belgium box (`BE_W/S/E/N = 1.5, 48.9, 7.5, 52.5`, matching #18) before writing `bounds=[BE_W,BE_S,BE_E,BE_N]`. This is a real reprojection from curved to regular, done correctly — it does not use `bbox()` at all; see #23 for the one real defect in this path. | None — this is the pattern #10 should be using instead of a naive linear re-grid | **Correct (reference)** |
| 23 | `infer_latest.py` npz `bounds` vs. backend/web/lib painters | `research/model/infer_latest.py:121-143` (writer) vs. `backend/src/pluvio_backend/colormap.py:draw_fiducials`, `history.py`, `web/src/map/RadarMap.tsx`, `lib/.../radar_map.dart` `OverlayImage` (readers) | `infer_latest.py` writes `bounds` as the CELL-CENTRE envelope (`store_grid.bounds`, or the `BE_W/S/E/N` centre-convention literals — see `Grid`'s docstring: "bounds is the envelope of the CELL CENTRES... painters that treat bounds as pixel EDGES... must use edge_bounds()"). The backend/web/lib painters that place the served overlay image treat that same `bounds` as pixel EDGES (`* h`, `int`, per 1.13's already-tracked centre-vs-edge split — `cache.GridSpec.latlon_to_cell` disagrees with `colormap.draw_fiducials`/`history.py` the same way). | A uniform half-cell shift (~2 km at the current backend grid resolution) of the entire served overlay/fiducials relative to the true radar positions | **Must fix — in backend/web/lib (1.13), out of scope here; documented, not touched** |
| 24 | `radar_single_site.py` `--grid-n` / `PLUVIO_GRID_N` ordering | `research/tools/radar_single_site.py:443,466` (was: import at 459, `setdefault` at 461) | `from model.geo import GRID, bbox` used to run **before** `os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))`. `geo.GRID` is resolved once at import time (`GRID = _default_grid()`, `geo.py:65`) — reading `PLUVIO_GRID_N` fresh, but only on THAT import. Since that was the first import of `model.geo` in the script, `--grid-n` was silently a no-op: the tool always gridded at the module's already-resolved 100x100 (or whatever `PLUVIO_GRID_N` was set to in the calling shell), never the value the user passed on the CLI (whose own default is 256). This is 1.11's env-latching mechanism biting a second call site. `verify_radar.py` always had the order right (`setdefault` at line 128, `from model.geo import GRID, bbox` at 133). | `--grid-n` requests were silently ignored; every run gridded at the default resolution regardless of the flag | **FIXED in this pass** — the `setdefault` now runs immediately after `parse_args()`, before any `model.geo` import; line ordering pinned for both tools by `test_grid_n_env_is_set_before_model_geo_is_imported` |

## "Must fix — flagged" items: why not fixed here

Items 9, 10, 12-17 (eight call sites) all share one root cause:
`radar_single_site.polar_to_grid` (and its `_polar_geometry` cache) computes
`row`/`col` for a lat/lon point by **linear interpolation across a `bounds`
box** (`col = (lons - w) / (e - w) * wd`, `row = (n - lats) / (n - s) * h`).
That is the definition of binning onto a *new, independently regular*
lat/lon grid — it is not, and cannot by construction be, the curved
stereographic `geo.grid_latlon()` grid of the same shape, regardless of
which `bounds` box is passed in. (`verify_radar.py`, entry #11, uses the
same binning but reprojects its OPERA reference onto the identical naive
box rather than the true curved grid, so it is not part of this group —
see #11.)

Making these tools bin onto the *true* curved grid instead would mean
replacing that linear formula with a nearest-neighbour lookup against the
10 000 actual `grid_latlon()` cell centres (e.g. a per-geometry `cKDTree`,
analogous to the existing hole-fill KDTree in `_polar_geometry`) — a real
reprojection change to a shared, cached, hot-path binning routine used by
all eight sites across QC, calibration, and verification tooling. Per the
guidance for this pass, that kind of fix is **flagged, not silently
applied**: swapping `bbox()` for `inner_rectangle()` at these call sites
would not fix anything (the binning is still linear-over-a-box; it would
just shrink the box), and doing the real fix — reprojecting the binning
itself — is out of scope here and belongs in its own change with its own
review, given the blast radius (`radar_composite`, `qpe_archive`,
`qpe_batch`, `gauge_validate`, `zr_calibrate`, `clutter_map`,
`beam_blockage.quality_grid`, and the `--compare-opera` path of
`radar_single_site.py` all call it).

Practical mitigation already true today: items #9, #12-#17 are
**internally self-consistent** (every radar/gauge/day in a given run is
binned onto the same naive-regular grid), so within-tool comparisons are
not corrupted by this bug. The risk is specifically at the seam where one
of these tools' output is compared against, or joined with, data keyed to
the true curved `geo.grid_latlon()`/`analysis_grid_dst()` grid — item #10
(`--compare-opera`) does exactly that today and is the single highest-
priority follow-up (see "The bug" above for why every historical run of it
is uninformative, not merely edge-biased). (#24, the
`--grid-n`/`PLUVIO_GRID_N` ordering bug in the same function, was a one-line
ordering fix and IS fixed in this pass — see above.)

## Not modified (documented only, per task scope)

- `backend/src/pluvio_backend/cache.py` (`DEFAULT_BOUNDS`, `GridSpec`)
- `web/src/*` (`RadarMap.tsx`, `api.ts`, `App.tsx`, `types.ts`,
  `VerifyView.tsx`)
- `lib/core/config/env.dart`, `lib/features/radar/presentation/widgets/radar_map.dart`

The served overlay pipeline is otherwise clean up to the point in #23: the
legacy `infer_latest.py` path reprojects (`griddata`) the curved field onto
a genuinely regular box *before* writing `bounds` (#22), so the only live,
user-visible georeferencing error reaching a real user today is #23's
half-cell centre-vs-edge shift (already tracked as 1.13) — not a curvature
error at all.

These all consume a hardcoded, genuinely regular Belgium/Benelux lat/lon box
that is **not derived from `geo.bbox()`** (item #18 above documents the
values don't even match) — so this pass's curvature bug does not reach them
directly. They are listed for completeness (the TODO explicitly calls out
"same blast radius as the trim bug") and because any future change that
*does* wire the backend/web/lib bounds to `geo.bbox()`/`envelope()` must use
`inner_rectangle()` instead if the intent is "this box is inside the radar's
true coverage", not `envelope()`/`bbox()`.
