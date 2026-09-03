"""Backend pixel conventions (1.13): `GridSpec.bounds` is a CELL-CENTRE
envelope (matches research/model/grid.py `Grid.bounds`, the store contract),
painters take `edge_bounds()`, and `latlon_to_cell` floors the fractional
EDGE-based index — the one convention, identical to `Grid.cell_of()`.

Before this, `latlon_to_cell` measured a fractional index off the centre
bounds while `colormap.draw_fiducials`' callers handed it raw centre bounds
where edge bounds belong; the two disagreed by up to a whole cell at the
south/east edge. This file checks that disagreement is gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from pluvio_backend.cache import GridSpec, edge_bounds
from pluvio_backend.colormap import draw_fiducials

BOUNDS = {"west": 1.5, "east": 7.5, "south": 48.9, "north": 52.5}
SHAPE = (192, 192)
CORNERS_AND_CENTRE = [(0, 0), (0, 191), (191, 0), (191, 191), (95, 95)]


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(bounds=BOUNDS, shape=SHAPE)


def _cell_size(shape: tuple[int, int]) -> tuple[float, float]:
    h, w = shape
    return (
        (BOUNDS["north"] - BOUNDS["south"]) / (h - 1),
        (BOUNDS["east"] - BOUNDS["west"]) / (w - 1),
    )


# -- round-trip: lat/lon -> cell -> centre, within half a cell -------------


@pytest.mark.parametrize("row,col", CORNERS_AND_CENTRE)
def test_cell_center_round_trips_exactly(grid, row, col):
    """A cell's own centre maps back to that cell, at the four corners and
    the middle alike — no float-noise off-by-one."""
    lat, lon = grid.cell_center_latlon(row, col)
    assert grid.latlon_to_cell(lat, lon) == (row, col)


@pytest.mark.parametrize("row,col", CORNERS_AND_CENTRE)
def test_latlon_round_trips_to_within_half_a_cell(grid, row, col):
    """lat/lon → cell → centre is never off by more than half a cell, in any
    direction, anywhere on the grid: each cell owns exactly the half-cell
    margin around its own centre (floor of the edge-based index == round of
    the centre-based index). Nudging by up to just under half a cell must
    still land in the same cell; nudging by just over must not."""
    lat, lon = grid.cell_center_latlon(row, col)
    dlat, dlon = _cell_size(grid.shape)
    h, w = grid.shape

    for f_lat, f_lon in [
        (0.0, 0.0),
        (0.49, 0.0),
        (-0.49, 0.0),
        (0.0, 0.49),
        (0.0, -0.49),
        (0.49, 0.49),
        (-0.49, -0.49),
    ]:
        probe_lat, probe_lon = lat + f_lat * dlat, lon + f_lon * dlon
        r, c = grid.latlon_to_cell(probe_lat, probe_lon)
        assert (r, c) == (row, col), f"nudge ({f_lat}, {f_lon}) left cell ({row}, {col})"
        back_lat, back_lon = grid.cell_center_latlon(r, c)
        assert abs(back_lat - probe_lat) <= dlat / 2 + 1e-9
        assert abs(back_lon - probe_lon) <= dlon / 2 + 1e-9

    # Just over half a cell belongs to the neighbour — only checkable where
    # a neighbour exists (interior side of each corner).
    if row + 1 < h:
        assert grid.latlon_to_cell(lat - 0.51 * dlat, lon)[0] == row + 1
    if col + 1 < w:
        assert grid.latlon_to_cell(lat, lon + 0.51 * dlon)[1] == col + 1


def test_latlon_to_cell_accepts_points_in_the_edge_margin(grid):
    """A point past the centre-bounds envelope but still inside the boundary
    cell's own footprint (`edge_bounds()`) resolves to that cell rather than
    raising — the same contract as research `Grid.cell_of()`. Past the
    footprint it raises."""
    _ew, es, _ee, en = grid.edge_bounds()
    assert es < BOUNDS["south"] and en > BOUNDS["north"]
    assert grid.latlon_to_cell(lat=BOUNDS["south"] - 1e-3, lon=BOUNDS["west"]) == (
        grid.shape[0] - 1,
        0,
    )
    assert grid.latlon_to_cell(lat=BOUNDS["north"] + 1e-3, lon=BOUNDS["east"]) == (
        0,
        grid.shape[1] - 1,
    )
    with pytest.raises(ValueError):
        grid.latlon_to_cell(lat=es - 0.01, lon=BOUNDS["west"])
    with pytest.raises(ValueError):
        grid.latlon_to_cell(lat=BOUNDS["south"], lon=_ee + 0.01)


def test_edge_bounds_inflates_by_half_a_cell(grid):
    ew, es, ee, en = grid.edge_bounds()
    dlat, dlon = _cell_size(grid.shape)
    assert ew == pytest.approx(BOUNDS["west"] - dlon / 2)
    assert ee == pytest.approx(BOUNDS["east"] + dlon / 2)
    assert es == pytest.approx(BOUNDS["south"] - dlat / 2)
    assert en == pytest.approx(BOUNDS["north"] + dlat / 2)
    assert edge_bounds(
        (BOUNDS["west"], BOUNDS["south"], BOUNDS["east"], BOUNDS["north"]), grid.shape
    ) == (ew, es, ee, en)


def test_degenerate_single_cell_grid_is_handled():
    """A 1x1 grid has no cell spacing to inflate by; nothing should divide by
    zero and every in-footprint point is cell (0, 0)."""
    g = GridSpec(bounds={"west": 3.0, "east": 3.0, "south": 51.0, "north": 51.0}, shape=(1, 1))
    assert g.edge_bounds() == (3.0, 51.0, 3.0, 51.0)
    assert g.cell_center_latlon(0, 0) == (51.0, 3.0)
    assert g.latlon_to_cell(51.0, 3.0) == (0, 0)


def test_cell_center_latlon_rejects_out_of_range_cells(grid):
    with pytest.raises(ValueError):
        grid.cell_center_latlon(grid.shape[0], 0)
    with pytest.raises(ValueError):
        grid.cell_center_latlon(0, -1)


# -- edge vs centre disagreement eliminated ---------------------------------


def test_fiducial_painted_at_a_cell_reads_back_at_that_cell(grid):
    """Paint a fiducial cross at a known (row, col) with the painter's own
    convention (`draw_fiducials` against `edge_bounds()`), then recover its
    cell with `latlon_to_cell` and confirm they agree — the 1.13 acceptance
    check: the half-cell edge-vs-centre bug is gone."""
    h, w = grid.shape
    # Near the south/east edge: half-cell errors exceed a whole cell there,
    # so they can't hide behind truncation.
    row_pick, col_pick = 5, 186
    target_lat, target_lon = grid.cell_center_latlon(row_pick, col_pick)

    rgba = np.zeros((h, w, 4), dtype="uint8")
    draw_fiducials(rgba, grid.edge_bounds())
    # draw_fiducials only paints its own FIDUCIALS city list, so replicate
    # its two index lines verbatim against the (row_pick, col_pick) target —
    # this checks the painter's *convention*, not its city list.
    wst, sth, est, nth = grid.edge_bounds()
    r = int((nth - target_lat) / (nth - sth) * h)
    c = int((target_lon - wst) / (est - wst) * w)
    assert (r, c) == (row_pick, col_pick)

    assert grid.latlon_to_cell(target_lat, target_lon) == (row_pick, col_pick)


def test_painted_fiducial_cities_read_back_at_their_own_cell(grid):
    """End-to-end through the real painter: every FIDUCIALS city inside the
    grid leaves its cross centred on the cell `latlon_to_cell` reports."""
    from pluvio_backend.colormap import FIDUCIALS

    h, w = grid.shape
    for _name, lat, lon in FIDUCIALS:
        rgba = np.zeros((h, w, 4), dtype="uint8")
        draw_fiducials(rgba, grid.edge_bounds())
        row, col = grid.latlon_to_cell(lat, lon)
        # The cross's centre pixel is painted magenta by both arms.
        assert tuple(rgba[row, col]) == (255, 0, 255, 255), f"{_name} cross missed its cell"


def test_using_centre_bounds_instead_of_edge_bounds_would_disagree(grid):
    """The two conventions are NOT interchangeable — this suite would catch
    the regression it guards against. Near the south/east corner, painting
    against the (wrong) centre bounds lands in a different cell."""
    h, w = grid.shape
    row_pick, col_pick = 191, 191
    target_lat, target_lon = grid.cell_center_latlon(row_pick, col_pick)

    w_, s_, e_, n_ = BOUNDS["west"], BOUNDS["south"], BOUNDS["east"], BOUNDS["north"]
    r_wrong = int((n_ - target_lat) / (n_ - s_) * h)
    c_wrong = int((target_lon - w_) / (e_ - w_) * w)
    assert (r_wrong, c_wrong) != (row_pick, col_pick)


def test_matches_research_grid_cell_of_formula(grid):
    """Equivalence with research/model/grid.py `Grid.cell_of()`, replicated
    here (it can't be imported without the research env): that one rounds the
    fractional CENTRE index, this one floors the fractional EDGE index — the
    same cell for every point, which is what "one convention" means. Checked
    over a deterministic sweep, including the half-cell margins."""
    h, w = grid.shape
    ew, es, ee, en = grid.edge_bounds()
    rng = np.random.default_rng(1313)
    lats = np.concatenate([rng.uniform(es, en, 300), [es, en, BOUNDS["south"], BOUNDS["north"]]])
    lons = np.concatenate([rng.uniform(ew, ee, 300), [ew, ee, BOUNDS["west"], BOUNDS["east"]]])
    for lat, lon in zip(lats, lons, strict=True):
        # --- Grid.cell_of(), verbatim apart from the bounds accessors ---
        col_ref = round((lon - BOUNDS["west"]) / (BOUNDS["east"] - BOUNDS["west"]) * (w - 1))
        row_ref = round((BOUNDS["north"] - lat) / (BOUNDS["north"] - BOUNDS["south"]) * (h - 1))
        ref = (min(max(int(row_ref), 0), h - 1), min(max(int(col_ref), 0), w - 1))
        # --- end verbatim block ---
        assert grid.latlon_to_cell(float(lat), float(lon)) == ref
