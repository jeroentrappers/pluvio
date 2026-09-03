"""model/motion.py — the canonical block-matching flow estimator shared by
tools/_advection.py (imports it) and backend/src/pluvio_backend/morph.py
(keeps a synced copy, since it can't import the research package)."""

from __future__ import annotations

import numpy as np
import pytest
from model.motion import block_flow, flow_field, warp

SHAPE = (128, 128)


def _rain_blob(shape=SHAPE, cy=None, cx=None, radius=12, rate=5.0) -> np.ndarray:
    h, w = shape
    if cy is None:
        cy = h // 2
    if cx is None:
        cx = w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros(shape, dtype="float32")
    field[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = rate
    return field


# ─────────────────────────────────────────────────── recovers a known shift


@pytest.mark.parametrize("dy,dx", [(3, 3), (2, 4)])
def test_flow_field_recovers_shift_without_subpixel(dy, dx):
    a = _rain_blob(cy=60, cx=60)
    b = _rain_blob(cy=60 + dy, cx=60 + dx)
    fy, fx = flow_field(a, b, max_shift=8, subpixel=False)
    region = np.s_[40:80, 40:88]
    assert np.median(fy[region]) == pytest.approx(dy, abs=0.5)
    assert np.median(fx[region]) == pytest.approx(dx, abs=0.5)


@pytest.mark.parametrize("dy,dx", [(3, 3), (2, 4)])
def test_flow_field_recovers_shift_with_subpixel(dy, dx):
    a = _rain_blob(cy=60, cx=60)
    b = _rain_blob(cy=60 + dy, cx=60 + dx)
    fy, fx = flow_field(a, b, max_shift=8, subpixel=True)
    region = np.s_[40:80, 40:88]
    assert np.median(fy[region]) == pytest.approx(dy, abs=0.25)
    assert np.median(fx[region]) == pytest.approx(dx, abs=0.25)


def test_pure_x_shift_has_small_cross_axis_component():
    a = _rain_blob(cy=60, cx=60)
    b = _rain_blob(cy=60, cx=63)  # pure +3 px in x
    fy, fx = flow_field(a, b, max_shift=8, subpixel=True)
    region = np.s_[40:80, 40:88]
    assert np.median(fx[region]) == pytest.approx(3.0, abs=0.25)
    assert abs(np.median(fy[region])) < 0.2


# ─────────────────────────────────────────────────────── mass-bias removed


def test_bright_compact_cell_and_dim_broad_region_get_correct_signed_flow():
    """A raw dot-product score is biased toward whichever candidate offset
    overlaps the most accumulated mass — a bright compact cell would drag
    a neighbouring dim region's estimate toward its own motion. With NCC
    scoring, each block's flow reflects only its own content."""
    a = np.zeros(SHAPE, dtype="float32")
    b = np.zeros(SHAPE, dtype="float32")
    # dim, broad region (rate 1) inside block (row 1, col 1) moving -2 px x
    a[40:60, 36:60] = 1.0
    b[40:60, 34:58] = 1.0
    # bright, compact cell (rate 20) inside block (row 1, col 2) moving +3 px x
    a[40:50, 74:84] = 20.0
    b[40:50, 77:87] = 20.0

    _vy, vx, valid = block_flow(a, b, max_shift=8, blocks=4)
    assert valid[1, 1] and valid[1, 2]
    assert vx[1, 1] == pytest.approx(-2.0, abs=0.5)
    assert vx[1, 2] == pytest.approx(3.0, abs=0.5)


def test_old_unnormalised_estimator_overshoots_ncc_does_not():
    """Reproduces the mass-bias failure mode that motivated the switch to
    NCC: a raw dot product on a block with an intensity gradient chases the
    higher-mass side of the search window and saturates at the search
    radius regardless of the true shift; NCC recovers the true shift."""
    h, w = SHAPE
    a = np.zeros((h, w), dtype="float32")
    b = np.zeros((h, w), dtype="float32")
    x0, x1, true_shift = 20, 100, 5
    ramp = np.linspace(1.0, 10.0, x1 - x0)  # intensity increases across the band
    a[40:88, x0:x1] = ramp
    b[40:88, x0 + true_shift:x1 + true_shift] = ramp

    def _raw_dot_dx(max_shift):
        la, lb = np.log1p(a), np.log1p(b)
        y0, y1, cx0, cx1 = 32, 64, 0, 32  # a block fully inside the ramp
        ref = la[y0:y1, cx0 + 20:cx1 + 40]
        best, bdx = -np.inf, 0
        for dx in range(-max_shift, max_shift + 1):
            xx0, xx1 = cx0 + 20 + dx, cx1 + 40 + dx
            if xx0 < 0 or xx1 > w:
                continue
            score = float((ref * lb[y0:y1, xx0:xx1]).sum())
            if score > best:
                best, bdx = score, dx
        return bdx

    old_dx = _raw_dot_dx(max_shift=7)
    old_error = abs(old_dx - true_shift) / true_shift
    assert old_error > 0.15  # reproduces the overshoot (here: saturates at the search radius)

    _vy, vx, _valid = block_flow(a, b, max_shift=7, blocks=4)
    # block (row 1, col 1) covers cols 32:64, fully inside the ramp
    ncc_dx = float(vx[1, 1])
    ncc_error = abs(ncc_dx - true_shift) / true_shift
    assert ncc_error < 0.10


# ──────────────────────────────────────────────────────────── edge cases


def test_dry_field_returns_zero_flow_without_nan():
    z = np.zeros(SHAPE, dtype="float32")
    fy, fx = flow_field(z, z)
    assert not np.isnan(fy).any()
    assert not np.isnan(fx).any()
    assert float(fy.max()) == 0.0
    assert float(fx.max()) == 0.0

    vy, vx, valid = block_flow(z, z)
    assert not valid.any()
    assert (vy == 0).all() and (vx == 0).all()


def test_too_few_wet_cells_flags_block_invalid_with_zero_flow():
    a = np.zeros(SHAPE, dtype="float32")
    b = np.zeros(SHAPE, dtype="float32")
    a[40, 40] = 1.0  # single wet cell: well under the wet-fraction gate
    b[40, 43] = 1.0
    vy, vx, valid = block_flow(a, b, max_shift=8, blocks=4)
    assert not valid.any()
    assert (vy == 0).all() and (vx == 0).all()


def test_nan_input_raises_clearly():
    a = _rain_blob()
    b = _rain_blob(cx=70)
    a_nan = a.copy()
    a_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        block_flow(a_nan, b)


def test_warp_moves_content_by_plus_d():
    field = np.zeros((40, 40), dtype="float32")
    field[12:18, 12:18] = 10.0

    def centroid(f):
        total = f.sum()
        yy, xx = np.meshgrid(np.arange(f.shape[0]), np.arange(f.shape[1]), indexing="ij")
        return np.array([(f * yy).sum() / total, (f * xx).sum() / total])

    c0 = centroid(field)
    warped = warp(field, np.full_like(field, 3.0), np.full_like(field, 5.0))
    c1 = centroid(warped)
    np.testing.assert_allclose(c1 - c0, [3.0, 5.0], atol=0.6)
