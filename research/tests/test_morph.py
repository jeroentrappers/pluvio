"""backend/src/pluvio_backend/morph.py — motion warp/blend, imported by path.

morph.py lives under backend/ (it's the serving-side nowcast-band densifier)
but is pure numpy, so we load it directly from its file path rather than
depending on the backend package being importable/installed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from model import motion

_MORPH_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "backend" / "src" / "pluvio_backend" / "morph.py"
)

# Flow (at block centres) and morph_pair summary stats from the dot-product
# `_block_flow` as it stood before the NCC-scoring switch (2.7), commit
# 4936690 — committed here rather than reconstructed from git history so the
# regression tests below still run in CI's depth-1 checkout.
_PRE_CHANGE_FIXTURE = json.loads(
    (pathlib.Path(__file__).resolve().parent / "fixtures" / "morph_pre_change_flow.json").read_text()
)


def _load_module_from_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_morph():
    return _load_module_from_path("pluvio_morph", _MORPH_PATH)


if not _MORPH_PATH.exists():
    pytest.skip(f"morph.py not found at {_MORPH_PATH}", allow_module_level=True)

morph = _load_morph()


def _flow_at_block_centres(flow: np.ndarray, shape: tuple[int, int]) -> tuple[list, list]:
    """Sample a full-resolution (2, H, W) flow field at the BLOCKS² block
    centres — np.interp is exact at its own knots, so this recovers the raw
    per-block (vy, vx) values `_block_flow` computed, without needing it to
    expose them separately."""
    h, w = shape
    ys = np.linspace(0, h, morph.BLOCKS + 1).astype(int)
    xs = np.linspace(0, w, morph.BLOCKS + 1).astype(int)
    cys = [(ys[i] + ys[i + 1]) // 2 for i in range(morph.BLOCKS)]
    cxs = [(xs[i] + xs[i + 1]) // 2 for i in range(morph.BLOCKS)]
    fy = [[float(flow[0][cy, cx]) for cx in cxs] for cy in cys]
    fx = [[float(flow[1][cy, cx]) for cx in cxs] for cy in cys]
    return fy, fx


def _block(shape, centre, half=3, value=10.0):
    field = np.zeros(shape, dtype="float32")
    cy, cx = centre
    field[cy - half: cy + half + 1, cx - half: cx + half + 1] = value
    return field


def _centroid(field: np.ndarray) -> np.ndarray:
    total = field.sum()
    yy, xx = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]), indexing="ij")
    return np.array([(field * yy).sum() / total, (field * xx).sum() / total])


def test_warp_moves_centroid_by_dy_dx():
    field = _block((40, 40), (15, 15))
    c0 = _centroid(field)

    dy, dx = 3, 5
    warped = morph._warp(field, np.full_like(field, dy), np.full_like(field, dx))
    c1 = _centroid(warped)

    np.testing.assert_allclose(c1 - c0, [dy, dx], atol=0.6)


def test_morph_pair_halfway_centroid_and_peak():
    a = _block((40, 40), (15, 15), value=10.0)
    b = _block((40, 40), (15, 19), value=10.0)  # b == a shifted +4 in x

    out = morph.morph_pair(a, b, 0.5)
    c_a = _centroid(a)
    c_out = _centroid(out)

    np.testing.assert_allclose(c_out - c_a, [0.0, 2.0], atol=0.75)
    assert abs(float(out.max()) - 10.0) / 10.0 < 0.10


def test_flow_for_pair_positive_fx_for_eastward_motion():
    a = _block((40, 40), (20, 15))
    b = _block((40, 40), (20, 21))  # moved east (+x), same row

    fy, fx = morph.flow_for_pair(a, b)
    assert fx.mean() > 0
    assert abs(fy.mean()) < 0.5  # pure x-motion: fy should stay ~0, not soak up fx


def test_morph_pair_matches_pre_change_output_on_isolated_cell():
    """A compact, isolated wet block (no competing mass elsewhere in the
    field) is where the old unnormalised dot-product and the new NCC score
    agree — both pick the same integer offset, so switching scorers must not
    move this case at all."""
    fixture = _PRE_CHANGE_FIXTURE["isolated_cell"]
    a = _block(tuple(fixture["shape"]), tuple(fixture["centre_a"]),
              half=fixture["half"], value=fixture["value"])
    b = _block(tuple(fixture["shape"]), tuple(fixture["centre_b"]),
              half=fixture["half"], value=fixture["value"])

    flow = morph.flow_for_pair(a, b)
    fy, fx = _flow_at_block_centres(flow, tuple(fixture["shape"]))
    np.testing.assert_allclose(fy, fixture["flow_fy_block"], atol=1e-5)
    np.testing.assert_allclose(fx, fixture["flow_fx_block"], atol=1e-5)

    out = morph.morph_pair(a, b, 0.5, flow=flow)
    assert float(out.sum()) == pytest.approx(fixture["morph_pair_sum"], abs=1e-3)
    assert float(out.max()) == pytest.approx(fixture["morph_pair_max"], abs=1e-3)


def test_morph_pair_diverges_from_pre_change_output_on_low_contrast_tail():
    """A wide Gaussian blob's low-contrast tail is exactly where the old
    dot-product scorer broke: with no dominant edge in view, it chased
    whichever candidate offset had marginally more accumulated mass and
    saturated at the search radius (±7 px) for a true +2 px shift (the
    fixture's block flow is +7/-7 throughout, never +2). The NCC-scored
    estimator does not have this bias, so it must NOT match the old output
    here; this test documents that the divergence is the fix, not a
    regression."""
    fixture = _PRE_CHANGE_FIXTURE["low_contrast_tail"]
    assert max(abs(v) for row in fixture["flow_fx_block"] for v in row) >= 6.5  # saturated, pre-change

    def smooth_blob(shape, cy, cx, sigma, rate):
        h, w = shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        return (rate * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))).astype("float32")

    shape = tuple(fixture["shape"])
    a = smooth_blob(shape, *fixture["centre_a"], fixture["sigma"], fixture["rate"])
    b = smooth_blob(shape, *fixture["centre_b"], fixture["sigma"], fixture["rate"])  # true shift: +2 px x

    flow = morph.flow_for_pair(a, b)
    _fy, fx = _flow_at_block_centres(flow, shape)
    # near the blob's core the new estimator tracks the true shift; nowhere
    # near the old fixture's saturated +7/-7.
    assert np.mean(fx[1:3]) == pytest.approx(2.0, abs=1.5)

    out = morph.morph_pair(a, b, 0.5, flow=flow)
    # total mass moved differs materially (the old estimate's saturated flow
    # over-advects the tails); a full-array atol comparison is meaningless
    # here since most cells are near-zero Gaussian tail.
    assert abs(float(out.sum()) - fixture["morph_pair_sum"]) > 100.0


def test_block_flow_matches_canonical_motion_flow_field():
    """Lockstep guard: morph._block_flow is a hand-kept copy of
    model.motion's algorithm (backend can't import research — see morph.py's
    docstring). On the same params (subpixel off, matching morph's module
    constants) the two must produce bit-identical output; a silent drift
    between the copies would otherwise only show up as a serving-vs-research
    behaviour mismatch nobody notices until production."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        shape = (40, 40)
        a = np.clip(rng.exponential(scale=1.5, size=shape) - 1.0, 0.0, None).astype("float32")
        b = np.clip(rng.exponential(scale=1.5, size=shape) - 1.0, 0.0, None).astype("float32")
        assert a.sum() > 0 and b.sum() > 0  # a genuinely wet field, not a degenerate draw

        morph_flow = morph._block_flow(a, b)
        motion_flow = motion.flow_field(a, b, max_shift=morph.MAX_SHIFT, blocks=morph.BLOCKS,
                                        wet_thr=morph.WET_THR, subpixel=False)
        np.testing.assert_array_equal(morph_flow, motion_flow)


def test_flow_for_pair_positive_fy_for_southward_motion():
    # Mirror of the eastward case, on the y axis: row 0 = north (geo.py,
    # zarr_dataset), so motion toward higher row indices is southward and
    # must show up as positive fy — catches an fy/fx axis swap that the
    # x-only case above can't.
    a = _block((40, 40), (15, 20))
    b = _block((40, 40), (21, 20))  # moved south (+y), same column

    fy, fx = morph.flow_for_pair(a, b)
    assert fy.mean() > 0
    assert abs(fx.mean()) < 0.5
