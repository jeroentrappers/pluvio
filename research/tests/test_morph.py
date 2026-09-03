"""backend/src/pluvio_backend/morph.py — motion warp/blend, imported by path.

morph.py lives under backend/ (it's the serving-side nowcast-band densifier)
but is pure numpy, so we load it directly from its file path rather than
depending on the backend package being importable/installed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import tempfile

import numpy as np
import pytest

_MORPH_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "backend" / "src" / "pluvio_backend" / "morph.py"
)

# Last commit that touched morph.py before the NCC-scoring switch (2.7) —
# used by the regression tests below as a live "pre-change" fixture instead
# of a committed binary blob.
_PRE_CHANGE_REV = "4936690"


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


def _load_pre_change_morph():
    """The dot-product `_block_flow` as it stood before the NCC switch,
    loaded straight from git history rather than a committed fixture file."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    rel = "backend/src/pluvio_backend/morph.py"
    try:
        source = subprocess.run(
            ["git", "show", f"{_PRE_CHANGE_REV}:{rel}"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("pre-change morph.py not available from git history")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(source)
        tmp_path = pathlib.Path(fh.name)
    try:
        return _load_module_from_path("pluvio_morph_pre_change", tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    old_morph = _load_pre_change_morph()
    a = _block((40, 40), (15, 15), value=10.0)
    b = _block((40, 40), (15, 19), value=10.0)
    out_old = old_morph.morph_pair(a, b, 0.5)
    out_new = morph.morph_pair(a, b, 0.5)
    np.testing.assert_allclose(out_new, out_old, atol=1e-5)


def test_morph_pair_diverges_from_pre_change_output_on_low_contrast_tail():
    """A wide Gaussian blob's low-contrast tail is exactly where the old
    dot-product scorer broke: with no dominant edge in view, it chased
    whichever candidate offset had marginally more accumulated mass and
    saturated at the search radius (±7 px) for a true ~2 px shift. The
    NCC-scored estimator does not have this bias, so it must NOT match the
    old output here; this test documents that the divergence is the fix,
    not a regression."""
    old_morph = _load_pre_change_morph()

    def smooth_blob(shape, cy, cx, sigma=8.0, rate=6.0):
        h, w = shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        return (rate * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))).astype("float32")

    a = smooth_blob((60, 60), 30, 30)
    b = smooth_blob((60, 60), 30, 32)  # true shift: +2 px in x

    _, old_fx = old_morph.flow_for_pair(a, b)
    assert float(np.abs(old_fx).max()) >= 6.5  # saturated at the search radius, pre-change

    _, new_fx = morph.flow_for_pair(a, b)
    # near the blob's core the new estimator tracks the true shift; nowhere
    # near the old estimator's saturated +7/-7.
    assert new_fx[25:35, 25:35].mean() == pytest.approx(2.0, abs=1.5)

    out_old = old_morph.morph_pair(a, b, 0.5)
    out_new = morph.morph_pair(a, b, 0.5)
    # total mass moved differs materially (the old estimate's saturated flow
    # over-advects the tails); a full-array atol comparison is meaningless
    # here since most cells are near-zero Gaussian tail.
    assert abs(float(out_new.sum()) - float(out_old.sum())) > 100.0


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
