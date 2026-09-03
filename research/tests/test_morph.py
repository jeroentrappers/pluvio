"""backend/src/pluvio_backend/morph.py — motion warp/blend, imported by path.

morph.py lives under backend/ (it's the serving-side nowcast-band densifier)
but is pure numpy, so we load it directly from its file path rather than
depending on the backend package being importable/installed.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_MORPH_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "backend" / "src" / "pluvio_backend" / "morph.py"
)


def _load_morph():
    spec = importlib.util.spec_from_file_location("pluvio_morph", _MORPH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if not _MORPH_PATH.exists():
    pytest.skip(f"morph.py not found at {_MORPH_PATH}", allow_module_level=True)

morph = _load_morph()


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
