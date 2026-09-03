"""Shared test fixtures.

``research/`` goes on sys.path via pyproject's ``[tool.pytest.ini_options]``
(``pythonpath = ["."]``, rootdir = research/) so tests can ``import model...``
/ ``import tools...`` the same way the scripts under research/ do (they all
``sys.path.insert`` their own parent directory) — not duplicated here. This
file provides a tiny synthetic zarr v2 store that mirrors the array
names/attrs/dtype conventions of ``tools/build_store_v3.py`` (regular
lat/lon grid, row 0 = north, float16 arrays, ``bounds`` + ``grid_n`` attrs)
so dataset/geo tests don't need a real (multi-GB) store. The store's own
geometry constants live in ``tests/_store_spec.py`` — this file holds only
the fixture.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tests._store_spec import (
    BOUNDS,
    CADENCE_MIN,
    GRID_N,
    LEADS_MIN,
    N_ISSUES,
    NAN_ISSUE_IDX,
)


@pytest.fixture()
def synthetic_store(tmp_path) -> pathlib.Path:
    """Build a tiny zarr v2 store at tmp_path/'store.zarr' and return its path."""
    import zarr

    rng = np.random.default_rng(0)
    n, leads, g = N_ISSUES, len(LEADS_MIN), GRID_N

    issue_time0 = 1_700_000_000  # arbitrary but fixed epoch second, on a 30-min grid
    issue_time = (issue_time0 + np.arange(n) * CADENCE_MIN * 60).astype("int64")

    radar = (rng.random((n, leads, g, g)) * 5.0).astype("float16")
    truth = (rng.random((n, g, g)) * 5.0).astype("float16")
    aux_a = rng.random((n, g, g)).astype("float16")
    aux_b = rng.random((n, g, g)).astype("float16")
    static_elev = (rng.random((g, g)) * 100.0).astype("float16")

    # NAN_ISSUE_IDX: truth entirely missing (all-NaN) — ZarrCorrectionDataset
    # must drop it as a *target* — plus a NaN patch in its own radar field, so
    # it still needs to be usable as *history* input for later issues without
    # poisoning build_input's output.
    truth[NAN_ISSUE_IDX] = np.nan
    radar[NAN_ISSUE_IDX, 0, :2, :2] = np.nan

    path = tmp_path / "store.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({
        "grid_n": GRID_N,
        "bounds": list(BOUNDS),
        "store_version": 3,
        "grid": "regular lat/lon, row 0 = north",
    })

    root.create_array("issue_time", data=issue_time, chunks="auto")
    root.create_array("leads_min", data=np.asarray(LEADS_MIN, dtype="int32"), chunks="auto")
    root.create_array("radar", data=radar, chunks=(16, leads, g, g))
    root.create_array("truth", data=truth, chunks=(16, g, g))
    root.create_array("msg_ir108", data=aux_a, chunks=(16, g, g))
    root.create_array("alaro_precip", data=aux_b, chunks=(16, g, g))
    root.create_array("static_elevation_m", data=static_elev, chunks="auto")

    return path
