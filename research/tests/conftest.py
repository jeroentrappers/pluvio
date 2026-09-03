"""Shared test fixtures.

Puts ``research/`` on sys.path so tests can ``import model...`` / ``import
tools...`` the same way the scripts under research/ do (they all
``sys.path.insert`` their own parent directory), and provides a tiny
synthetic zarr v2 store that mirrors the array names/attrs conventions of
``tools/build_store_v3.py`` (regular lat/lon grid, row 0 = north, ``bounds``
+ ``grid_n`` attrs) so dataset/geo tests don't need a real (multi-GB) store.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

RESEARCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

# Synthetic store geometry — small but shaped like a real build_store_v3
# store: N issues on a 30-min cadence, 4 leads, one aux/static family, a
# "truth" array separate from "radar" (the v2 truth-array curriculum).
N_ISSUES = 40
CADENCE_MIN = 30
GRID_N = 24
LEADS_MIN = [0, 30, 60, 90]
BOUNDS = (1.5, 48.9, 7.5, 54.2)  # (west, south, east, north) — row 0 = north


@pytest.fixture()
def synthetic_store(tmp_path) -> pathlib.Path:
    """Build a tiny zarr v2 store at tmp_path/'store.zarr' and return its path."""
    import zarr

    rng = np.random.default_rng(0)
    n, leads, g = N_ISSUES, len(LEADS_MIN), GRID_N

    issue_time0 = 1_700_000_000  # arbitrary but fixed epoch second, on a 30-min grid
    issue_time = (issue_time0 + np.arange(n) * CADENCE_MIN * 60).astype("int64")

    radar = rng.random((n, leads, g, g), dtype="float64").astype("float32") * 5.0
    truth = rng.random((n, g, g), dtype="float64").astype("float32") * 5.0
    aux_a = rng.random((n, g, g), dtype="float64").astype("float32")
    aux_b = rng.random((n, g, g), dtype="float64").astype("float32")
    static_elev = rng.random((g, g), dtype="float64").astype("float32") * 100.0

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
