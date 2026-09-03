"""Constants describing the synthetic zarr v2 store shared by the test suite.

Kept out of conftest.py so plain test modules can import these constants
directly (``from tests._store_spec import ...``) without going through
pytest's own conftest-collection import path — pulling them via
``tests.conftest`` created a second, separately-imported copy of the module
under a different name.
"""

from __future__ import annotations

N_ISSUES = 40
CADENCE_MIN = 30
GRID_N = 24
LEADS_MIN = [0, 30, 60, 90]
BOUNDS = (1.5, 48.9, 7.5, 54.2)  # (west, south, east, north) — row 0 = north

# An issue with all-NaN truth (source missing there) plus a NaN patch in its
# own radar field, mirroring a real gap in the store.
NAN_ISSUE_IDX = 20
