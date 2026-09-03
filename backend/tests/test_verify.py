"""Georeference of the observed composite in the Verify view.

Regression for the reviewed defect: `QPE_BOUNDS = (1.5, 48.9, 7.5, 52.5)` was
the forecast SERVING box, not the research analysis grid the QPE day zarrs are
composited onto. It squashed the whole 768x768 composite onto the 100x100
serving box (so station truth was read ~237 km away) and, because it made the
two boxes identical, hid the fact that the serving box reaches 0.5 deg further
south than the composite ever covers.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import numpy as np
import pytest

from pluvio_backend import verify

QPE_N = 768
BE_BOUNDS = (1.5, 48.9, 7.5, 52.5)  # tools/forecast_archive.py npz bounds
BE_N = 100
BRUSSELS = (50.85, 4.35)  # lat, lon
SERVING_BOX_DEFECT = (1.5, 48.9, 7.5, 52.5)

VALID = int(datetime(2026, 9, 2, 0, 30, tzinfo=UTC).timestamp())


def _write_qpe_day(root: pathlib.Path, epoch: int, frame: np.ndarray, bounds=None) -> None:
    """A (288, N, N) day zarr with a single slot chunk materialised."""
    import zarr

    ts = datetime.fromtimestamp(epoch, UTC)
    path = root / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    arr = g.create_array(
        "rate",
        shape=(288, *frame.shape),
        dtype="float16",
        chunks=(1, *frame.shape),
        fill_value=np.nan,
    )
    arr[(epoch % 86400) // 300] = frame.astype("float16")
    if bounds is not None:
        g.attrs["bounds"] = [float(x) for x in bounds]


def _composite_with_patch() -> tuple[np.ndarray, tuple[int, int]]:
    w, s, e, n = verify.RESEARCH_GRID_BOUNDS
    lat, lon = BRUSSELS
    row = int((n - lat) / (n - s) * QPE_N)
    col = int((lon - w) / (e - w) * QPE_N)
    frame = np.zeros((QPE_N, QPE_N), dtype="float32")
    frame[row - 4 : row + 5, col - 4 : col + 5] = 8.0
    return frame, (row, col)


def test_research_grid_bounds_is_not_the_serving_box() -> None:
    assert verify.RESEARCH_GRID_BOUNDS != SERVING_BOX_DEFECT
    w, s, e, n = verify.RESEARCH_GRID_BOUNDS
    lat, lon = BRUSSELS
    assert w < lon < e and s < lat < n
    # the serving box reaches south of the composite: they are not nested
    assert BE_BOUNDS[1] < s


def test_brussels_lands_in_the_expected_composite_cell() -> None:
    _frame, (row, col) = _composite_with_patch()
    assert (row, col) == (602, 302)  # measured during review


def test_store_bounds_attr_wins_over_the_fallback(tmp_path, monkeypatch) -> None:
    import zarr

    attr_bounds = (0.0, 45.0, 10.0, 55.0)
    frame = np.zeros((16, 16), dtype="float32")
    _write_qpe_day(tmp_path, VALID, frame, bounds=attr_bounds)
    ts = datetime.fromtimestamp(VALID, UTC)
    root = zarr.open_group(str(tmp_path / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"), mode="r")
    assert verify._store_bounds(root) == pytest.approx(attr_bounds)


def test_store_bounds_falls_back_when_no_attr(tmp_path) -> None:
    import zarr

    _write_qpe_day(tmp_path, VALID, np.zeros((16, 16), dtype="float32"))
    ts = datetime.fromtimestamp(VALID, UTC)
    root = zarr.open_group(str(tmp_path / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"), mode="r")
    assert verify._store_bounds(root) == pytest.approx(verify.RESEARCH_GRID_BOUNDS)


def test_observed_on_averages_in_place_and_leaves_uncovered_nan(tmp_path, monkeypatch) -> None:
    frame, _cell = _composite_with_patch()
    _write_qpe_day(tmp_path, VALID, frame)
    monkeypatch.setattr(verify, "QPE_ROOT", tmp_path)

    obs = verify.observed_on(VALID, BE_BOUNDS, (BE_N, BE_N))
    assert obs is not None

    bw, bs, be_, bn = BE_BOUNDS
    lat, lon = BRUSSELS
    frow = int((bn - lat) / (bn - bs) * BE_N)
    fcol = int((lon - bw) / (be_ - bw) * BE_N)
    assert (frow, fcol) == (45, 47)
    # the patch fills this target cell's whole footprint -> honest mean of 8.0
    assert obs[frow, fcol] == pytest.approx(8.0)
    # away from the patch the composite measured dry, so 0.0 (not NaN)
    assert obs[10, 10] == pytest.approx(0.0)
    # rows entirely south of the composite are unobserved, not dry
    south = int((bn - verify.RESEARCH_GRID_BOUNDS[1]) / (bn - bs) * BE_N) + 1
    assert np.isnan(obs[south:, :]).all()
    assert np.isfinite(obs).sum() < BE_N * BE_N


def test_observed_on_wraps_the_last_slot_of_a_day(tmp_path, monkeypatch) -> None:
    late = int(datetime(2026, 9, 2, 23, 59, tzinfo=UTC).timestamp())
    next_midnight = int(datetime(2026, 9, 3, 0, 0, tzinfo=UTC).timestamp())
    frame = np.full((QPE_N, QPE_N), 3.0, dtype="float32")
    _write_qpe_day(tmp_path, next_midnight, frame)
    monkeypatch.setattr(verify, "QPE_ROOT", tmp_path)

    obs = verify.observed_on(late, BE_BOUNDS, (BE_N, BE_N))
    assert obs is not None
    assert obs[45, 47] == pytest.approx(3.0)


def test_scores_only_counts_observed_cells(tmp_path, monkeypatch) -> None:
    frame, _cell = _composite_with_patch()
    _write_qpe_day(tmp_path, VALID, frame)
    monkeypatch.setattr(verify, "QPE_ROOT", tmp_path)

    obs = verify.observed_on(VALID, BE_BOUNDS, (BE_N, BE_N))
    assert obs is not None

    # A forecast that is exactly right wherever the composite observed, and
    # arbitrary (5 mm/h) where it did not: only the observed cells may count,
    # so bias must be 0 and CSI 1 despite the garbage over the uncovered part.
    issue = VALID - 30 * 60
    ts = datetime.fromtimestamp(issue, UTC)
    day_dir = tmp_path / "fc" / f"{ts:%Y/%m/%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    rates = np.where(np.isfinite(obs), obs, 5.0)[None, :, :].astype("float16")
    np.savez_compressed(
        day_dir / f"forecast_{ts:%H%M}.npz",
        leads=np.asarray([30], dtype="int32"),
        rates=rates,
        bounds=np.asarray(BE_BOUNDS, dtype="float64"),
    )
    monkeypatch.setattr(verify, "ARCHIVE_ROOT", tmp_path / "fc")

    out = verify.scores(issue, 30)
    assert out is not None
    assert out["n_valid"] == int(np.isfinite(obs).sum())
    assert out["n_valid"] < BE_N * BE_N
    assert out["csi_1.0"] == pytest.approx(1.0)
    assert out["bias_mm_h"] == pytest.approx(0.0)
    assert out["mae_mm_h"] == pytest.approx(0.0)
