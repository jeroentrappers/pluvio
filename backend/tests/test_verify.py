"""Georeference of the observed composite in the Verify view.

Regression for the reviewed defect: `QPE_BOUNDS = (1.5, 48.9, 7.5, 52.5)` was
the forecast SERVING box, not the grid the QPE day zarrs are composited onto.
It squashed the whole 768x768 composite onto the 100x100 serving box, so
station truth was read ~237 km away, and it made the target-grid validity mask
vacuous.

The store's own `bounds` attr is the only accepted source. Re-deriving the
analysis grid from the research package would be ~60 km out at the south edge:
the production archiver runs from /opt/pluvio/radarproc, whose model/geo.py
predates both the 700/765 trim and the registration bias, so its stores are
binned onto PROD_QPE_BOUNDS below rather than the research derivation.
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
# Measured on hetz1: what the archived day-stores are actually binned onto.
PROD_QPE_BOUNDS = (0.0, 48.895301818847656, 10.856452941894531, 55.973602294921875)
# What a derivation from the research checkout would return instead.
DERIVED_FROM_RESEARCH = (0.07, 49.4386863708, 10.9264535904, 55.9736022949)

VALID = int(datetime(2026, 9, 2, 0, 30, tzinfo=UTC).timestamp())


def _write_qpe_day(
    root: pathlib.Path, epoch: int, frame: np.ndarray, bounds=PROD_QPE_BOUNDS
) -> None:
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
    """0 mm/h (measured dry) with a 9x9 patch of 8 mm/h on Brussels, and NaN
    south of ~49.5 deg where no radar reaches — inside the serving box, so the
    target-grid validity mask has to carry it through."""
    w, s, e, n = PROD_QPE_BOUNDS
    lat, lon = BRUSSELS
    row = int((n - lat) / (n - s) * QPE_N)
    col = int((lon - w) / (e - w) * QPE_N)
    frame = np.zeros((QPE_N, QPE_N), dtype="float32")
    frame[row - 4 : row + 5, col - 4 : col + 5] = 8.0
    frame[int((n - 49.5) / (n - s) * QPE_N) :, :] = np.nan
    return frame, (row, col)


def test_prod_bounds_is_neither_the_serving_box_nor_a_research_derivation() -> None:
    assert PROD_QPE_BOUNDS != SERVING_BOX_DEFECT
    assert PROD_QPE_BOUNDS != DERIVED_FROM_RESEARCH
    w, s, e, n = PROD_QPE_BOUNDS
    lat, lon = BRUSSELS
    assert w < lon < e and s < lat < n
    # the research derivation's south edge is ~60 km north of the real one
    assert (DERIVED_FROM_RESEARCH[1] - s) * 111 > 55
    # and verify.py must not carry a derivation to fall back to
    assert not hasattr(verify, "RESEARCH_GRID_BOUNDS")


def test_brussels_lands_in_the_expected_composite_cell() -> None:
    _frame, (row, col) = _composite_with_patch()
    assert (row, col) == (555, 307)  # measured on hetz1
    # a research derivation would have read row 602 — 47 rows, ~48 km north
    _w, s, _e, n = DERIVED_FROM_RESEARCH
    assert int((n - BRUSSELS[0]) / (n - s) * QPE_N) == 602


def test_store_bounds_reads_the_attr(tmp_path) -> None:
    import zarr

    attr_bounds = (0.0, 45.0, 10.0, 55.0)
    frame = np.zeros((16, 16), dtype="float32")
    _write_qpe_day(tmp_path, VALID, frame, bounds=attr_bounds)
    ts = datetime.fromtimestamp(VALID, UTC)
    zp = tmp_path / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    root = zarr.open_group(str(zp), mode="r")
    assert verify._store_bounds(root, zp) == pytest.approx(attr_bounds)


def test_store_without_bounds_attr_is_an_error(tmp_path, monkeypatch) -> None:
    """No fallback: every candidate derivation is wrong by tens of km, so a
    store that does not say where it is has to stop the request."""
    _write_qpe_day(tmp_path, VALID, np.zeros((16, 16), dtype="float32"), bounds=None)
    ts = datetime.fromtimestamp(VALID, UTC)
    zp = tmp_path / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    monkeypatch.setattr(verify, "QPE_ROOT", tmp_path)
    with pytest.raises(verify.QpeGeometryError) as exc:
        verify.observed_on(VALID, BE_BOUNDS, (BE_N, BE_N))
    assert str(zp) in str(exc.value)
    assert "bounds" in str(exc.value)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [5.0, 1.0, 1.0, 2.0], "nope"])
def test_unusable_bounds_attr_is_an_error(tmp_path, monkeypatch, bad) -> None:
    import zarr

    _write_qpe_day(tmp_path, VALID, np.zeros((16, 16), dtype="float32"), bounds=None)
    ts = datetime.fromtimestamp(VALID, UTC)
    zp = tmp_path / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    zarr.open_group(str(zp), mode="a").attrs["bounds"] = bad
    monkeypatch.setattr(verify, "QPE_ROOT", tmp_path)
    with pytest.raises(verify.QpeGeometryError):
        verify.observed_on(VALID, BE_BOUNDS, (BE_N, BE_N))


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
    # rows wholly inside the composite's uncovered region stay unobserved
    # (lat 49.5 is target row ~83.3, so row 85 down is entirely below it)
    assert np.isnan(obs[85:, :]).all()
    assert np.isfinite(obs[80, :]).all()
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


def test_regrid_accumulates_in_float64() -> None:
    """The integral image is a running total over the whole 768^2 field. In
    float32 it reaches ~1e6, where the representable spacing is ~0.06 mm/h, so
    a block mean near the far corner drifts into the third decimal `scores()`
    reports. Measured below: ~6e-3 mm/h."""
    rng = np.random.default_rng(0)
    src = rng.random((768, 768)).astype("float32") * 3.0
    src[300:500, 300:500] = 60.0  # a heavy core to load the sum
    src = src.astype("float16").astype("float32")  # as the store holds it
    box = (0.0, 0.0, 8.0, 8.0)

    # 192x192 target over the same box -> each target cell is exactly 4x4
    # source cells, so the expected value is a plain mean of a known window.
    out = verify._regrid_block_mean(src, box, box, (192, 192))
    expected = float(src[760:764, 760:764].astype("float64").mean())
    assert out[190, 190] == pytest.approx(expected, abs=1e-9)

    f32 = np.zeros((769, 769), "float32")
    f32[1:, 1:] = src.cumsum(0).cumsum(1)
    naive = float(f32[764, 764] - f32[760, 764] - f32[764, 760] + f32[760, 760]) / 16
    assert abs(naive - expected) > 1e-3


def test_regrid_handles_a_target_box_hanging_off_the_source() -> None:
    """The two boxes are nested today, but nothing guarantees it — they were
    not under the reviewed defect's bounds. A target cell with no source
    footprint must be NaN, not an edge value stretched to fill it."""
    src = np.ones((4, 4), dtype="float32")
    out = verify._regrid_block_mean(src, (0.0, 0.0, 4.0, 4.0), (2.0, 0.0, 6.0, 4.0), (2, 2))
    assert out[:, 0] == pytest.approx([1.0, 1.0])
    assert np.isnan(out[:, 1]).all()
    away = verify._regrid_block_mean(
        src, (0.0, 0.0, 4.0, 4.0), (10.0, 0.0, 14.0, 4.0), (2, 2)
    )
    assert np.isnan(away).all()
