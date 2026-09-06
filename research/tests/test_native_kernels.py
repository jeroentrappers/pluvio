"""The compiled kernels must be bit-identical to the Python reference.

Skipped when research/native isn't built (a box without a compiler runs the
Python path). PLUVIO_NATIVE=0 forces the reference, which is how each test
gets its expected values.
"""

from __future__ import annotations

import numpy as np
import pytest

from model import motion

native = pytest.importorskip("pluvio_native", reason="research/native not built")

# Every compiled backend importable here is held to the same contract: the
# C++ (research/native) and the Rust (research/native_rs) must both be
# bit-for-bit identical to the Python reference AND to each other.
BACKENDS = {"c++": native}
try:  # pragma: no cover - depends on what is built on this machine
    import pluvio_native_rs

    BACKENDS["rust"] = pluvio_native_rs
except ImportError:
    pass


def _reference_block_flow(a, b, **kw):
    saved = motion._NATIVE
    motion._NATIVE = None                      # force the Python reference
    try:
        return motion.block_flow(a, b, **kw)
    finally:
        motion._NATIVE = saved


def _wet_pair(seed: int, hw=(192, 192), n_cells=6, shift=(3, -2)):
    rng = np.random.default_rng(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros(hw, dtype="float32")
    margin = max(2.0, min(h, w) * 0.1)
    for _ in range(n_cells):
        cy, cx = rng.uniform(margin, h - margin), rng.uniform(margin, w - margin)
        amp, sig = rng.uniform(1, 15), rng.uniform(3, 9)
        a += (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig**2))).astype("float32")
    b = np.roll(np.roll(a, shift[0], axis=0), shift[1], axis=1)
    b += rng.normal(0, 0.05, hw).astype("float32")
    return a, np.clip(b, 0, None).astype("float32")


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("subpixel", [False, True])
def test_block_flow_matches_the_python_reference_bit_for_bit(backend, seed, subpixel, monkeypatch):
    monkeypatch.setattr(motion, "_NATIVE", BACKENDS[backend])
    a, b = _wet_pair(seed)
    kw = {"max_shift": 12, "blocks": 4, "subpixel": subpixel}
    vy_n, vx_n, ok_n = motion.block_flow(a, b, **kw)
    vy_p, vx_p, ok_p = _reference_block_flow(a, b, **kw)
    np.testing.assert_array_equal(vy_n, vy_p)
    np.testing.assert_array_equal(vx_n, vx_p)
    np.testing.assert_array_equal(ok_n, ok_p)
    assert ok_n.any()


def test_block_flow_degenerate_cases_match():
    dry = np.zeros((64, 64), dtype="float32")
    for a, b in [(dry, dry), (dry, _wet_pair(3, (64, 64))[0]), (_wet_pair(4, (64, 64))[0], dry)]:
        n = motion.block_flow(a, b, max_shift=6, blocks=4)
        p = _reference_block_flow(a, b, max_shift=6, blocks=4)
        for x, y in zip(n, p, strict=True):
            np.testing.assert_array_equal(x, y)
    # a block grid finer than the field (empty blocks) and a flat non-zero field
    flat = np.full((9, 9), 2.0, dtype="float32")
    n = motion.block_flow(flat, flat, max_shift=2, blocks=4)
    p = _reference_block_flow(flat, flat, max_shift=2, blocks=4)
    for x, y in zip(n, p, strict=True):
        np.testing.assert_array_equal(x, y)


def test_block_flow_still_raises_on_non_finite_input():
    a, b = _wet_pair(5, (32, 32))
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        motion.block_flow(a, b, max_shift=3, blocks=2)


def _polar_reference(vals, row, col, heights, inb, fill_cells, fill_bins, h, wd, max_beam_m):
    ok = np.isfinite(vals) & (heights <= max_beam_m) & inb
    acc = np.zeros((h, wd), "f8")
    cnt = np.zeros((h, wd), "i8")
    np.add.at(acc, (row[ok], col[ok]), vals[ok])
    np.add.at(cnt, (row[ok], col[ok]), 1)
    out = np.full((h, wd), np.nan, "f4")
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]
    need = ~hit.ravel()[fill_cells]
    good = np.isfinite(vals[fill_bins]) & (heights[fill_bins] <= max_beam_m)
    sel = need & good
    out.ravel()[fill_cells[sel]] = vals[fill_bins[sel]]
    return out


def test_polar_bin_matches_the_numpy_reference():
    rng = np.random.default_rng(7)
    h = wd = 96
    n = 40_000
    vals = rng.uniform(0, 20, n)
    vals[rng.random(n) < 0.1] = np.nan                       # missing bins
    row = rng.integers(-3, h + 3, n)                          # some out of range
    col = rng.integers(-3, wd + 3, n)
    inb = (row >= 0) & (row < h) & (col >= 0) & (col < wd)
    heights = rng.uniform(0, 4000, n)
    row_c, col_c = np.clip(row, 0, h - 1), np.clip(col, 0, wd - 1)
    fill_cells = rng.integers(0, h * wd, 5000)
    fill_bins = rng.integers(0, n, 5000)
    args = (vals, row_c, col_c, heights, inb, fill_cells, fill_bins, h, wd, 2000.0)
    got = native.polar_bin(np.ascontiguousarray(vals), np.ascontiguousarray(row_c, dtype="int64"),
                           np.ascontiguousarray(col_c, dtype="int64"), np.ascontiguousarray(heights),
                           np.ascontiguousarray(inb), np.ascontiguousarray(fill_cells, dtype="int64"),
                           np.ascontiguousarray(fill_bins, dtype="int64"), h, wd, 2000.0)
    want = _polar_reference(*args)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    m = ~np.isnan(want)
    np.testing.assert_array_equal(got[m], want[m])


def test_polar_bin_all_masked_gives_all_nan():
    n, h, wd = 100, 8, 8
    vals = np.full(n, np.nan)
    z = np.zeros(n, dtype="int64")
    got = native.polar_bin(vals, z, z, np.zeros(n), np.ones(n, dtype=bool),
                           np.zeros(0, dtype="int64"), np.zeros(0, dtype="int64"), h, wd, 2000.0)
    assert np.isnan(got).all()


@pytest.mark.parametrize("threads", [1, 2, 8])
def test_thread_count_does_not_change_the_result(threads):
    a, b = _wet_pair(11)
    kw = {"max_shift": 10, "blocks": 4}
    ref = motion.block_flow(a, b, threads=1, **kw)
    got = motion.block_flow(a, b, threads=threads, **kw)
    for x, y in zip(ref, got, strict=True):
        np.testing.assert_array_equal(x, y)


def test_python_fallback_when_the_extension_is_disabled(monkeypatch):
    a, b = _wet_pair(12, (48, 48))
    monkeypatch.setenv("PLUVIO_NATIVE", "0")
    monkeypatch.setattr(motion, "_NATIVE", motion._UNSET)
    assert motion._native() is None
    fallback = motion.block_flow(a, b, max_shift=6, blocks=4)
    monkeypatch.setattr(motion, "_NATIVE", native)
    for x, y in zip(fallback, motion.block_flow(a, b, max_shift=6, blocks=4), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.skipif(len(BACKENDS) < 2, reason="only one compiled backend built here")
def test_every_compiled_backend_agrees_with_every_other():
    """The C++ and the Rust kernels must be interchangeable, not merely both
    'close enough' to Python — a caller may run either."""
    a, b = _wet_pair(21)
    la, lb = np.log1p(np.maximum(a, 0.0)), np.log1p(np.maximum(b, 0.0))
    wet = la > np.log1p(motion.WET_THR)
    outs = {name: mod.block_flow(la, lb, wet, 12, 4, motion.MIN_WET_FRAC, True, 1)
            for name, mod in BACKENDS.items()}
    ref_name, ref = next(iter(outs.items()))
    for name, got in outs.items():
        for x, y in zip(ref, got, strict=True):
            np.testing.assert_array_equal(x, y, err_msg=f"{name} differs from {ref_name}")

    rng = np.random.default_rng(3)
    n, h, wd = 5000, 32, 32
    args = (rng.uniform(0, 20, n), rng.integers(0, h, n).astype("int64"),
            rng.integers(0, wd, n).astype("int64"), rng.uniform(0, 4000, n),
            rng.random(n) > 0.2, rng.integers(0, h * wd, 500).astype("int64"),
            rng.integers(0, n, 500).astype("int64"), h, wd, 2000.0)
    fields = {name: mod.polar_bin(*args) for name, mod in BACKENDS.items()}
    ref_name, ref = next(iter(fields.items()))
    for name, got in fields.items():
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref), err_msg=name)
        np.testing.assert_array_equal(got[~np.isnan(ref)], ref[~np.isnan(ref)], err_msg=name)
