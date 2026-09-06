//! Rust port of research/native's kernels (see research/docs/native_kernels.md).
//!
//! Same arithmetic, same order of operations, same masks as the Python
//! reference and the C++ build — `research/tests/test_native_kernels.py`
//! asserts all three agree bit-for-bit. Threading is per block, exactly like
//! the C++ version, so the comparison measures the language/codegen, not a
//! different algorithm.

use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::thread;

const EPS: f64 = 1e-6; // motion._EPS

/// Mean-subtracted, std-normalised cross correlation, float64 accumulation.
/// `ref_mean`/`ref_norm` are hoisted by the caller (offset-independent).
#[inline]
fn ncc(a: &[f32], a_off: usize, b: &[f32], b_off: usize, stride: usize,
       bh: usize, bw: usize, ref_mean: f64, ref_norm: f64) -> f64 {
    let mut csum = 0.0f64;
    for y in 0..bh {
        let row = &b[b_off + y * stride..b_off + y * stride + bw];
        for &v in row {
            csum += v as f64;
        }
    }
    let cmean = csum / (bh * bw) as f64;
    let mut dot = 0.0f64;
    let mut cnorm2 = 0.0f64;
    for y in 0..bh {
        let arow = &a[a_off + y * stride..a_off + y * stride + bw];
        let brow = &b[b_off + y * stride..b_off + y * stride + bw];
        for x in 0..bw {
            let rv = arow[x] as f64 - ref_mean;
            let cv = brow[x] as f64 - cmean;
            dot += rv * cv;
            cnorm2 += cv * cv;
        }
    }
    let denom = ref_norm * cnorm2.sqrt();
    if denom < EPS {
        return f64::NEG_INFINITY;
    }
    dot / denom
}

#[inline]
fn parabolic_offset(sm: f64, s0: f64, sp: f64) -> f64 {
    let denom = sm - 2.0 * s0 + sp;
    if !denom.is_finite() || denom.abs() < 1e-9 {
        return 0.0;
    }
    (0.5 * (sm - sp) / denom).clamp(-0.5, 0.5)
}

/// numpy's `np.linspace(0, n, blocks + 1).astype(int)` — truncation.
fn linspace_edges(n: usize, blocks: usize) -> Vec<usize> {
    (0..=blocks)
        .map(|i| ((n as f64) * (i as f64) / (blocks as f64)) as usize)
        .collect()
}

struct BlockOut {
    bi: usize,
    bj: usize,
    vy: f32,
    vx: f32,
}

#[allow(clippy::too_many_arguments)]
fn block_flow_impl(la: ArrayView2<f32>, lb: ArrayView2<f32>, wet: ArrayView2<bool>,
                   max_shift: i64, blocks: usize, min_wet_frac: f64, subpixel: bool,
                   threads: i64) -> (Array2<f32>, Array2<f32>, Array2<bool>) {
    let (h, w) = (la.shape()[0], la.shape()[1]);
    let a = la.as_slice().expect("la must be C-contiguous");
    let b = lb.as_slice().expect("lb must be C-contiguous");
    let wm = wet.as_slice().expect("wet_a must be C-contiguous");
    let ys = linspace_edges(h, blocks);
    let xs = linspace_edges(w, blocks);
    let m = max_shift;
    let span = (2 * m + 1) as usize;

    let n_items = blocks * blocks;
    let hw = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n_threads = if threads > 0 {
        (threads as usize).min(n_items)
    } else {
        hw.min(n_items)
    }
    .max(1);

    let one = |t0: usize| -> Vec<BlockOut> {
        let mut out = Vec::new();
        let mut scores = vec![f64::NEG_INFINITY; span * span];
        let mut item = t0;
        while item < n_items {
            let (bi, bj) = (item / blocks, item % blocks);
            item += n_threads;
            let (y0, y1) = (ys[bi], ys[bi + 1]);
            let (x0, x1) = (xs[bj], xs[bj + 1]);
            if y1 <= y0 || x1 <= x0 {
                continue;
            }
            let (bh, bw) = (y1 - y0, x1 - x0);
            let mut n_wet = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    if wm[y * w + x] {
                        n_wet += 1;
                    }
                }
            }
            if (n_wet as f64) / ((bh * bw) as f64) < min_wet_frac {
                continue;
            }
            let mut rsum = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    rsum += a[y * w + x] as f64;
                }
            }
            let rmean = rsum / ((bh * bw) as f64);
            let mut rnorm2 = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = a[y * w + x] as f64 - rmean;
                    rnorm2 += d * d;
                }
            }
            let rnorm = rnorm2.sqrt();

            scores.iter_mut().for_each(|s| *s = f64::NEG_INFINITY);
            let mut best = f64::NEG_INFINITY;
            let (mut bdy, mut bdx) = (0i64, 0i64);
            for dy in -m..=m {
                let (yy0, yy1) = (y0 as i64 + dy, y1 as i64 + dy);
                if yy0 < 0 || yy1 > h as i64 {
                    continue;
                }
                for dx in -m..=m {
                    let (xx0, xx1) = (x0 as i64 + dx, x1 as i64 + dx);
                    if xx0 < 0 || xx1 > w as i64 {
                        continue;
                    }
                    let s = ncc(a, y0 * w + x0, b, (yy0 as usize) * w + xx0 as usize,
                                w, bh, bw, rmean, rnorm);
                    scores[((dy + m) as usize) * span + (dx + m) as usize] = s;
                    if s > best {
                        best = s;
                        bdy = dy;
                        bdx = dx;
                    }
                }
            }
            let (mut fdy, mut fdx) = (bdy as f64, bdx as f64);
            if subpixel {
                let at = |dy: i64, dx: i64| -> f64 {
                    if dy < -m || dy > m || dx < -m || dx > m {
                        f64::NAN
                    } else {
                        scores[((dy + m) as usize) * span + (dx + m) as usize]
                    }
                };
                let (sym, syp) = (at(bdy - 1, bdx), at(bdy + 1, bdx));
                if sym.is_finite() && syp.is_finite() {
                    fdy += parabolic_offset(sym, best, syp);
                }
                let (sxm, sxp) = (at(bdy, bdx - 1), at(bdy, bdx + 1));
                if sxm.is_finite() && sxp.is_finite() {
                    fdx += parabolic_offset(sxm, best, sxp);
                }
            }
            out.push(BlockOut { bi, bj, vy: fdy as f32, vx: fdx as f32 });
        }
        out
    };

    let results: Vec<BlockOut> = if n_threads == 1 {
        one(0)
    } else {
        thread::scope(|scope| {
            let handles: Vec<_> = (1..n_threads).map(|t| scope.spawn(move || one(t))).collect();
            let mut all = one(0);
            for hdl in handles {
                all.extend(hdl.join().expect("worker thread panicked"));
            }
            all
        })
    };

    let mut vy = Array2::<f32>::zeros((blocks, blocks));
    let mut vx = Array2::<f32>::zeros((blocks, blocks));
    let mut valid = Array2::<bool>::from_elem((blocks, blocks), false);
    for r in results {
        vy[[r.bi, r.bj]] = r.vy;
        vx[[r.bi, r.bj]] = r.vx;
        valid[[r.bi, r.bj]] = true;
    }
    (vy, vx, valid)
}

#[allow(clippy::too_many_arguments)]
fn polar_bin_impl(vals: ArrayView1<f64>, row: ArrayView1<i64>, col: ArrayView1<i64>,
                  heights: ArrayView1<f64>, inb: ArrayView1<bool>,
                  fill_cells: ArrayView1<i64>, fill_bins: ArrayView1<i64>,
                  h: usize, wd: usize, max_beam_m: f64) -> Array2<f32> {
    let n = vals.len();
    let mut acc = vec![0.0f64; h * wd];
    let mut cnt = vec![0i64; h * wd];
    for i in 0..n {
        let v = vals[i];
        if !v.is_finite() || !(heights[i] <= max_beam_m) || !inb[i] {
            continue;
        }
        let k = (row[i] as usize) * wd + col[i] as usize;
        acc[k] += v;
        cnt[k] += 1;
    }
    let mut out = vec![f32::NAN; h * wd];
    for k in 0..acc.len() {
        if cnt[k] > 0 {
            out[k] = (acc[k] / cnt[k] as f64) as f32;
        }
    }
    for i in 0..fill_cells.len() {
        let cell = fill_cells[i] as usize;
        if cnt[cell] > 0 {
            continue;
        }
        let b = fill_bins[i] as usize;
        let v = vals[b];
        if !v.is_finite() || !(heights[b] <= max_beam_m) {
            continue;
        }
        out[cell] = v as f32;
    }
    Array2::from_shape_vec((h, wd), out).expect("shape")
}

#[pymodule]
fn pluvio_native_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "block_flow", signature = (la, lb, wet_a, max_shift, blocks, min_wet_frac, subpixel, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn block_flow<'py>(py: Python<'py>, la: PyReadonlyArray2<'py, f32>, lb: PyReadonlyArray2<'py, f32>,
                       wet_a: PyReadonlyArray2<'py, bool>, max_shift: i64, blocks: i64,
                       min_wet_frac: f64, subpixel: bool, threads: i64)
                       -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
        let (a, b, wm) = (la.as_array(), lb.as_array(), wet_a.as_array());
        if a.shape() != b.shape() || a.shape() != wm.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err("block_flow: shape mismatch"));
        }
        let (vy, vx, valid) = py.allow_threads(|| {
            block_flow_impl(a, b, wm, max_shift, blocks as usize, min_wet_frac, subpixel, threads)
        });
        Ok((vy.into_pyarray_bound(py), vx.into_pyarray_bound(py), valid.into_pyarray_bound(py)))
    }

    #[pyfn(m)]
    #[pyo3(name = "polar_bin")]
    #[allow(clippy::too_many_arguments)]
    fn polar_bin<'py>(py: Python<'py>, vals: PyReadonlyArray1<'py, f64>, row: PyReadonlyArray1<'py, i64>,
                      col: PyReadonlyArray1<'py, i64>, heights: PyReadonlyArray1<'py, f64>,
                      inb: PyReadonlyArray1<'py, bool>, fill_cells: PyReadonlyArray1<'py, i64>,
                      fill_bins: PyReadonlyArray1<'py, i64>, h: i64, wd: i64, max_beam_m: f64)
                      -> Bound<'py, PyArray2<f32>> {
        let (v, r, c) = (vals.as_array(), row.as_array(), col.as_array());
        let (hh, ib) = (heights.as_array(), inb.as_array());
        let (fc, fb) = (fill_cells.as_array(), fill_bins.as_array());
        let out = py.allow_threads(|| {
            polar_bin_impl(v, r, c, hh, ib, fc, fb, h as usize, wd as usize, max_beam_m)
        });
        out.into_pyarray_bound(py)
    }
    Ok(())
}
