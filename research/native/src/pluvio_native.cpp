// Native kernels for the two hot loops the Python profile puts on the
// critical path (see research/docs/native_kernels.md):
//
//   block_flow  — NCC block-matching motion search. Python spends its time in
//                 a (2m+1)^2 x blocks^2 loop of small numpy reductions; here
//                 it is one pass of scalar arithmetic per candidate offset,
//                 with the reference block's mean/norm hoisted out.
//   polar_bin   — scatter-mean of polar radar bins onto the analysis grid plus
//                 the in-disc hole fill. Python does it with np.add.at, which
//                 is an unbuffered gather-scatter and the slowest way numpy
//                 can add.
//
// Both mirror the Python reference EXACTLY (same masks, same tie-breaking,
// same float64 accumulation) — research/tests/test_native_kernels.py asserts
// bit-for-bit equality on random and degenerate inputs, and the Python
// implementations stay as the reference/fallback.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kEps = 1e-6;   // motion._EPS

// Mean-subtracted, std-normalised cross correlation of two equally sized
// blocks, accumulated in float64 exactly like motion._ncc_score. `ref_mean`
// and `ref_norm` are hoisted by the caller (they do not depend on the offset).
inline double ncc(const float* a, long a_stride, const float* b, long b_stride,
                  long bh, long bw, double ref_mean, double ref_norm) {
    double csum = 0.0;
    for (long y = 0; y < bh; ++y) {
        const float* brow = b + y * b_stride;
        for (long x = 0; x < bw; ++x) csum += brow[x];
    }
    const double cmean = csum / static_cast<double>(bh * bw);
    double dot = 0.0, cnorm2 = 0.0;
    for (long y = 0; y < bh; ++y) {
        const float* arow = a + y * a_stride;
        const float* brow = b + y * b_stride;
        for (long x = 0; x < bw; ++x) {
            const double rv = static_cast<double>(arow[x]) - ref_mean;
            const double cv = static_cast<double>(brow[x]) - cmean;
            dot += rv * cv;
            cnorm2 += cv * cv;
        }
    }
    const double denom = ref_norm * std::sqrt(cnorm2);
    if (denom < kEps) return -std::numeric_limits<double>::infinity();
    return dot / denom;
}

inline double parabolic_offset(double sm, double s0, double sp) {
    const double denom = sm - 2.0 * s0 + sp;
    if (!std::isfinite(denom) || std::fabs(denom) < 1e-9) return 0.0;
    return std::clamp(0.5 * (sm - sp) / denom, -0.5, 0.5);
}

// numpy's np.linspace(0, n, blocks+1).astype(int) — truncation, not rounding.
std::vector<long> linspace_edges(long n, long blocks) {
    std::vector<long> e(static_cast<size_t>(blocks) + 1);
    for (long i = 0; i <= blocks; ++i)
        e[static_cast<size_t>(i)] = static_cast<long>(
            static_cast<double>(n) * static_cast<double>(i) / static_cast<double>(blocks));
    return e;
}

}  // namespace

// (vy, vx, valid) for the log1p-transformed pair, matching motion.block_flow.
// `la`/`lb` are already log1p(max(rate, 0)); `wet_a` is the block-gate mask.
py::tuple block_flow_native(py::array_t<float, py::array::c_style | py::array::forcecast> la,
                            py::array_t<float, py::array::c_style | py::array::forcecast> lb,
                            py::array_t<bool, py::array::c_style | py::array::forcecast> wet_a,
                            long max_shift, long blocks, double min_wet_frac, bool subpixel,
                            long threads) {
    auto A = la.unchecked<2>();
    auto B = lb.unchecked<2>();
    auto W = wet_a.unchecked<2>();
    const long h = A.shape(0), w = A.shape(1);
    if (B.shape(0) != h || B.shape(1) != w || W.shape(0) != h || W.shape(1) != w)
        throw std::invalid_argument("block_flow_native: shape mismatch");

    py::array_t<float> vy({blocks, blocks}), vx({blocks, blocks});
    py::array_t<bool> valid({blocks, blocks});
    auto VY = vy.mutable_unchecked<2>();
    auto VX = vx.mutable_unchecked<2>();
    auto VA = valid.mutable_unchecked<2>();
    for (long i = 0; i < blocks; ++i)
        for (long j = 0; j < blocks; ++j) { VY(i, j) = 0.0f; VX(i, j) = 0.0f; VA(i, j) = false; }

    const std::vector<long> ys = linspace_edges(h, blocks);
    const std::vector<long> xs = linspace_edges(w, blocks);
    const float* a_base = A.data(0, 0);
    const float* b_base = B.data(0, 0);
    const bool* w_base = W.data(0, 0);
    const long m = max_shift;
    const long span = 2 * m + 1;

    // One block per work item: blocks are independent (each reads the two
    // fields, writes only its own cell), so threading changes nothing about
    // the result — the arithmetic per block is untouched and bit-exact.
    float* VYp = vy.mutable_data(0, 0);
    float* VXp = vx.mutable_data(0, 0);
    bool* VAp = valid.mutable_data(0, 0);
    const long n_items = blocks * blocks;
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    long n_threads = std::min<long>(static_cast<long>(hw_threads), n_items);
    if (threads > 0) n_threads = std::min<long>(threads, n_items);
    if (n_threads < 1) n_threads = 1;

    auto worker = [&](long t0) {
        std::vector<double> scores(static_cast<size_t>(span) * static_cast<size_t>(span));
        for (long item = t0; item < n_items; item += n_threads) {
            const long bi = item / blocks, bj = item % blocks;
            const long y0 = ys[(size_t)bi], y1 = ys[(size_t)bi + 1];
            const long x0 = xs[(size_t)bj], x1 = xs[(size_t)bj + 1];
            if (y1 <= y0 || x1 <= x0) continue;
            const long bh = y1 - y0, bw = x1 - x0;

            long wet = 0;
            for (long y = y0; y < y1; ++y)
                for (long x = x0; x < x1; ++x) wet += w_base[y * w + x] ? 1 : 0;
            if (static_cast<double>(wet) / static_cast<double>(bh * bw) < min_wet_frac) continue;

            double rsum = 0.0;
            for (long y = y0; y < y1; ++y)
                for (long x = x0; x < x1; ++x) rsum += static_cast<double>(a_base[y * w + x]);
            const double rmean = rsum / static_cast<double>(bh * bw);
            double rnorm2 = 0.0;
            for (long y = y0; y < y1; ++y)
                for (long x = x0; x < x1; ++x) {
                    const double d = static_cast<double>(a_base[y * w + x]) - rmean;
                    rnorm2 += d * d;
                }
            const double rnorm = std::sqrt(rnorm2);

            std::fill(scores.begin(), scores.end(),
                      -std::numeric_limits<double>::infinity());
            double best = -std::numeric_limits<double>::infinity();
            long bdy = 0, bdx = 0;
            for (long dy = -m; dy <= m; ++dy) {
                const long yy0 = y0 + dy, yy1 = y1 + dy;
                if (yy0 < 0 || yy1 > h) continue;
                for (long dx = -m; dx <= m; ++dx) {
                    const long xx0 = x0 + dx, xx1 = x1 + dx;
                    if (xx0 < 0 || xx1 > w) continue;
                    const double s = ncc(a_base + y0 * w + x0, w,
                                         b_base + yy0 * w + xx0, w,
                                         bh, bw, rmean, rnorm);
                    scores[(size_t)((dy + m) * span + (dx + m))] = s;
                    if (s > best) { best = s; bdy = dy; bdx = dx; }
                }
            }
            double fdy = static_cast<double>(bdy), fdx = static_cast<double>(bdx);
            if (subpixel) {
                auto at = [&](long dy, long dx) -> double {
                    if (dy < -m || dy > m || dx < -m || dx > m)
                        return std::numeric_limits<double>::quiet_NaN();
                    return scores[(size_t)((dy + m) * span + (dx + m))];
                };
                const double sym = at(bdy - 1, bdx), syp = at(bdy + 1, bdx);
                if (std::isfinite(sym) && std::isfinite(syp)) fdy += parabolic_offset(sym, best, syp);
                const double sxm = at(bdy, bdx - 1), sxp = at(bdy, bdx + 1);
                if (std::isfinite(sxm) && std::isfinite(sxp)) fdx += parabolic_offset(sxm, best, sxp);
            }
            VYp[bi * blocks + bj] = static_cast<float>(fdy);
            VXp[bi * blocks + bj] = static_cast<float>(fdx);
            VAp[bi * blocks + bj] = true;
        }
    };

    {
        py::gil_scoped_release release;   // pure C++ from here: let Python run
        if (n_threads == 1) {
            worker(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<size_t>(n_threads) - 1);
            for (long t = 1; t < n_threads; ++t) pool.emplace_back(worker, t);
            worker(0);
            for (auto& th : pool) th.join();
        }
    }
    return py::make_tuple(vy, vx, valid);
}

// Scatter-mean of polar bins onto the grid + in-disc hole fill, matching
// radar_single_site.polar_to_grid's numpy body (np.add.at over float64,
// then the nearest-bin fill for cells no ray landed in).
py::array_t<float> polar_bin_native(
    py::array_t<double, py::array::c_style | py::array::forcecast> vals,
    py::array_t<long, py::array::c_style | py::array::forcecast> row,
    py::array_t<long, py::array::c_style | py::array::forcecast> col,
    py::array_t<double, py::array::c_style | py::array::forcecast> heights,
    py::array_t<bool, py::array::c_style | py::array::forcecast> inb,
    py::array_t<long, py::array::c_style | py::array::forcecast> fill_cells,
    py::array_t<long, py::array::c_style | py::array::forcecast> fill_bins,
    long h, long wd, double max_beam_m) {
    auto V = vals.unchecked<1>();
    auto R = row.unchecked<1>();
    auto C = col.unchecked<1>();
    auto H = heights.unchecked<1>();
    auto I = inb.unchecked<1>();
    const long n = V.shape(0);

    py::array_t<float> out({h, wd});
    float* O = out.mutable_data(0, 0);
    std::vector<double> acc(static_cast<size_t>(h) * static_cast<size_t>(wd), 0.0);
    std::vector<long> cnt(static_cast<size_t>(h) * static_cast<size_t>(wd), 0);
    for (long i = 0; i < n; ++i) {
        const double v = V(i);
        if (!std::isfinite(v) || !(H(i) <= max_beam_m) || !I(i)) continue;
        const size_t k = static_cast<size_t>(R(i)) * static_cast<size_t>(wd) + static_cast<size_t>(C(i));
        acc[k] += v;
        cnt[k] += 1;
    }
    for (size_t k = 0; k < acc.size(); ++k)
        O[k] = cnt[k] > 0 ? static_cast<float>(acc[k] / static_cast<double>(cnt[k]))
                          : std::numeric_limits<float>::quiet_NaN();

    auto FC = fill_cells.unchecked<1>();
    auto FB = fill_bins.unchecked<1>();
    for (long i = 0; i < FC.shape(0); ++i) {
        const size_t cell = static_cast<size_t>(FC(i));
        if (cnt[cell] > 0) continue;                       // a ray already landed here
        const long b = FB(i);
        const double v = V(b);
        if (!std::isfinite(v) || !(H(b) <= max_beam_m)) continue;
        O[cell] = static_cast<float>(v);
    }
    return out;
}

PYBIND11_MODULE(pluvio_native, m) {
    m.doc() = "Native kernels mirroring model.motion / tools.radar_single_site";
    m.def("block_flow", &block_flow_native, py::arg("la"), py::arg("lb"), py::arg("wet_a"),
          py::arg("max_shift"), py::arg("blocks"), py::arg("min_wet_frac"), py::arg("subpixel"),
          py::arg("threads") = 0);
    m.def("polar_bin", &polar_bin_native, py::arg("vals"), py::arg("row"), py::arg("col"),
          py::arg("heights"), py::arg("inb"), py::arg("fill_cells"), py::arg("fill_bins"),
          py::arg("h"), py::arg("wd"), py::arg("max_beam_m"));
}
