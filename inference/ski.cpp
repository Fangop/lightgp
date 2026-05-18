// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "ski.h"

#ifdef LIGHTGP_HAS_ACCELERATE
#include "ski_accel.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

#include "../core/dispatch.h"

#ifdef LIGHTGP_HAS_CUDA
#include "ski_cuda.h"
#endif

namespace lightgp {

namespace {

/// Cardinal cubic interpolation weights for t ∈ [0, 1]. Indices map to grid points
/// (j-1, j, j+1, j+2) where j is the lower bound of the containing cell.
inline void cubic_weights(float t, float w[4]) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    w[0] = -0.5f * t3 +        t2 - 0.5f * t;
    w[1] =  1.5f * t3 - 2.5f * t2 + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] =  0.5f * t3 - 0.5f * t2;
}

}  // namespace

// -----------------------------------------------------------------------------
// SKIGrid
// -----------------------------------------------------------------------------

SKIGrid SKIGrid::from_data(const Tensor& X, int points_per_dim, float padding_fraction) {
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());
    if (points_per_dim <= 0) {
        const double per_dim = std::ceil(std::pow(static_cast<double>(N), 1.0 / D));
        points_per_dim = std::max(50, static_cast<int>(per_dim));
    }
    SKIGrid g;
    g.D = D;
    g.grid_sizes.assign(D, points_per_dim);
    g.grid_mins.resize(D);
    g.grid_steps.resize(D);
    g.total_points = 1;
    for (int d = 0; d < D; ++d) {
        float lo = std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < N; ++i) {
            const float v = X(i, d);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const float range = std::max(hi - lo, 1e-6f);
        const float pad = padding_fraction * range;
        g.grid_mins[d] = lo - pad;
        const float gmax = hi + pad;
        g.grid_steps[d] = (gmax - g.grid_mins[d]) / static_cast<float>(points_per_dim - 1);
        g.total_points *= points_per_dim;
    }
    return g;
}

std::vector<float> SKIGrid::grid_point(int flat_index) const {
    std::vector<float> out(D);
    int rem = flat_index;
    // axis-major (row-major over the D axes): last axis varies fastest.
    for (int d = D - 1; d >= 0; --d) {
        const int idx_d = rem % grid_sizes[d];
        rem /= grid_sizes[d];
        out[d] = grid_mins[d] + grid_steps[d] * idx_d;
    }
    return out;
}

std::vector<int> SKIGrid::strides() const {
    std::vector<int> s(D, 1);
    for (int d = D - 2; d >= 0; --d) s[d] = s[d + 1] * grid_sizes[d + 1];
    return s;
}

// -----------------------------------------------------------------------------
// SparseMatrix
// -----------------------------------------------------------------------------

Tensor SparseMatrix::matvec(const Tensor& v) const {
    assert(v.rows() == static_cast<std::size_t>(cols));
    assert(v.cols() == 1);
    Tensor out(rows, 1);
    for (int i = 0; i < rows; ++i) {
        float s = 0.0f;
        const int s0 = row_ptr[i], s1 = row_ptr[i + 1];
        for (int k = s0; k < s1; ++k) s += values[k] * v(col_idx[k], 0);
        out(i, 0) = s;
    }
    return out;
}

Tensor SparseMatrix::matvec_transpose(const Tensor& v) const {
    assert(v.rows() == static_cast<std::size_t>(rows));
    assert(v.cols() == 1);
    Tensor out = Tensor::zeros(cols, 1);
    for (int i = 0; i < rows; ++i) {
        const float vi = v(i, 0);
        const int s0 = row_ptr[i], s1 = row_ptr[i + 1];
        for (int k = s0; k < s1; ++k) out(col_idx[k], 0) += values[k] * vi;
    }
    return out;
}

// -----------------------------------------------------------------------------
// build_interpolation_matrix
// -----------------------------------------------------------------------------

SparseMatrix build_interpolation_matrix(const Tensor& X, const SKIGrid& grid) {
    const int N = static_cast<int>(X.rows());
    const int D = grid.D;
    assert(static_cast<int>(X.cols()) == D);

    SparseMatrix W;
    W.rows = N;
    W.cols = grid.total_points;
    W.row_ptr.assign(N + 1, 0);
    // Upper bound on nnz per row is 4^D; we'll prune zeros below.
    int per_row_cap = 1;
    for (int d = 0; d < D; ++d) per_row_cap *= 4;
    W.col_idx.reserve(static_cast<std::size_t>(N) * per_row_cap);
    W.values.reserve(static_cast<std::size_t>(N) * per_row_cap);

    const std::vector<int> strides = grid.strides();

    // Per-axis scratch for the 4 weights and 4 clamped grid indices.
    std::vector<float> w_axis(static_cast<std::size_t>(4 * D));
    std::vector<int>   i_axis(static_cast<std::size_t>(4 * D));

    for (int n = 0; n < N; ++n) {
        // 1D cubic weights per axis.
        for (int d = 0; d < D; ++d) {
            const float x = X(n, d);
            const float u = (x - grid.grid_mins[d]) / grid.grid_steps[d];
            const int j = static_cast<int>(std::floor(u));
            const float t = u - static_cast<float>(j);
            float ws[4];
            cubic_weights(t, ws);
            for (int k = 0; k < 4; ++k) {
                int idx = j - 1 + k;
                if (idx < 0) idx = 0;
                if (idx >= grid.grid_sizes[d]) idx = grid.grid_sizes[d] - 1;
                w_axis[4 * d + k] = ws[k];
                i_axis[4 * d + k] = idx;
            }
        }
        // Tensor-product of the per-axis weights → up to 4^D entries.
        // Loop over the 4^D combinations using base-4 mixed radix.
        for (int comb = 0; comb < per_row_cap; ++comb) {
            int rest = comb;
            int flat = 0;
            float weight = 1.0f;
            for (int d = 0; d < D; ++d) {
                const int k = rest & 3; rest >>= 2;
                weight *= w_axis[4 * d + k];
                flat += i_axis[4 * d + k] * strides[d];
            }
            if (weight != 0.0f) {
                W.col_idx.push_back(flat);
                W.values.push_back(weight);
            }
        }
        W.row_ptr[n + 1] = static_cast<int>(W.values.size());
    }
    return W;
}

// -----------------------------------------------------------------------------
// Toeplitz column computation
// -----------------------------------------------------------------------------

Tensor toeplitz_column_1d(const Kernel& kernel, int M, float step) {
    // Build (M, 1) and (1, 1) inputs and evaluate k(x_i, x_j) for x_j = j*step, x_i = 0.
    Tensor X1(1, 1);
    X1(0, 0) = 0.0f;
    Tensor X2(M, 1);
    for (int j = 0; j < M; ++j) X2(j, 0) = step * static_cast<float>(j);
    Tensor K = kernel.compute(X1, X2, Backend::CPU);  // (1, M)
    Tensor col(M, 1);
    for (int j = 0; j < M; ++j) col(j, 0) = K(0, j);
    return col;
}

std::vector<Tensor> toeplitz_columns(const Kernel& kernel, const SKIGrid& grid) {
    std::vector<Tensor> cols;
    cols.reserve(grid.D);
    for (int d = 0; d < grid.D; ++d) {
        cols.push_back(toeplitz_column_1d(kernel, grid.grid_sizes[d], grid.grid_steps[d]));
    }
    return cols;
}

// -----------------------------------------------------------------------------
// CPU Toeplitz / Kronecker-Toeplitz matvec
// -----------------------------------------------------------------------------

Tensor toeplitz_matvec_cpu(const Tensor& toeplitz_col, const Tensor& v) {
    const int M = static_cast<int>(toeplitz_col.rows());
    assert(static_cast<int>(v.rows()) == M);
    Tensor out = Tensor::zeros(M, 1);
    for (int i = 0; i < M; ++i) {
        float s = 0.0f;
        for (int j = 0; j < M; ++j) {
            const int lag = std::abs(i - j);
            s += toeplitz_col(lag, 0) * v(j, 0);
        }
        out(i, 0) = s;
    }
    return out;
}

namespace {

/// Apply a 1D Toeplitz matrix of size M_d to every "fiber" of `v` along axis d.
/// `v` is logically shaped (grid_sizes[0], ..., grid_sizes[D-1]); we operate on the
/// flat buffer in-place via temporary buffers.
void apply_toeplitz_along_axis(const Tensor& col, std::vector<float>& buf,
                               const std::vector<int>& shape, int axis) {
    const int D = static_cast<int>(shape.size());
    const int M_d = shape[axis];
    // Strides in the original shape; same axis-major (last varies fastest) convention.
    std::vector<int> strides(D, 1);
    for (int d = D - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

    const int stride_axis = strides[axis];
    const int total = static_cast<int>(buf.size());
    const int num_fibers = total / M_d;

    std::vector<float> fiber(M_d), out(M_d);
    // Iterate over all base offsets (everything except the `axis` coordinate).
    for (int f = 0; f < num_fibers; ++f) {
        // Decode f into the D-1 non-axis coordinates and compute the base offset.
        int rem = f;
        int base = 0;
        for (int d = D - 1; d >= 0; --d) {
            if (d == axis) continue;
            const int idx_d = rem % shape[d];
            rem /= shape[d];
            base += idx_d * strides[d];
        }
        for (int j = 0; j < M_d; ++j) fiber[j] = buf[base + j * stride_axis];
        // Dense 1D Toeplitz matvec along the fiber.
        for (int i = 0; i < M_d; ++i) {
            float s = 0.0f;
            for (int j = 0; j < M_d; ++j) {
                const int lag = std::abs(i - j);
                s += col(lag, 0) * fiber[j];
            }
            out[i] = s;
        }
        for (int j = 0; j < M_d; ++j) buf[base + j * stride_axis] = out[j];
    }
}

}  // namespace

Tensor kron_toeplitz_matvec_cpu(const std::vector<Tensor>& cols,
                                const std::vector<int>& grid_sizes,
                                const Tensor& v) {
    const int D = static_cast<int>(grid_sizes.size());
    int total = 1;
    for (int d = 0; d < D; ++d) total *= grid_sizes[d];
    assert(static_cast<int>(v.rows()) == total);
    std::vector<float> buf(total);
    for (int i = 0; i < total; ++i) buf[i] = v(i, 0);
    for (int d = 0; d < D; ++d) {
        apply_toeplitz_along_axis(cols[d], buf, grid_sizes, d);
    }
    Tensor out(total, 1);
    for (int i = 0; i < total; ++i) out(i, 0) = buf[i];
    return out;
}

// -----------------------------------------------------------------------------
// SKIData
// -----------------------------------------------------------------------------

struct SKIData::Impl {
    SKIGrid grid;
    SparseMatrix W;
    std::vector<Tensor> toeplitz_cols;  // one per axis (1D), or one (D=1)
    float noise_var = 0.0f;
    Backend backend = Backend::CPU;

#ifdef LIGHTGP_HAS_CUDA
    // CUDA-resident state. Lazily initialised on first matvec when backend == CUDA.
    // Uses the custom deleter defined in ski_cuda.cu so the unique_ptr can hold the
    // incomplete SkiCudaState type here.
    mutable SkiCudaStatePtr cuda_state;
#endif
#ifdef LIGHTGP_HAS_ACCELERATE
    // Per-axis FFT plans (vDSP DFT setups + precomputed kernel FFTs). Lazily built
    // on the first CPU matvec so the construction cost is paid only if used.
    mutable std::vector<std::unique_ptr<ToeplitzFFTCpu>> fft_plans;
#endif
};

SKIData::SKIData(SKIGrid grid, SparseMatrix W, std::vector<Tensor> toeplitz_cols,
                 float noise_variance, Backend backend)
    : impl_(std::make_unique<Impl>()) {
    impl_->grid = std::move(grid);
    impl_->W = std::move(W);
    impl_->toeplitz_cols = std::move(toeplitz_cols);
    impl_->noise_var = noise_variance;
    impl_->backend = backend;
}

SKIData::SKIData(SKIData&&) noexcept = default;
SKIData& SKIData::operator=(SKIData&&) noexcept = default;
SKIData::~SKIData() = default;

const SKIGrid& SKIData::grid() const { return impl_->grid; }
int SKIData::N() const { return impl_->W.rows; }
int SKIData::M() const { return impl_->W.cols; }
float SKIData::noise_variance() const { return impl_->noise_var; }
Backend SKIData::backend() const { return impl_->backend; }

Tensor SKIData::matvec(const Tensor& v) const {
    assert(static_cast<int>(v.rows()) == impl_->W.rows);
    assert(v.cols() == 1);
#ifdef LIGHTGP_HAS_CUDA
    if (impl_->backend == Backend::CUDA) {
        if (!impl_->cuda_state) {
            impl_->cuda_state = make_ski_cuda_state(impl_->grid, impl_->W,
                                                    impl_->toeplitz_cols);
        }
        if (impl_->cuda_state) {
            return ski_matvec_cuda(*impl_->cuda_state, v, impl_->noise_var);
        }
    }
#endif

    // CPU path: u = W^T v ; w = K_grid u ; y = W w + sn2 v.
    Tensor u = impl_->W.matvec_transpose(v);

#ifdef LIGHTGP_HAS_ACCELERATE
    // vDSP FFT path — O(M log M) per axis Toeplitz matvec instead of O(M^2).
    if (impl_->fft_plans.empty()) {
        impl_->fft_plans.reserve(impl_->toeplitz_cols.size());
        for (const Tensor& col : impl_->toeplitz_cols) {
            impl_->fft_plans.emplace_back(std::make_unique<ToeplitzFFTCpu>(col));
        }
    }
    Tensor w = kron_toeplitz_matvec_accelerate(impl_->fft_plans,
                                               impl_->grid.grid_sizes, u);
#else
    Tensor w = kron_toeplitz_matvec_cpu(impl_->toeplitz_cols, impl_->grid.grid_sizes, u);
#endif

    Tensor y = impl_->W.matvec(w);
    for (int i = 0; i < static_cast<int>(y.rows()); ++i) {
        y(i, 0) += impl_->noise_var * v(i, 0);
    }
    return y;
}

MatvecFn SKIData::matvec_fn() const {
    return [this](const Tensor& v) { return this->matvec(v); };
}

// -----------------------------------------------------------------------------
// build_ski
// -----------------------------------------------------------------------------

SKIData build_ski(const Tensor& X, const Kernel& kernel,
                  float noise_variance, int points_per_dim, Backend backend) {
    SKIGrid grid = SKIGrid::from_data(X, points_per_dim);
    SparseMatrix W = build_interpolation_matrix(X, grid);
    std::vector<Tensor> cols = toeplitz_columns(kernel, grid);
    return SKIData(std::move(grid), std::move(W), std::move(cols),
                   noise_variance, backend);
}

}  // namespace lightgp
