// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../core/backend.h"
#include "../core/matvec.h"
#include "../core/tensor.h"
#include "../kernels/kernel_base.h"

namespace lightgp {

/// Regular tensor-product grid spanning the data bounding box (padded).
/// For SKI/KISS-GP: inducing points placed on this grid, kernel matrix
/// between grid points is (multi-D) Toeplitz, so Kv is O(M log M) via FFT.
struct SKIGrid {
    int D = 0;                          ///< input dimension
    std::vector<int> grid_sizes;        ///< M_d for d = 0..D-1
    std::vector<float> grid_mins;       ///< min coordinate per dim (after padding)
    std::vector<float> grid_steps;      ///< (max - min) / (M_d - 1)
    int total_points = 0;               ///< prod_d grid_sizes[d]

    /// Build a grid that covers [min - padding*range, max + padding*range] in each
    /// data dimension with M_d points per axis. Default heuristic: M_d = max(50, ceil(N^(1/D))).
    static SKIGrid from_data(const Tensor& X, int points_per_dim = 0,
                             float padding_fraction = 0.1f);

    /// Decode a flat index into the D-dimensional coordinate of that grid point.
    std::vector<float> grid_point(int flat_index) const;

    /// Sum of strides for axis-major flat indexing: stride[d] = prod_{k>d} grid_sizes[k].
    std::vector<int> strides() const;
};

/// CSR sparse matrix specialized for the cubic-interpolation operator W: N data points
/// embedded into an M-point regular grid via ~4^D weights each (16 in 2D, 64 in 3D).
struct SparseMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_ptr;       ///< size rows+1
    std::vector<int> col_idx;
    std::vector<float> values;

    /// y = W * v  (rows x 1)
    Tensor matvec(const Tensor& v) const;
    /// y = W^T * v  (cols x 1)
    Tensor matvec_transpose(const Tensor& v) const;
};

/// Build the N x M cubic-interpolation matrix from data points X (N x D) into `grid`.
/// Each row of W has at most 4^D nonzero entries. Out-of-range indices on the boundary
/// are clamped to the valid range; choose `padding_fraction > 0` in the grid to avoid
/// boundary bias.
SparseMatrix build_interpolation_matrix(const Tensor& X, const SKIGrid& grid);

/// Compute the first row/column of the M-point Toeplitz kernel matrix on a 1D grid:
/// t[j] = kernel(0, j*h), j = 0..M-1. For composite/anisotropic kernels we evaluate
/// the kernel on (0, j*h) which is meaningful when the kernel is stationary in the
/// requested axis. Returns an (M, 1) Tensor.
///
/// For multi-D Kronecker-Toeplitz grids, call this once per axis with the per-axis
/// grid step and the per-axis 1D kernel. The full grid kernel is K_1 ⊗ ... ⊗ K_D.
Tensor toeplitz_column_1d(const Kernel& kernel, int M, float step);

/// One Toeplitz column per axis for a product grid. Each entry of the returned vector
/// is the (M_d, 1) Toeplitz column on axis d.
std::vector<Tensor> toeplitz_columns(const Kernel& kernel, const SKIGrid& grid);

/// Dense reference Toeplitz matvec on CPU: t the (M, 1) first column, v (M, 1),
/// returns (M, 1). O(M^2) — only used for correctness tests and the no-CUDA path.
Tensor toeplitz_matvec_cpu(const Tensor& toeplitz_col, const Tensor& v);

/// Kronecker-Toeplitz matvec on CPU: applies each axis Toeplitz matrix in turn.
/// Cost O(M * (M_1 + M_2 + ... + M_D)) — fine at the sizes where SKI even makes sense
/// in CPU-only mode (small smoke tests).
Tensor kron_toeplitz_matvec_cpu(const std::vector<Tensor>& cols,
                                const std::vector<int>& grid_sizes,
                                const Tensor& v);

/// Opaque SKI state. Holds: grid, W, per-axis Toeplitz columns, precomputed cuFFT
/// plans (when built on CUDA), and the noise variance. The matvec method evaluates
/// (W K_grid W^T + sn2 I) v on the requested backend.
class SKIData {
public:
    SKIData(SKIGrid grid, SparseMatrix W, std::vector<Tensor> toeplitz_cols,
            float noise_variance, Backend backend);
    SKIData(SKIData&&) noexcept;
    SKIData& operator=(SKIData&&) noexcept;
    ~SKIData();
    SKIData(const SKIData&) = delete;
    SKIData& operator=(const SKIData&) = delete;

    /// y = (K_approx + sn2 I) v  where K_approx = W K_grid W^T.
    /// Backend::CUDA uses cuFFT + a CSR matvec kernel; CPU uses the dense reference.
    Tensor matvec(const Tensor& v) const;

    /// Bound matvec callback for CG / SLQ.
    MatvecFn matvec_fn() const;

    const SKIGrid& grid() const;
    int N() const;
    int M() const;
    float noise_variance() const;
    Backend backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Convenience: build SKIGrid + W + Toeplitz columns from training data and kernel.
SKIData build_ski(const Tensor& X, const Kernel& kernel,
                  float noise_variance, int points_per_dim = 0,
                  Backend backend = Backend::CPU);

}  // namespace lightgp
