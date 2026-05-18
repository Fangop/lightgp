// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <memory>
#include <vector>

#include "../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

struct SKIGrid;
struct SparseMatrix;

/// Opaque holder for the CUDA-resident SKI state:
///   - device-side CSR (W) and W^T (column-major reordering for the W matvec)
///   - per-axis Toeplitz columns embedded as length-2M_d real signals
///   - per-axis cuFFT R2C/C2R plans
///   - per-axis FFTed kernel signals (complex, length M_d + 1)
///   - device workspace buffers
///
/// Fully defined in ski_cuda.cu so the header stays CUDA-free.
struct SkiCudaState;

/// Custom deleter so std::unique_ptr<SkiCudaState> can hold an incomplete type
/// — the deleter's operator() body is defined in ski_cuda.cu where the type is complete.
struct SkiCudaStateDeleter {
    void operator()(SkiCudaState* p) const noexcept;
};

using SkiCudaStatePtr = std::unique_ptr<SkiCudaState, SkiCudaStateDeleter>;

/// Build the CUDA-resident state from the host-side grid + W + per-axis Toeplitz columns.
/// Returns null if CUDA is unavailable or any allocation/plan creation fails.
SkiCudaStatePtr make_ski_cuda_state(const SKIGrid& grid,
                                    const SparseMatrix& W,
                                    const std::vector<Tensor>& toeplitz_cols);

/// Evaluate y = (W K_grid W^T + sn2 I) v on CUDA. v and the returned Tensor are host-side.
Tensor ski_matvec_cuda(const SkiCudaState& state, const Tensor& v, float noise_variance);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
