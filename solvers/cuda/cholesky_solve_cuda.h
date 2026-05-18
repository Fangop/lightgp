// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// Solve K x = b given the lower Cholesky factor L of K on CUDA via two cuBLAS strsm calls.
/// b may be N x 1 or N x M; result has the same shape as b. Falls back to CPU on unavailable device.
Tensor cholesky_solve_cuda(const Tensor& L, const Tensor& b);

/// Forward substitution only: solve L y = b for lower-triangular L.
Tensor forward_solve_cuda(const Tensor& L, const Tensor& b);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
