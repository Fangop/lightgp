// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_METAL

/// Solve K x = b given the lower Cholesky factor L of K, on Metal.
/// Runs forward substitution (L y = b) then backward substitution (L^T x = y).
/// b may be N x 1 or N x M; one threadgroup per RHS column.
/// Falls back to cholesky_solve (CPU) if Metal is unavailable.
Tensor cholesky_solve_metal(const Tensor& L, const Tensor& b);

/// Forward substitution only: solve L y = b where L is lower triangular.
/// Used by predict() when only L^{-1} K_star is needed (variance reduction).
Tensor forward_solve_metal(const Tensor& L, const Tensor& b);

#endif  // LIGHTGP_HAS_METAL

}  // namespace lightgp
