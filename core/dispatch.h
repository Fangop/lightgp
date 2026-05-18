// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "backend.h"
#include "kernel.h"
#include "tensor.h"

namespace lightgp {

/// Compute the RBF kernel matrix on the requested backend, falling back to CPU
/// if the backend isn't compiled in or available. Emits a one-time warning per backend on fallback.
Tensor dispatch_rbf_kernel(const Tensor& X1, const Tensor& X2,
                           float length_scale, float signal_variance,
                           Backend backend);

/// Compute a kernel matrix of the given KernelType. RBF routes through dispatch_rbf_kernel
/// (Metal-accelerated); Matern variants currently run on CPU regardless of backend
/// (Metal Matern shaders are a follow-up — see report.md next moves).
Tensor dispatch_kernel(const Tensor& X1, const Tensor& X2,
                       float length_scale, float signal_variance,
                       KernelType type, Backend backend);

/// Cholesky factorization with jitter retry, routed by backend.
/// On Metal, the blocked solver falls back to CPU when N <= block_size internally.
bool dispatch_cholesky_with_jitter(const Tensor& K, Tensor& L, float& jitter_used,
                                   Backend backend, int block_size = 128);

/// Solve K x = b given the lower Cholesky factor L of K (K = L L^T), routed by backend.
Tensor dispatch_cholesky_solve(const Tensor& L, const Tensor& b, Backend backend);

/// Forward substitution only: solve L y = b for lower triangular L, routed by backend.
Tensor dispatch_forward_solve(const Tensor& L, const Tensor& b, Backend backend);

}  // namespace lightgp
