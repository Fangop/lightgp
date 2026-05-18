// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// RBF kernel matrix computed on CUDA. Uses the cuBLAS distance trick
///     r²(i, j) = ||x_i||² + ||x_j||² - 2 x_i · x_j
/// where the cross term goes through `cublasSgemm` and the elementwise reduction +
/// exp is a small custom CUDA kernel. Matches `rbf_kernel_cpu` to atol ~1e-5.
/// Falls back to CPU if the CUDA context failed to initialize.
Tensor rbf_kernel_cuda(const Tensor& X1, const Tensor& X2,
                       float length_scale, float signal_variance);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
