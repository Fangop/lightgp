// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/kernel.h"
#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// Matérn kernel matrix on CUDA. `type` must be one of `Matern12 / Matern32 / Matern52`.
/// Uses the cuBLAS distance trick + a custom variant-dispatched elementwise kernel.
/// Falls back to `matern_kernel_cpu` if the CUDA context is unavailable.
Tensor matern_kernel_cuda(const Tensor& X1, const Tensor& X2,
                          float length_scale, float signal_variance,
                          KernelType type);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
