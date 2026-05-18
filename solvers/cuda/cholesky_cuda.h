// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// Cholesky factorization K = L L^T on CUDA via cuSOLVER `cusolverDnSpotrf`.
/// On entry K is the SPD matrix (row-major); on exit L holds the lower factor
/// (strictly-upper triangle zeroed). Returns false on non-SPD input.
/// Falls back to CPU if the CUDA context is unavailable.
bool cholesky_cuda(const Tensor& K, Tensor& L);

/// Retry `cholesky_cuda` with increasing jitter on the diagonal until SPD.
bool cholesky_cuda_with_jitter(const Tensor& K, Tensor& L, float& jitter_used);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
