// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_METAL

/// Matrix-free RBF kernel-vector product on Metal: w = (K + sn2 * I) * v
/// where K is the N x N RBF kernel of X with itself (never materialized).
/// X is N x D, v is N x 1. Returns w as an N x 1 Tensor.
/// Falls back to forming K via rbf_kernel_metal then matmul if Metal is unavailable.
Tensor rbf_matvec_metal(const Tensor& X, const Tensor& v,
                        float length_scale, float signal_variance, float noise_variance);

#endif  // LIGHTGP_HAS_METAL

}  // namespace lightgp
