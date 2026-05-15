#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// Matrix-free RBF kernel-vector product on CUDA: w = (K + sn2 * I) * v
/// where K is the N x N RBF kernel of X with itself (never materialized).
/// X is N x D row-major, v is N x 1; returns w as an N x 1 Tensor.
/// Falls back to forming K via `rbf_kernel_cuda` + matmul if the device is unavailable.
Tensor rbf_matvec_cuda(const Tensor& X, const Tensor& v,
                       float length_scale, float signal_variance, float noise_variance);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
