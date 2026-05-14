#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_METAL

/// Right-looking blocked Cholesky: K = L L^T.
/// Diagonal-block factorization and per-step TRSM run on CPU; the O(N^2*b) trailing
/// rank-block updates are executed by the Metal GEMM kernel. Returns false if any
/// diagonal block is not positive definite. Falls back to cholesky_cpu when N <= block_size
/// or Metal is unavailable.
bool cholesky_metal(const Tensor& K, Tensor& L, int block_size = 128);

/// Retry cholesky_metal with increasing jitter on the diagonal until SPD.
bool cholesky_metal_with_jitter(const Tensor& K, Tensor& L, float& jitter_used,
                                int block_size = 128);

#endif  // LIGHTGP_HAS_METAL

}  // namespace lightgp
