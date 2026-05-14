#pragma once

#include "../../core/tensor.h"

namespace lightgp {

/// Lower Cholesky factor: K = L L^T. Returns false if K is not positive definite.
bool cholesky_cpu(const Tensor& K, Tensor& L);

/// log|K| = 2 * sum(log(L_ii)) given the lower Cholesky factor L.
float log_det_from_cholesky(const Tensor& L);

/// Solve K x = b given the lower Cholesky factor L of K. b may be N x 1 or N x k.
Tensor cholesky_solve(const Tensor& L, const Tensor& b);

/// Try Cholesky with increasing jitter on the diagonal until SPD; jitter_used is the value applied.
bool cholesky_with_jitter(const Tensor& K, Tensor& L, float& jitter_used);

}  // namespace lightgp
