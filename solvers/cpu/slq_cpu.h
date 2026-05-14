#pragma once

#include <cstdint>

#include "../../core/matvec.h"
#include "../../core/tensor.h"

namespace lightgp {

/// Stochastic Lanczos quadrature estimator for log|A| where A is symmetric positive definite.
/// Averages over n_probes Rademacher probes; each probe runs n_iters of Lanczos with full
/// reorthogonalization, eigendecomposes the small tridiagonal, then approximates the
/// quadratic form v^T log(A) v.
float slq_log_det_cpu(const MatvecFn& matvec,
                      std::size_t n,
                      int n_probes = 20,
                      int n_iters = 30,
                      std::uint64_t seed = 0);

/// Convenience overload for an explicitly materialized A.
float slq_log_det_cpu(const Tensor& A,
                      int n_probes = 20,
                      int n_iters = 30,
                      std::uint64_t seed = 0);

}  // namespace lightgp
