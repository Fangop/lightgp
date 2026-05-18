// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "kernel_base.h"

namespace lightgp {

Tensor Kernel::compute_diag(const Tensor& X) const {
    // Default: build the full N x N kernel and pull the diagonal. Subclasses
    // (RBF, Matern) override with the trivial closed form.
    Tensor K = compute(X, X, Backend::CPU);
    const std::size_t N = X.rows();
    Tensor d(N, 1);
    for (std::size_t i = 0; i < N; ++i) d(i, 0) = K(i, i);
    return d;
}

}  // namespace lightgp
