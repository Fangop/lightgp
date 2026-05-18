// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../core/backend.h"
#include "../core/tensor.h"

namespace lightgp {

/// Abstract base class for GP kernels. Subclasses implement at minimum
/// `compute`, `compute_diag`, `num_params`, `get_log_params`, `set_log_params`, `name`.
/// Composite kernels (sum, product, scale) wrap other Kernels and forward calls.
class Kernel {
public:
    virtual ~Kernel() = default;

    /// N x M kernel matrix between X1 (N x D) and X2 (M x D).
    virtual Tensor compute(const Tensor& X1, const Tensor& X2,
                           Backend backend = Backend::CPU) const = 0;

    /// Diagonal of the N x N kernel of X with itself: returns an N x 1 vector.
    /// Default implementation pulls the diagonal out of compute(X, X) — subclasses
    /// override for kernels where the diagonal is trivially expressible (e.g. RBF: σ²).
    virtual Tensor compute_diag(const Tensor& X) const;

    /// Number of learnable (log-parameterized) hyperparameters.
    virtual int num_params() const = 0;
    /// Flat get/set of log-hyperparameters.
    virtual std::vector<float> get_log_params() const = 0;
    virtual void set_log_params(const std::vector<float>& params) = 0;

    /// Human-readable name (used in operator+ / scale() composition names too).
    virtual std::string name() const = 0;
};

}  // namespace lightgp
