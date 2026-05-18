// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../core/backend.h"
#include "../core/kernel.h"
#include "../core/tensor.h"

namespace lightgp {

/// Hyperparameters for the sparse GP. Mirrors GPHyperparams.
struct GPSparseHyperparams {
    KernelType kernel = KernelType::RBF;
    float length_scale = 1.0f;
    float signal_variance = 1.0f;
    float noise_variance = 1e-2f;
};

/// Sparse Gaussian Process regression using Titsias (2009) FITC/VFE-style approximation.
/// Storage is O(N + M^2), training cost is O(NM^2 + M^3) — scales to N=50k+ with M~200.
/// Inducing points are picked by farthest-point sampling on the training inputs; they
/// are not currently jointly optimized with hyperparameters (see report.md next-moves).
class GPSparse {
public:
    explicit GPSparse(GPSparseHyperparams hp = {}, Backend backend = Backend::CPU);

    /// Fit on (X_train, y_train) with num_inducing inducing points selected from X_train.
    /// Returns false if either inner Cholesky fails (numerical issue).
    bool fit(const Tensor& X_train, const Tensor& y_train,
             std::size_t num_inducing, std::uint64_t seed = 42);

    /// Predict posterior mean and (marginal) variance at X_test. Both M_test x 1.
    bool predict(const Tensor& X_test, Tensor& mean_out, Tensor& var_out) const;

    /// VFE/FITC log marginal likelihood at the current hyperparameters and inducing set.
    float log_marginal_likelihood() const;

    /// Current hyperparameters.
    const GPSparseHyperparams& hyperparams() const { return hp_; }
    /// Inducing locations (M x D), copied from training inputs at fit time.
    const Tensor& inducing_points() const { return Z_; }
    /// True once fit() has succeeded.
    bool fitted() const { return fitted_; }

private:
    Backend effective_backend() const;

    GPSparseHyperparams hp_;
    Backend backend_ = Backend::CPU;
    Tensor X_train_, y_train_;
    Tensor Z_;          // M x D inducing locations
    Tensor K_uf_;       // M x N inducing-training cross kernel (cached for log_ml)
    Tensor L_uu_;       // M x M lower Cholesky of K_uu + jitter
    Tensor L_Sigma_;    // M x M lower Cholesky of Σ = K_uu + K_uf Λ^{-1} K_fu
    Tensor alpha_;      // M x 1 = Σ^{-1} K_uf Λ^{-1} y
    Tensor Lambda_;     // N x 1 effective per-point noise diagonal
    bool fitted_ = false;
};

}  // namespace lightgp
