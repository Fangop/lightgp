// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "kernel_base.h"
#include "../core/kernel.h"  // KernelType

namespace lightgp {

/// Matérn kernel with nu ∈ {0.5, 1.5, 2.5}. Two log-hyperparameters
/// (length_scale, signal_variance); the smoothness `nu` is fixed at construction.
class MaternKernel : public Kernel {
public:
    explicit MaternKernel(float nu = 2.5f,
                          float length_scale = 1.0f,
                          float signal_variance = 1.0f);

    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return 2; }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override;

    float nu() const { return nu_; }

private:
    float nu_;
    KernelType type_;  // Matern12 / Matern32 / Matern52
    float length_scale_;
    float signal_variance_;
};

}  // namespace lightgp
