// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "kernel_base.h"

namespace lightgp {

/// RBF (squared exponential) kernel: k(x, x') = σ² exp(-‖x − x'‖² / (2 l²)).
/// Two scalar log-hyperparameters: log(length_scale), log(signal_variance).
class RBFKernel : public Kernel {
public:
    explicit RBFKernel(float length_scale = 1.0f, float signal_variance = 1.0f);

    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return 2; }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override { return "RBF"; }

    float length_scale() const { return length_scale_; }
    float signal_variance() const { return signal_variance_; }

private:
    float length_scale_;
    float signal_variance_;
};

}  // namespace lightgp
