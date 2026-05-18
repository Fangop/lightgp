// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "kernel_base.h"

namespace lightgp {

/// Periodic kernel (1D): k(x, x') = σ² exp(-2 sin²(π|x − x'| / p) / l²).
/// Three log-hyperparameters: log(length_scale), log(period), log(signal_variance).
/// CPU-only for now; the same tiling pattern as RBF would port to Metal cleanly.
class PeriodicKernel : public Kernel {
public:
    explicit PeriodicKernel(float length_scale = 1.0f,
                            float period = 1.0f,
                            float signal_variance = 1.0f);

    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return 3; }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override { return "Periodic"; }

    float length_scale() const { return length_scale_; }
    float period() const { return period_; }
    float signal_variance() const { return signal_variance_; }

private:
    float length_scale_;
    float period_;
    float signal_variance_;
};

}  // namespace lightgp
