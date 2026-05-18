// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "kernel_base.h"

namespace lightgp {

/// Linear kernel: k(x, x') = σ² (x − c)ᵀ(x' − c).
/// Trivially expressible as a (scaled, shifted) GEMM; the implementation uses
/// Tensor::matmul which routes through Accelerate on CPU.
/// Hyperparameter: log(signal_variance). Offset `c` is fixed at construction
/// (not currently optimized — set per-application via the constructor).
class LinearKernel : public Kernel {
public:
    /// `input_dim` matters only when offset is non-default; otherwise the
    /// kernel does not need it (it's just a dot product).
    explicit LinearKernel(float signal_variance = 1.0f,
                          std::size_t input_dim = 1,
                          float offset = 0.0f);

    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return 1; }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override { return "Linear"; }

    float signal_variance() const { return signal_variance_; }
    float offset() const { return offset_; }

private:
    float signal_variance_;
    std::size_t input_dim_;
    float offset_;  // scalar offset broadcast across all dims
};

}  // namespace lightgp
