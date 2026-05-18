// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensor.h"

namespace lightgp {

/// Abstract GP mean function. Returns an N x 1 vector of mean values for an N x D input.
/// Hyperparameters (when any) are real-valued (not log-transformed) since means can be
/// any sign — unlike kernel hyperparameters which must be positive.
class MeanFunction {
public:
    virtual ~MeanFunction() = default;
    virtual Tensor compute(const Tensor& X) const = 0;
    virtual int num_params() const = 0;
    virtual std::vector<float> get_params() const = 0;
    virtual void set_params(const std::vector<float>& params) = 0;
    virtual std::string name() const = 0;
};

/// Zero mean — no parameters. The default.
class ZeroMean : public MeanFunction {
public:
    Tensor compute(const Tensor& X) const override;
    int num_params() const override { return 0; }
    std::vector<float> get_params() const override { return {}; }
    void set_params(const std::vector<float>&) override {}
    std::string name() const override { return "Zero"; }
};

/// Constant mean — one scalar parameter `c`. Returns c * ones(N).
class ConstantMean : public MeanFunction {
public:
    explicit ConstantMean(float c = 0.0f) : c_(c) {}
    Tensor compute(const Tensor& X) const override;
    int num_params() const override { return 1; }
    std::vector<float> get_params() const override { return {c_}; }
    void set_params(const std::vector<float>& p) override;
    std::string name() const override { return "Constant"; }

    float value() const { return c_; }

private:
    float c_;
};

/// Linear mean — w (D-vector) + bias b. Returns X @ w + b.
class LinearMean : public MeanFunction {
public:
    explicit LinearMean(std::size_t input_dim);
    LinearMean(std::vector<float> w, float b);
    Tensor compute(const Tensor& X) const override;
    int num_params() const override { return static_cast<int>(w_.size()) + 1; }
    std::vector<float> get_params() const override;
    void set_params(const std::vector<float>& p) override;
    std::string name() const override { return "Linear"; }

private:
    std::vector<float> w_;
    float b_;
};

}  // namespace lightgp
