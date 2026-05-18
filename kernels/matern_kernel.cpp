// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "matern_kernel.h"

#include <cassert>
#include <cmath>

#include "../core/dispatch.h"

namespace lightgp {

namespace {
KernelType resolve_type(float nu) {
    if (nu < 1.0f) return KernelType::Matern12;
    if (nu < 2.0f) return KernelType::Matern32;
    return KernelType::Matern52;
}
}  // namespace

MaternKernel::MaternKernel(float nu, float length_scale, float signal_variance)
    : nu_(nu), type_(resolve_type(nu)),
      length_scale_(length_scale), signal_variance_(signal_variance) {}

Tensor MaternKernel::compute(const Tensor& X1, const Tensor& X2, Backend backend) const {
    return dispatch_kernel(X1, X2, length_scale_, signal_variance_, type_, backend);
}

Tensor MaternKernel::compute_diag(const Tensor& X) const {
    // k(x, x) = σ² for all Matern variants (r = 0).
    Tensor d(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) d(i, 0) = signal_variance_;
    return d;
}

std::vector<float> MaternKernel::get_log_params() const {
    return {std::log(length_scale_), std::log(signal_variance_)};
}

void MaternKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == 2);
    length_scale_ = std::exp(p[0]);
    signal_variance_ = std::exp(p[1]);
}

std::string MaternKernel::name() const {
    if (type_ == KernelType::Matern12) return "Matern12";
    if (type_ == KernelType::Matern32) return "Matern32";
    return "Matern52";
}

}  // namespace lightgp
