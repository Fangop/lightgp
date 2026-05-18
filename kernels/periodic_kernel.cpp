// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "periodic_kernel.h"

#include <cassert>
#include <cmath>

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

PeriodicKernel::PeriodicKernel(float length_scale, float period, float signal_variance)
    : length_scale_(length_scale), period_(period), signal_variance_(signal_variance) {}

Tensor PeriodicKernel::compute(const Tensor& X1, const Tensor& X2, Backend /*backend*/) const {
    // CPU implementation only — Metal port follows the same tiling as RBF small_d,
    // swapping the scalar function. Backend argument accepted for API symmetry.
    assert(X1.cols() == X2.cols());
    const std::size_t n = X1.rows();
    const std::size_t m = X2.rows();
    const std::size_t d = X1.cols();
    const float inv_l2 = 1.0f / (length_scale_ * length_scale_);
    const float omega = kPi / period_;

    Tensor K(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            // sum over dims of sin²(π * (x_i - x_j) / period) / length_scale²
            float acc = 0.0f;
            for (std::size_t k = 0; k < d; ++k) {
                const float diff = X1(i, k) - X2(j, k);
                const float s = std::sin(omega * diff);
                acc += s * s;
            }
            K(i, j) = signal_variance_ * std::exp(-2.0f * acc * inv_l2);
        }
    }
    return K;
}

Tensor PeriodicKernel::compute_diag(const Tensor& X) const {
    Tensor d(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) d(i, 0) = signal_variance_;
    return d;
}

std::vector<float> PeriodicKernel::get_log_params() const {
    return {std::log(length_scale_), std::log(period_), std::log(signal_variance_)};
}

void PeriodicKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == 3);
    length_scale_ = std::exp(p[0]);
    period_ = std::exp(p[1]);
    signal_variance_ = std::exp(p[2]);
}

}  // namespace lightgp
