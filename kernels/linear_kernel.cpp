// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "linear_kernel.h"

#include <cassert>
#include <cmath>

namespace lightgp {

LinearKernel::LinearKernel(float signal_variance, std::size_t input_dim, float offset)
    : signal_variance_(signal_variance), input_dim_(input_dim), offset_(offset) {}

Tensor LinearKernel::compute(const Tensor& X1, const Tensor& X2, Backend /*backend*/) const {
    assert(X1.cols() == X2.cols());
    // K = sf2 * (X1 - c) (X2 - c)^T. If c = 0, just sf2 * X1 X2^T (GEMM).
    if (offset_ == 0.0f) {
        Tensor cross = X1.matmul(X2.transpose());
        // Scale in-place.
        for (std::size_t i = 0; i < cross.size(); ++i) cross.data()[i] *= signal_variance_;
        return cross;
    }
    // Shift, then GEMM.
    Tensor X1s(X1.rows(), X1.cols());
    Tensor X2s(X2.rows(), X2.cols());
    for (std::size_t i = 0; i < X1.rows(); ++i)
        for (std::size_t d = 0; d < X1.cols(); ++d)
            X1s(i, d) = X1(i, d) - offset_;
    for (std::size_t i = 0; i < X2.rows(); ++i)
        for (std::size_t d = 0; d < X2.cols(); ++d)
            X2s(i, d) = X2(i, d) - offset_;
    Tensor cross = X1s.matmul(X2s.transpose());
    for (std::size_t i = 0; i < cross.size(); ++i) cross.data()[i] *= signal_variance_;
    return cross;
}

Tensor LinearKernel::compute_diag(const Tensor& X) const {
    Tensor d(X.rows(), 1);
    const std::size_t D = X.cols();
    for (std::size_t i = 0; i < X.rows(); ++i) {
        float s = 0.0f;
        for (std::size_t k = 0; k < D; ++k) {
            const float xk = X(i, k) - offset_;
            s += xk * xk;
        }
        d(i, 0) = signal_variance_ * s;
    }
    return d;
}

std::vector<float> LinearKernel::get_log_params() const {
    return {std::log(signal_variance_)};
}

void LinearKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == 1);
    signal_variance_ = std::exp(p[0]);
}

}  // namespace lightgp
