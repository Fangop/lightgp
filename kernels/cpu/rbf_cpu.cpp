// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "rbf_cpu.h"

#include <cassert>
#include <cmath>
#include <vector>

#ifdef LIGHTGP_HAS_ACCELERATE
#include "../../core/blas_accel.h"
#endif

namespace lightgp {

static inline float squared_distance(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t k = 0; k < d; ++k) {
        const float diff = a[k] - b[k];
        s += diff * diff;
    }
    return s;
}

Tensor rbf_kernel_cpu(const Tensor& X1, const Tensor& X2,
                      float length_scale, float signal_variance) {
    assert(X1.cols() == X2.cols());
    const std::size_t n = X1.rows();
    const std::size_t m = X2.rows();
    const std::size_t d = X1.cols();
    const float inv_2l2 = 0.5f / (length_scale * length_scale);

#ifdef LIGHTGP_HAS_ACCELERATE
    // BLAS distance trick: r²(i,j) = ||x_i||² + ||x_j||² - 2 x_i·x_j.
    // The cross term X1 @ X2^T goes through Accelerate sgemm (AMX-accelerated on M-series).
    std::vector<float> sq1(n), sq2(m);
    for (std::size_t i = 0; i < n; ++i) {
        const float* xi = X1.data() + i * d;
        float s = 0.0f;
        for (std::size_t k = 0; k < d; ++k) s += xi[k] * xi[k];
        sq1[i] = s;
    }
    for (std::size_t j = 0; j < m; ++j) {
        const float* xj = X2.data() + j * d;
        float s = 0.0f;
        for (std::size_t k = 0; k < d; ++k) s += xj[k] * xj[k];
        sq2[j] = s;
    }
    // cross = X1 @ X2^T  (N x M). Tensor::matmul already goes through Accelerate sgemm.
    Tensor cross = X1.matmul(X2.transpose());

    Tensor K(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const float r2 = sq1[i] + sq2[j] - 2.0f * cross(i, j);
            const float r2_clamped = (r2 > 0.0f) ? r2 : 0.0f;  // numerical floor
            K(i, j) = signal_variance * std::exp(-inv_2l2 * r2_clamped);
        }
    }
    return K;
#else
    Tensor K(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        const float* xi = X1.data() + i * d;
        for (std::size_t j = 0; j < m; ++j) {
            const float* xj = X2.data() + j * d;
            const float r2 = squared_distance(xi, xj, d);
            K(i, j) = signal_variance * std::exp(-inv_2l2 * r2);
        }
    }
    return K;
#endif
}

void rbf_kernel_gradients_cpu(const Tensor& X1, const Tensor& X2,
                              float length_scale, float signal_variance,
                              Tensor& grad_log_l, Tensor& grad_log_sf2) {
    assert(X1.cols() == X2.cols());
    const std::size_t n = X1.rows();
    const std::size_t m = X2.rows();
    const std::size_t d = X1.cols();
    const float l2 = length_scale * length_scale;
    const float inv_2l2 = 0.5f / l2;
    grad_log_l = Tensor::zeros(n, m);
    grad_log_sf2 = Tensor::zeros(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        const float* xi = X1.data() + i * d;
        for (std::size_t j = 0; j < m; ++j) {
            const float* xj = X2.data() + j * d;
            const float r2 = squared_distance(xi, xj, d);
            const float k = signal_variance * std::exp(-inv_2l2 * r2);
            // dK/d(log sf2) = K (since K ∝ sf2 = exp(log_sf2)).
            grad_log_sf2(i, j) = k;
            // dK/d(log l) = K * r^2 / l^2.
            grad_log_l(i, j) = k * (r2 / l2);
        }
    }
}

}  // namespace lightgp
