// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "matern_cpu.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace lightgp {

namespace {
constexpr float kSqrt3 = 1.7320508075688772f;
constexpr float kSqrt5 = 2.2360679774997896f;

inline float squared_distance(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t k = 0; k < d; ++k) {
        const float diff = a[k] - b[k];
        s += diff * diff;
    }
    return s;
}
}  // namespace

Tensor matern_kernel_cpu(const Tensor& X1, const Tensor& X2,
                         float length_scale, float signal_variance,
                         KernelType type) {
    assert(X1.cols() == X2.cols());
    assert(type != KernelType::RBF);  // route RBF through rbf_kernel_cpu

    const std::size_t n = X1.rows();
    const std::size_t m = X2.rows();
    const std::size_t d = X1.cols();
    const float inv_l = 1.0f / length_scale;
    const float inv_l2 = inv_l * inv_l;

#ifdef LIGHTGP_HAS_ACCELERATE
    // BLAS distance trick: r²(i,j) = ||x_i||² + ||x_j||² - 2 x_i·x_j (cross via sgemm).
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
    Tensor cross = X1.matmul(X2.transpose());
    Tensor K(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const float raw = sq1[i] + sq2[j] - 2.0f * cross(i, j);
            const float r2 = (raw > 0.0f) ? raw : 0.0f;
            const float r = std::sqrt(r2);
            float k = 0.0f;
            switch (type) {
                case KernelType::Matern12: k = std::exp(-r * inv_l); break;
                case KernelType::Matern32: {
                    const float u = kSqrt3 * r * inv_l;
                    k = (1.0f + u) * std::exp(-u); break;
                }
                case KernelType::Matern52: {
                    const float u = kSqrt5 * r * inv_l;
                    k = (1.0f + u + (5.0f / 3.0f) * r2 * inv_l2) * std::exp(-u); break;
                }
                default: break;
            }
            K(i, j) = signal_variance * k;
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
            const float r = std::sqrt(r2);
            float k = 0.0f;
            switch (type) {
                case KernelType::Matern12: k = std::exp(-r * inv_l); break;
                case KernelType::Matern32: {
                    const float u = kSqrt3 * r * inv_l;
                    k = (1.0f + u) * std::exp(-u); break;
                }
                case KernelType::Matern52: {
                    const float u = kSqrt5 * r * inv_l;
                    k = (1.0f + u + (5.0f / 3.0f) * r2 * inv_l2) * std::exp(-u); break;
                }
                default: break;
            }
            K(i, j) = signal_variance * k;
        }
    }
    return K;
#endif
}

}  // namespace lightgp
