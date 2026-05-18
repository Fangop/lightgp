// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#include <cmath>

#include "../core/backend.h"
#include "../core/kernel.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../kernels/cpu/matern_cpu.h"

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;

Tensor make_grid(int n, float lo, float hi) {
    Tensor X(n, 1);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(n - 1);
    }
    return X;
}
}  // namespace

void run_matern_tests() {
    std::printf("[matern] starting...\n");

    // Matern values at r=0: should all equal sf2.
    {
        Tensor X(1, 1, {0.0f});
        for (KernelType t : {KernelType::Matern12, KernelType::Matern32, KernelType::Matern52}) {
            Tensor K = matern_kernel_cpu(X, X, /*l=*/1.0f, /*sf2=*/2.5f, t);
            LIGHTGP_CHECK_NEAR(K(0, 0), 2.5f, 1e-6f);
        }
    }

    // Closed-form at r=l with sf2=1:
    //   Matern12: exp(-1) ≈ 0.3679
    //   Matern32: (1 + sqrt(3)) exp(-sqrt(3)) ≈ 0.4833
    //   Matern52: (1 + sqrt(5) + 5/3) exp(-sqrt(5)) ≈ 0.5223
    {
        Tensor X(2, 1, {0.0f, 1.0f});
        Tensor K12 = matern_kernel_cpu(X, X, 1.0f, 1.0f, KernelType::Matern12);
        LIGHTGP_CHECK_NEAR(K12(0, 1), std::exp(-1.0f), 1e-6f);
        Tensor K32 = matern_kernel_cpu(X, X, 1.0f, 1.0f, KernelType::Matern32);
        LIGHTGP_CHECK_NEAR(K32(0, 1),
                         (1.0f + std::sqrt(3.0f)) * std::exp(-std::sqrt(3.0f)), 1e-6f);
        Tensor K52 = matern_kernel_cpu(X, X, 1.0f, 1.0f, KernelType::Matern52);
        LIGHTGP_CHECK_NEAR(K52(0, 1),
                         (1.0f + std::sqrt(5.0f) + 5.0f / 3.0f) * std::exp(-std::sqrt(5.0f)),
                         1e-6f);
    }

    // Symmetry: K(X, X) is symmetric.
    {
        Tensor X = Tensor::randn(8, 3, 11);
        for (KernelType t : {KernelType::Matern12, KernelType::Matern32, KernelType::Matern52}) {
            Tensor K = matern_kernel_cpu(X, X, 0.5f, 1.5f, t);
            for (std::size_t i = 0; i < 8; ++i) {
                for (std::size_t j = i + 1; j < 8; ++j) {
                    LIGHTGP_CHECK_NEAR(K(i, j), K(j, i), 1e-6f);
                }
            }
        }
    }

    // Diagonal of K(X, X) is always sf2.
    {
        Tensor X = Tensor::randn(20, 4, 22);
        Tensor K = matern_kernel_cpu(X, X, 2.0f, 3.5f, KernelType::Matern32);
        for (std::size_t i = 0; i < 20; ++i) {
            LIGHTGP_CHECK_NEAR(K(i, i), 3.5f, 1e-5f);
        }
    }

    // GPExact end-to-end: each Matern variant should fit sin(x) reasonably.
    {
        const int N = 30;
        Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
        Tensor y(N, 1);
        for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));
        Tensor X_test = make_grid(10, 0.5f, 2.0f * kPi - 0.5f);

        for (KernelType t : {KernelType::Matern12, KernelType::Matern32, KernelType::Matern52}) {
            GPHyperparams hp;
            hp.kernel = t;
            hp.length_scale = 1.0f;
            hp.signal_variance = 1.0f;
            hp.noise_variance = 1e-3f;
            GPExact gp(hp);
            LIGHTGP_CHECK(gp.fit(X, y));
            Tensor m, v;
            LIGHTGP_CHECK(gp.predict(X_test, m, v));
            for (std::size_t i = 0; i < X_test.rows(); ++i) {
                const float truth = std::sin(X_test(i, 0));
                // Matern-1/2 is rough; allow larger error than Matern-5/2.
                const float tol = (t == KernelType::Matern12) ? 0.3f : 0.15f;
                LIGHTGP_CHECK_NEAR(m(i, 0), truth, tol);
                LIGHTGP_CHECK(v(i, 0) >= 0.0f);
            }
        }
    }
}

}  // namespace lightgp
