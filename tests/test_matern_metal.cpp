// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#ifdef LIGHTGP_HAS_METAL

#include <cmath>

#include "../core/backend.h"
#include "../core/dispatch.h"
#include "../core/kernel.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../kernels/cpu/matern_cpu.h"
#include "../kernels/metal/matern_metal.h"
#include "../kernels/metal/metal_context.h"

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

void run_matern_metal_tests() {
    std::printf("[matern_metal] starting...\n");

    if (!MetalContext::instance().available()) {
        std::fprintf(stderr, "[matern_metal] SKIP — Metal unavailable: %s\n",
                     MetalContext::instance().error().c_str());
        return;
    }

    auto compare = [](const Tensor& a, const Tensor& b, float atol) {
        LIGHTGP_CHECK(a.rows() == b.rows() && a.cols() == b.cols());
        for (std::size_t i = 0; i < a.size(); ++i) {
            LIGHTGP_CHECK_NEAR(a.data()[i], b.data()[i], atol);
        }
    };

    // All three Matern variants across the three shape paths (small_d, scalar tiled, float4 tiled).
    for (KernelType type : {KernelType::Matern12, KernelType::Matern32, KernelType::Matern52}) {
        // D=1 (small_d).
        {
            Tensor X = Tensor::randn(20, 1, 1);
            Tensor Kc = matern_kernel_cpu(X, X, 1.3f, 0.7f, type);
            Tensor Km = matern_kernel_metal(X, X, 1.3f, 0.7f, type);
            compare(Kc, Km, 1e-5f);
        }
        // D=3 (scalar tiled, non-multiple of 4).
        {
            Tensor X1 = Tensor::randn(17, 3, 2);
            Tensor X2 = Tensor::randn(11, 3, 3);
            Tensor Kc = matern_kernel_cpu(X1, X2, 0.6f, 2.0f, type);
            Tensor Km = matern_kernel_metal(X1, X2, 0.6f, 2.0f, type);
            compare(Kc, Km, 1e-5f);
        }
        // D=4 (small_d — D<=16 path).
        {
            Tensor X = Tensor::randn(33, 4, 4);
            Tensor Kc = matern_kernel_cpu(X, X, 1.0f, 1.0f, type);
            Tensor Km = matern_kernel_metal(X, X, 1.0f, 1.0f, type);
            compare(Kc, Km, 1e-5f);
        }
        // D=64 (float4 tiled path).
        {
            Tensor X1 = Tensor::randn(40, 64, 5);
            Tensor X2 = Tensor::randn(25, 64, 6);
            Tensor Kc = matern_kernel_cpu(X1, X2, 2.5f, 1.5f, type);
            Tensor Km = matern_kernel_metal(X1, X2, 2.5f, 1.5f, type);
            compare(Kc, Km, 1e-4f);
        }
        // D=17 (scalar tiled path, > 16, not multiple of 4).
        {
            Tensor X = Tensor::randn(50, 17, 7);
            Tensor Kc = matern_kernel_cpu(X, X, 1.5f, 0.5f, type);
            Tensor Km = matern_kernel_metal(X, X, 1.5f, 0.5f, type);
            compare(Kc, Km, 1e-4f);
        }
    }

    // GPExact end-to-end with Matern-5/2 on Metal: should match GPExact CPU within 1e-4.
    {
        const int N = 30;
        Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
        Tensor y(N, 1);
        for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));

        GPHyperparams hp;
        hp.kernel = KernelType::Matern52;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-3f;

        GPExact gp_cpu(hp, Backend::CPU);
        GPExact gp_metal(hp, Backend::Metal);
        LIGHTGP_CHECK(gp_cpu.fit(X, y));
        LIGHTGP_CHECK(gp_metal.fit(X, y));

        Tensor X_test = make_grid(15, 0.5f, 2.0f * kPi - 0.5f);
        Tensor m_cpu, v_cpu, m_metal, v_metal;
        LIGHTGP_CHECK(gp_cpu.predict(X_test, m_cpu, v_cpu));
        LIGHTGP_CHECK(gp_metal.predict(X_test, m_metal, v_metal));
        for (std::size_t i = 0; i < m_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(m_metal.data()[i], m_cpu.data()[i], 5e-4f);
            LIGHTGP_CHECK_NEAR(v_metal.data()[i], v_cpu.data()[i], 5e-4f);
        }
    }
}

}  // namespace lightgp

#else  // LIGHTGP_HAS_METAL

namespace lightgp {
void run_matern_metal_tests() {
    std::printf("[matern_metal] SKIP — built without LIGHTGP_HAS_METAL\n");
}
}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
