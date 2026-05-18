// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#ifdef LIGHTGP_HAS_METAL

#include <cmath>

#include "../core/tensor.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_metal.h"

namespace lightgp {

void run_rbf_metal_tests() {
    std::printf("[rbf_metal] starting...\n");

    MetalContext& ctx = MetalContext::instance();
    if (!ctx.available()) {
        std::fprintf(stderr, "[rbf_metal] SKIP — Metal unavailable: %s\n", ctx.error().c_str());
        return;
    }

    auto compare = [](const Tensor& a, const Tensor& b, float atol) {
        LIGHTGP_CHECK(a.rows() == b.rows() && a.cols() == b.cols());
        for (std::size_t i = 0; i < a.size(); ++i) {
            LIGHTGP_CHECK_NEAR(a.data()[i], b.data()[i], atol);
        }
    };

    // 1D, small N.
    {
        Tensor X = Tensor::randn(8, 1, 11);
        Tensor Kc = rbf_kernel_cpu(X, X, 1.3f, 0.7f);
        Tensor Km = rbf_kernel_metal(X, X, 1.3f, 0.7f);
        compare(Kc, Km, 1e-5f);
    }

    // 2D, asymmetric N x M.
    {
        Tensor X1 = Tensor::randn(17, 3, 22);
        Tensor X2 = Tensor::randn(9, 3, 33);
        Tensor Kc = rbf_kernel_cpu(X1, X2, 0.6f, 2.0f);
        Tensor Km = rbf_kernel_metal(X1, X2, 0.6f, 2.0f);
        compare(Kc, Km, 1e-5f);
    }

    // Larger square, varying hyperparameters.
    {
        Tensor X = Tensor::randn(64, 5, 44);
        Tensor Kc = rbf_kernel_cpu(X, X, 2.0f, 1.5f);
        Tensor Km = rbf_kernel_metal(X, X, 2.0f, 1.5f);
        compare(Kc, Km, 1e-5f);
    }

    // Edge: dimensions not divisible by the 16x16 threadgroup.
    {
        Tensor X = Tensor::randn(33, 2, 55);
        Tensor Kc = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
        Tensor Km = rbf_kernel_metal(X, X, 1.0f, 1.0f);
        compare(Kc, Km, 1e-5f);
    }

    // Float4 path: D divisible by 4 — should produce identical results to scalar path.
    {
        Tensor X = Tensor::randn(50, 4, 66);
        Tensor Kc = rbf_kernel_cpu(X, X, 1.2f, 0.9f);
        Tensor Km = rbf_kernel_metal(X, X, 1.2f, 0.9f);
        compare(Kc, Km, 1e-5f);
    }
    {
        Tensor X = Tensor::randn(80, 16, 77);
        Tensor Kc = rbf_kernel_cpu(X, X, 2.5f, 1.5f);
        Tensor Km = rbf_kernel_metal(X, X, 2.5f, 1.5f);
        compare(Kc, Km, 1e-5f);
    }
    // D > D_TILE_F4*4: forces multiple d_base iterations in the f4 kernel.
    {
        Tensor X1 = Tensor::randn(40, 128, 88);
        Tensor X2 = Tensor::randn(25, 128, 99);
        Tensor Kc = rbf_kernel_cpu(X1, X2, 3.0f, 0.5f);
        Tensor Km = rbf_kernel_metal(X1, X2, 3.0f, 0.5f);
        compare(Kc, Km, 1e-5f);
    }
}

}  // namespace lightgp

#else  // LIGHTGP_HAS_METAL

namespace lightgp {
void run_rbf_metal_tests() {
    std::printf("[rbf_metal] SKIP — built without LIGHTGP_HAS_METAL\n");
}
}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
