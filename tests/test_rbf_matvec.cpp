// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#ifdef LIGHTGP_HAS_METAL

#include <cmath>

#include "../core/tensor.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_matvec.h"
#include "../solvers/cpu/cg_cpu.h"

namespace lightgp {

void run_rbf_matvec_tests() {
    std::printf("[rbf_matvec] starting...\n");

    if (!MetalContext::instance().available()) {
        std::fprintf(stderr, "[rbf_matvec] SKIP — Metal unavailable: %s\n",
                     MetalContext::instance().error().c_str());
        return;
    }

    auto reference = [](const Tensor& X, const Tensor& v,
                        float l, float sf2, float sn2) {
        Tensor K = rbf_kernel_cpu(X, X, l, sf2);
        K.add_jitter(sn2);
        return K.matmul(v);
    };

    auto compare = [](const Tensor& a, const Tensor& b, float atol) {
        LIGHTGP_CHECK(a.rows() == b.rows() && a.cols() == b.cols());
        for (std::size_t i = 0; i < a.size(); ++i) {
            LIGHTGP_CHECK_NEAR(a.data()[i], b.data()[i], atol);
        }
    };

    // 1D, small N.
    {
        Tensor X = Tensor::randn(50, 1, 11);
        Tensor v = Tensor::randn(50, 1, 12);
        Tensor w_ref = reference(X, v, 1.0f, 1.0f, 1e-3f);
        Tensor w_metal = rbf_matvec_metal(X, v, 1.0f, 1.0f, 1e-3f);
        compare(w_ref, w_metal, 5e-4f);
    }

    // 4D, larger N.
    {
        Tensor X = Tensor::randn(200, 4, 21);
        Tensor v = Tensor::randn(200, 1, 22);
        Tensor w_ref = reference(X, v, 1.3f, 0.8f, 1e-2f);
        Tensor w_metal = rbf_matvec_metal(X, v, 1.3f, 0.8f, 1e-2f);
        compare(w_ref, w_metal, 2e-3f);
    }

    // Non-multiple-of-tile N.
    {
        Tensor X = Tensor::randn(513, 4, 31);
        Tensor v = Tensor::randn(513, 1, 32);
        Tensor w_ref = reference(X, v, 2.0f, 1.5f, 1e-2f);
        Tensor w_metal = rbf_matvec_metal(X, v, 2.0f, 1.5f, 1e-2f);
        compare(w_ref, w_metal, 5e-3f);
    }

    // CG with Metal matvec should produce the same solution as CG with explicit K.
    {
        Tensor X = Tensor::randn(150, 2, 41);
        Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
        K.add_jitter(1e-2f);
        Tensor y = Tensor::randn(150, 1, 42);

        Tensor x_explicit;
        CGResult r1 = cg_solve_cpu(K, y, x_explicit, /*tol=*/1e-5f, /*max_iter=*/500);
        LIGHTGP_CHECK(r1.converged);

        Tensor x_mf;
        auto matvec = [&](const Tensor& v) {
            return rbf_matvec_metal(X, v, 1.0f, 1.0f, 1e-2f);
        };
        CGResult r2 = cg_solve_matvec(matvec, X.rows(), y, x_mf,
                                      /*tol=*/1e-5f, /*max_iter=*/500);
        LIGHTGP_CHECK(r2.converged);
        compare(x_explicit, x_mf, 5e-3f);
    }
}

}  // namespace lightgp

#else  // LIGHTGP_HAS_METAL

namespace lightgp {
void run_rbf_matvec_tests() {
    std::printf("[rbf_matvec] SKIP — built without LIGHTGP_HAS_METAL\n");
}
}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
