// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#include <cmath>

#include "../core/tensor.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../solvers/cpu/cg_cpu.h"
#include "../solvers/cpu/cholesky_cpu.h"

namespace lightgp {

void run_cg_cpu_tests() {
    std::printf("[cg_cpu] starting...\n");

    // Known SPD 2x2: K = [[4,2],[2,3]], b = [6,5]; expected x = [1,1].
    // CG theoretically converges in <= n iterations, but float32 + a non-trivial
    // preconditioner can take a few extra iterations to drive the residual under tol.
    {
        Tensor K(2, 2, {4.0f, 2.0f, 2.0f, 3.0f});
        Tensor b(2, 1, {6.0f, 5.0f});
        Tensor x;
        CGResult r = cg_solve_cpu(K, b, x, /*tol=*/1e-6f, /*max_iter=*/50);
        LIGHTGP_CHECK(r.converged);
        LIGHTGP_CHECK_NEAR(x(0, 0), 1.0f, 1e-4f);
        LIGHTGP_CHECK_NEAR(x(1, 0), 1.0f, 1e-4f);
        LIGHTGP_CHECK(r.iterations <= 10);
    }

    // Random SPD: A = M M^T + ε I; verify CG agrees with Cholesky.
    {
        Tensor M = Tensor::randn(20, 20, 101);
        Tensor A = M.matmul(M.transpose());
        A.add_jitter(1e-2f);
        Tensor b = Tensor::randn(20, 1, 202);

        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(A, L));
        Tensor x_chol = cholesky_solve(L, b);

        Tensor x_cg;
        CGResult r = cg_solve_cpu(A, b, x_cg, /*tol=*/1e-6f, /*max_iter=*/500);
        LIGHTGP_CHECK(r.converged);
        for (std::size_t i = 0; i < x_chol.size(); ++i) {
            LIGHTGP_CHECK_NEAR(x_cg.data()[i], x_chol.data()[i], 1e-3f);
        }
    }

    // RBF kernel + noise: realistic GP context.
    {
        Tensor X = Tensor::randn(30, 2, 303);
        Tensor Kfull = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
        Kfull.add_jitter(1e-2f);
        Tensor y = Tensor::randn(30, 1, 404);

        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(Kfull, L));
        Tensor alpha_chol = cholesky_solve(L, y);

        Tensor alpha_cg;
        CGResult r = cg_solve_cpu(Kfull, y, alpha_cg, /*tol=*/1e-6f, /*max_iter=*/500);
        LIGHTGP_CHECK(r.converged);
        for (std::size_t i = 0; i < alpha_chol.size(); ++i) {
            LIGHTGP_CHECK_NEAR(alpha_cg.data()[i], alpha_chol.data()[i], 1e-3f);
        }
    }

    // Identity: CG converges in one iteration.
    {
        Tensor I = Tensor::eye(10);
        Tensor b = Tensor::ones(10, 1);
        Tensor x;
        CGResult r = cg_solve_cpu(I, b, x, /*tol=*/1e-10f);
        LIGHTGP_CHECK(r.converged);
        LIGHTGP_CHECK(r.iterations == 1);
        for (std::size_t i = 0; i < 10; ++i) LIGHTGP_CHECK_NEAR(x(i, 0), 1.0f, 1e-6f);
    }
}

}  // namespace lightgp
