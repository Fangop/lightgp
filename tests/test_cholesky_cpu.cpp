// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#include <cmath>

#include "../core/tensor.h"
#include "../solvers/cpu/cholesky_cpu.h"

namespace lightgp {

void run_cholesky_cpu_tests() {
    std::printf("[cholesky_cpu] starting...\n");

    // K = [[4,2],[2,3]] -> L = [[2,0],[1, sqrt(2)]]
    Tensor K(2, 2, {4.0f, 2.0f, 2.0f, 3.0f});
    Tensor L;
    LIGHTGP_CHECK(cholesky_cpu(K, L));
    LIGHTGP_CHECK_NEAR(L(0, 0), 2.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(L(0, 1), 0.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(L(1, 0), 1.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(L(1, 1), std::sqrt(2.0f), 1e-6f);

    // L L^T == K
    Tensor LLt = L.matmul(L.transpose());
    LIGHTGP_CHECK_NEAR(LLt(0, 0), 4.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(LLt(0, 1), 2.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(LLt(1, 0), 2.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(LLt(1, 1), 3.0f, 1e-5f);

    // log|K| for [[4,2],[2,3]]: det = 12 - 4 = 8.
    const float ld = log_det_from_cholesky(L);
    LIGHTGP_CHECK_NEAR(ld, std::log(8.0f), 1e-5f);

    // Solve K x = b with b = [6,5], expected x = [1,1].
    Tensor b(2, 1, {6.0f, 5.0f});
    Tensor x = cholesky_solve(L, b);
    LIGHTGP_CHECK_NEAR(x(0, 0), 1.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(x(1, 0), 1.0f, 1e-5f);

    // Multi-RHS solve: B = [[6,4],[5,3]] -> X such that K X = B.
    // Column 1: [1,1]; column 2: K^-1 [4,3] = 1/8 * [3*4-2*3, -2*4+4*3] = [6/8, 4/8] = [0.75, 0.5].
    Tensor B2(2, 2, {6.0f, 4.0f, 5.0f, 3.0f});
    Tensor X2 = cholesky_solve(L, B2);
    LIGHTGP_CHECK_NEAR(X2(0, 0), 1.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(X2(1, 0), 1.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(X2(0, 1), 0.75f, 1e-5f);
    LIGHTGP_CHECK_NEAR(X2(1, 1), 0.5f, 1e-5f);

    // Non-PD matrix must fail.
    Tensor NPD(2, 2, {1.0f, 2.0f, 2.0f, 1.0f});  // eigenvalues 3, -1
    Tensor Lnp;
    LIGHTGP_CHECK(!cholesky_cpu(NPD, Lnp));

    // Singular matrix (all zeros): cholesky fails outright, jitter rescues it.
    Tensor Z = Tensor::zeros(3, 3);
    Tensor Lz;
    LIGHTGP_CHECK(!cholesky_cpu(Z, Lz));
    float jit = -1.0f;
    LIGHTGP_CHECK(cholesky_with_jitter(Z, Lz, jit));
    LIGHTGP_CHECK(jit > 0.0f);
    LIGHTGP_CHECK_NEAR(Lz(0, 0), std::sqrt(jit), 1e-5f);

    // 3x3 SPD round-trip: A = M^T M for random M.
    Tensor M = Tensor::randn(3, 3, 7);
    Tensor Mt = M.transpose();
    Tensor A = Mt.matmul(M);
    A.add_jitter(1e-3f);
    Tensor La;
    LIGHTGP_CHECK(cholesky_cpu(A, La));
    Tensor AReco = La.matmul(La.transpose());
    for (std::size_t i = 0; i < A.size(); ++i) {
        LIGHTGP_CHECK_NEAR(AReco.data()[i], A.data()[i], 1e-4f);
    }
}

}  // namespace lightgp
