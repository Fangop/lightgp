#include "test_utils.h"

#ifdef LIGHTGP_HAS_METAL

#include <cmath>

#include "../core/tensor.h"
#include "../kernels/metal/metal_context.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "../solvers/metal/cholesky_solve_metal.h"

namespace lightgp {

void run_cholesky_solve_metal_tests() {
    std::printf("[cholesky_solve_metal] starting...\n");

    if (!MetalContext::instance().available()) {
        std::fprintf(stderr, "[cholesky_solve_metal] SKIP — Metal unavailable: %s\n",
                     MetalContext::instance().error().c_str());
        return;
    }

    // Known 2x2 SPD: K = [[4,2],[2,3]] -> L = [[2,0],[1, sqrt(2)]], b = [6,5], x = [1,1].
    {
        Tensor K(2, 2, {4.0f, 2.0f, 2.0f, 3.0f});
        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(K, L));
        Tensor b(2, 1, {6.0f, 5.0f});
        Tensor x = cholesky_solve_metal(L, b);
        LIGHTGP_CHECK_NEAR(x(0, 0), 1.0f, 1e-4f);
        LIGHTGP_CHECK_NEAR(x(1, 0), 1.0f, 1e-4f);
    }

    // Random SPD, single RHS — must match CPU cholesky_solve.
    {
        Tensor M = Tensor::randn(50, 50, 1);
        Tensor A = M.matmul(M.transpose());
        A.add_jitter(1e-2f);
        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(A, L));
        Tensor b = Tensor::randn(50, 1, 2);

        Tensor x_cpu = cholesky_solve(L, b);
        Tensor x_metal = cholesky_solve_metal(L, b);
        for (std::size_t i = 0; i < x_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(x_metal.data()[i], x_cpu.data()[i], 1e-3f);
        }
    }

    // Multi-RHS — exercises parallelism across columns.
    {
        Tensor M = Tensor::randn(100, 100, 3);
        Tensor A = M.matmul(M.transpose());
        A.add_jitter(1e-2f);
        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(A, L));
        Tensor B = Tensor::randn(100, 30, 4);

        Tensor X_cpu = cholesky_solve(L, B);
        Tensor X_metal = cholesky_solve_metal(L, B);
        for (std::size_t i = 0; i < X_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(X_metal.data()[i], X_cpu.data()[i], 5e-3f);
        }
    }

    // Larger N with non-power-of-two dimension.
    {
        Tensor M = Tensor::randn(257, 257, 5);
        Tensor A = M.matmul(M.transpose());
        A.add_jitter(1e-2f);
        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(A, L));
        Tensor B = Tensor::randn(257, 16, 6);

        Tensor X_cpu = cholesky_solve(L, B);
        Tensor X_metal = cholesky_solve_metal(L, B);
        for (std::size_t i = 0; i < X_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(X_metal.data()[i], X_cpu.data()[i], 1e-2f);
        }
    }

    // Identity Cholesky factor: trivial solve.
    {
        Tensor L = Tensor::eye(20);
        Tensor B = Tensor::ones(20, 5);
        Tensor X = cholesky_solve_metal(L, B);
        for (std::size_t i = 0; i < X.size(); ++i) {
            LIGHTGP_CHECK_NEAR(X.data()[i], 1.0f, 1e-5f);
        }
    }
}

}  // namespace lightgp

#else  // LIGHTGP_HAS_METAL

namespace lightgp {
void run_cholesky_solve_metal_tests() {
    std::printf("[cholesky_solve_metal] SKIP — built without LIGHTGP_HAS_METAL\n");
}
}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
