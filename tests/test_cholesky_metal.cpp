#include "test_utils.h"

#ifdef LIGHTGP_HAS_METAL

#include <cmath>

#include "../core/tensor.h"
#include "../kernels/metal/gemm_metal.h"
#include "../kernels/metal/metal_context.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "../solvers/metal/cholesky_metal.h"

namespace lightgp {

void run_cholesky_metal_tests() {
    std::printf("[cholesky_metal] starting...\n");

    if (!MetalContext::instance().available()) {
        std::fprintf(stderr, "[cholesky_metal] SKIP — Metal unavailable: %s\n",
                     MetalContext::instance().error().c_str());
        return;
    }

    // GEMM correctness: agreement with CPU matmul (small — naive path).
    {
        Tensor A = Tensor::randn(50, 30, 1);
        Tensor B = Tensor::randn(30, 40, 2);
        Tensor C_cpu = A.matmul(B);
        Tensor C_metal = gemm_metal(A, B);
        LIGHTGP_CHECK(C_cpu.rows() == C_metal.rows() && C_cpu.cols() == C_metal.cols());
        for (std::size_t i = 0; i < C_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(C_cpu.data()[i], C_metal.data()[i], 1e-3f);
        }
    }
    // GEMM at non-threadgroup-multiple sizes (still naive — M, N < 64).
    {
        Tensor A = Tensor::randn(33, 17, 3);
        Tensor B = Tensor::randn(17, 41, 4);
        Tensor C_cpu = A.matmul(B);
        Tensor C_metal = gemm_metal(A, B);
        for (std::size_t i = 0; i < C_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(C_cpu.data()[i], C_metal.data()[i], 1e-3f);
        }
    }
    // GEMM tiled path: M, N >= 64.
    {
        Tensor A = Tensor::randn(128, 64, 5);
        Tensor B = Tensor::randn(64, 96, 6);
        Tensor C_cpu = A.matmul(B);
        Tensor C_metal = gemm_metal(A, B);
        for (std::size_t i = 0; i < C_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(C_cpu.data()[i], C_metal.data()[i], 5e-3f);
        }
    }
    // GEMM tiled path with non-multiple-of-64 dims (edge tile bounds checking).
    {
        Tensor A = Tensor::randn(100, 73, 7);
        Tensor B = Tensor::randn(73, 89, 8);
        Tensor C_cpu = A.matmul(B);
        Tensor C_metal = gemm_metal(A, B);
        for (std::size_t i = 0; i < C_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(C_cpu.data()[i], C_metal.data()[i], 5e-3f);
        }
    }

    // Blocked Cholesky correctness on SPD matrix: A = M M^T + εI.
    auto random_spd = [](std::size_t n, std::uint64_t seed) {
        Tensor M = Tensor::randn(n, n, seed);
        Tensor A = M.matmul(M.transpose());
        A.add_jitter(1e-2f);
        return A;
    };

    auto compare_reco = [](const Tensor& A, const Tensor& L, float atol) {
        Tensor A_reco = L.matmul(L.transpose());
        LIGHTGP_CHECK(A.rows() == A_reco.rows() && A.cols() == A_reco.cols());
        for (std::size_t i = 0; i < A.size(); ++i) {
            LIGHTGP_CHECK_NEAR(A.data()[i], A_reco.data()[i], atol);
        }
    };

    // N <= block_size: degenerate to CPU path.
    {
        Tensor A = random_spd(50, 10);
        Tensor L;
        LIGHTGP_CHECK(cholesky_metal(A, L, /*block_size=*/64));
        compare_reco(A, L, 1e-3f);
    }
    // N > block_size: exercises Metal trailing-update path.
    {
        Tensor A = random_spd(200, 20);
        Tensor L;
        LIGHTGP_CHECK(cholesky_metal(A, L, /*block_size=*/64));
        compare_reco(A, L, 2e-3f);
    }
    // N not a multiple of block_size.
    {
        Tensor A = random_spd(300, 30);
        Tensor L;
        LIGHTGP_CHECK(cholesky_metal(A, L, /*block_size=*/128));
        compare_reco(A, L, 5e-3f);
    }
    // Larger run with smaller block size — multiple block iterations.
    {
        Tensor A = random_spd(257, 40);
        Tensor L_metal;
        LIGHTGP_CHECK(cholesky_metal(A, L_metal, /*block_size=*/64));
        compare_reco(A, L_metal, 5e-3f);

        // Compare against CPU Cholesky (unique with positive diagonals).
        Tensor L_cpu;
        LIGHTGP_CHECK(cholesky_cpu(A, L_cpu));
        for (std::size_t i = 0; i < L_cpu.size(); ++i) {
            LIGHTGP_CHECK_NEAR(L_metal.data()[i], L_cpu.data()[i], 5e-3f);
        }
    }

    // Jitter retry on a singular matrix.
    {
        Tensor Z = Tensor::zeros(150, 150);
        Tensor L;
        float jit = -1.0f;
        LIGHTGP_CHECK(cholesky_metal_with_jitter(Z, L, jit, /*block_size=*/64));
        LIGHTGP_CHECK(jit > 0.0f);
    }
}

}  // namespace lightgp

#else  // LIGHTGP_HAS_METAL

namespace lightgp {
void run_cholesky_metal_tests() {
    std::printf("[cholesky_metal] SKIP — built without LIGHTGP_HAS_METAL\n");
}
}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
