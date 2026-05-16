#include "test_utils.h"

#include <cmath>
#include <random>
#include <vector>

#include "../core/backend.h"
#include "../core/dispatch.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"
#include "../kernels/cpu/matern_cpu.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../solvers/cpu/cholesky_cpu.h"

#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#include "../kernels/cuda/matern_cuda.h"
#include "../kernels/cuda/rbf_cuda.h"
#include "../kernels/cuda/rbf_matvec_cuda.h"
#include "../solvers/cuda/cholesky_cuda.h"
#include "../solvers/cuda/cholesky_solve_cuda.h"
#endif

namespace lightgp {

namespace {

Tensor random_X(int n, int d, std::uint64_t seed) {
    Tensor X(n, d);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(rng);
    return X;
}

Tensor random_y(int n, std::uint64_t seed) {
    Tensor y(n, 1);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) y(i, 0) = dist(rng);
    return y;
}

#ifdef LIGHTGP_HAS_CUDA
constexpr float kTol = 1e-4f;

void check_close(const Tensor& a, const Tensor& b, float atol) {
    LIGHTGP_CHECK(a.rows() == b.rows());
    LIGHTGP_CHECK(a.cols() == b.cols());
    for (std::size_t i = 0; i < a.size(); ++i) {
        LIGHTGP_CHECK_NEAR(a.data()[i], b.data()[i], atol);
    }
}
#endif

}  // namespace

void run_cuda_tests() {
    std::printf("[cuda] starting...\n");
#ifndef LIGHTGP_HAS_CUDA
    std::printf("[cuda] SKIP — built without LIGHTGP_HAS_CUDA\n");
    return;
#else
    if (!CudaContext::instance().available()) {
        std::printf("[cuda] SKIP — CUDA device unavailable (%s)\n",
                    CudaContext::instance().error().c_str());
        return;
    }

    // RBF: CUDA vs CPU across a few shapes.
    for (int n : {100, 500, 1000}) {
        Tensor X = random_X(n, 8, 0xC0DEu + n);
        Tensor Y = random_X(n / 2 + 3, 8, 0xBEEFu + n);
        Tensor Kc = rbf_kernel_cpu(X, Y, 0.7f, 1.3f);
        Tensor Kg = rbf_kernel_cuda(X, Y, 0.7f, 1.3f);
        check_close(Kc, Kg, kTol);
    }

    // Matérn 1/2, 3/2, 5/2: CUDA vs CPU.
    for (auto type : {KernelType::Matern12, KernelType::Matern32, KernelType::Matern52}) {
        Tensor X = random_X(256, 4, 0xF00Du);
        Tensor Y = random_X(96, 4, 0xC0FFu);
        Tensor Kc = matern_kernel_cpu(X, Y, 0.5f, 0.9f, type);
        Tensor Kg = matern_kernel_cuda(X, Y, 0.5f, 0.9f, type);
        check_close(Kc, Kg, kTol);
    }

    // Cholesky: form K + jitter on CPU, factorize on both, compare.
    {
        Tensor X = random_X(200, 6, 0x1234u);
        Tensor K = rbf_kernel_cpu(X, X, 0.6f, 1.0f);
        K.add_jitter(1e-3f);
        Tensor Lc, Lg;
        LIGHTGP_CHECK(cholesky_cpu(K, Lc));
        LIGHTGP_CHECK(cholesky_cuda(K, Lg));
        // Lower-triangle compare only (LAPACK / cuSOLVER produce identical L given UPLO trick).
        for (std::size_t i = 0; i < Lc.rows(); ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                LIGHTGP_CHECK_NEAR(Lc(i, j), Lg(i, j), 1e-4f);
            }
        }
    }

    // Cholesky-solve: CPU vs CUDA on a multi-RHS problem.
    {
        Tensor X = random_X(150, 4, 0x5678u);
        Tensor K = rbf_kernel_cpu(X, X, 0.8f, 1.1f);
        K.add_jitter(1e-3f);
        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(K, L));
        Tensor B = random_y(150, 0xABCDu);
        Tensor Xc = cholesky_solve(L, B);
        Tensor Xg = cholesky_solve_cuda(L, B);
        check_close(Xc, Xg, 5e-4f);
    }

    // Matrix-free RBF matvec: CUDA vs CPU (explicit K then matmul).
    {
        Tensor X = random_X(400, 8, 0xDEAFu);
        Tensor v = random_y(400, 0xCAFEu);
        const float l = 0.7f, sf2 = 1.2f, sn2 = 0.05f;
        Tensor K = rbf_kernel_cpu(X, X, l, sf2);
        K.add_jitter(sn2);
        Tensor w_ref = K.matmul(v);
        Tensor w_g = rbf_matvec_cuda(X, v, l, sf2, sn2);
        check_close(w_ref, w_g, 5e-4f);
    }

    // End-to-end GPExact: CPU vs CUDA, Cholesky solver.
    {
        const int n = 256;
        Tensor X = random_X(n, 3, 0x11111111u);
        Tensor y(n, 1);
        for (int i = 0; i < n; ++i)
            y(i, 0) = std::sin(X(i, 0)) + 0.1f * X(i, 1) + 0.05f * X(i, 2);

        GPHyperparams hp;
        hp.length_scale = 0.8f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-3f;

        GPExact gpc(hp, Backend::CPU, Solver::Cholesky);
        GPExact gpg(hp, Backend::CUDA, Solver::Cholesky);
        LIGHTGP_CHECK(gpc.fit(X, y));
        LIGHTGP_CHECK(gpg.fit(X, y));

        Tensor X_test = random_X(32, 3, 0x22222222u);
        Tensor mc, vc, mg, vg;
        LIGHTGP_CHECK(gpc.predict(X_test, mc, vc));
        LIGHTGP_CHECK(gpg.predict(X_test, mg, vg));
        check_close(mc, mg, 5e-4f);
        check_close(vc, vg, 5e-4f);
    }

    // End-to-end GPSparse: CPU vs CUDA.
    {
        const int n = 2000;
        Tensor X = random_X(n, 2, 0x33333333u);
        Tensor y(n, 1);
        for (int i = 0; i < n; ++i)
            y(i, 0) = std::sin(2.0f * X(i, 0)) + 0.3f * std::cos(X(i, 1));

        GPSparseHyperparams hp;
        hp.length_scale = 0.5f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 0.05f;

        GPSparse gpc(hp, Backend::CPU);
        GPSparse gpg(hp, Backend::CUDA);
        LIGHTGP_CHECK(gpc.fit(X, y, /*num_inducing=*/100));
        LIGHTGP_CHECK(gpg.fit(X, y, /*num_inducing=*/100));

        Tensor X_test = random_X(64, 2, 0x44444444u);
        Tensor mc, vc, mg, vg;
        LIGHTGP_CHECK(gpc.predict(X_test, mc, vc));
        LIGHTGP_CHECK(gpg.predict(X_test, mg, vg));
        // CPU path uses explicit transpose+matmul through OpenBLAS; CUDA path uses
        // cuBLAS sgemm with op_B=T on the same buffer. Algebraically identical,
        // numerically different in float32 by ~3e-3 at N=2000/M=100. Tolerance set
        // just above this rounding-order drift.
        check_close(mc, mg, 5e-3f);
        check_close(vc, vg, 5e-3f);
    }
#endif  // LIGHTGP_HAS_CUDA
}

}  // namespace lightgp
