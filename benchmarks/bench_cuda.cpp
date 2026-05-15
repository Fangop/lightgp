// bench_cuda — single binary that captures every CUDA timing number for the lightgp paper.
// Mirrors bench_paper.cpp's JSON-per-line schema so the records can be joined on
// (method, device, N, D, M) with the CPU/Metal/GPyTorch outputs.
//
// Sizes mirror bench_paper for direct cross-platform comparison (see Stage 2c in task1.md).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

#include "../core/backend.h"
#include "../core/dispatch.h"
#include "../core/kernel.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#include "../kernels/cuda/matern_cuda.h"
#include "../kernels/cuda/rbf_cuda.h"
#include "../kernels/cuda/rbf_matvec_cuda.h"
#include "../solvers/cuda/cholesky_cuda.h"
#include "../solvers/cuda/cholesky_solve_cuda.h"
#endif

namespace {

constexpr const char* kVersion = "lightgp/cuda";

void emit(const std::string& method, const std::string& device,
          long N, long D, long M, double total_ms, int runs,
          const std::string& notes = "") {
    std::printf("{\"method\":\"%s\",\"device\":\"%s\",\"N\":%ld,\"D\":%ld,",
                method.c_str(), device.c_str(), N, D);
    if (M >= 0) std::printf("\"M\":%ld,", M);
    else        std::printf("\"M\":null,");
    std::printf("\"total_ms\":%.4f,\"runs\":%d,\"version\":\"%s\"",
                total_ms, runs, kVersion);
    if (!notes.empty()) std::printf(",\"notes\":\"%s\"", notes.c_str());
    std::printf("}\n");
}

lightgp::Tensor make_X(long N, long D, std::uint64_t seed) {
    return lightgp::Tensor::randn(N, D, seed);
}

lightgp::Tensor make_y(const lightgp::Tensor& X) {
    lightgp::Tensor y(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        float s = 0.0f;
        for (std::size_t d = 0; d < X.cols(); ++d) s += X(i, d);
        y(i, 0) = std::sin(s);
    }
    return y;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace lightgp;
    const int runs = (argc > 1) ? std::atoi(argv[1]) : 3;

#ifndef LIGHTGP_HAS_CUDA
    std::fprintf(stderr, "bench_cuda: compiled without LIGHTGP_HAS_CUDA; exiting.\n");
    return 1;
#else
    if (!CudaContext::instance().available()) {
        std::fprintf(stderr, "bench_cuda: CUDA unavailable (%s).\n",
                     CudaContext::instance().error().c_str());
        return 1;
    }

    // CPU baseline is capped at this N — the reference / OpenBLAS path is slow enough
    // that the larger sizes would dominate wall time without changing the story.
    // CUDA timings sweep the full range.
    constexpr int kCpuCap = 2048;

    // ---- Section: RBF kernel matrix construction ----------------------------------
    for (auto cfg : std::initializer_list<std::pair<int, int>>{
             {1000, 8}, {2048, 4}, {5000, 64}, {8192, 16}}) {
        const int N = cfg.first, D = cfg.second;
        Tensor X = make_X(N, D, 0xC0DEu + N);
        if (N <= kCpuCap) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                (void)rbf_kernel_cpu(X, X, 1.0f, 1.0f);
            });
            emit("rbf_kmat", "cpu", N, D, /*M=*/-1, cpu_ms, runs);
        }
        const double cuda_ms = bench::median_ms(runs, [&]() {
            (void)rbf_kernel_cuda(X, X, 1.0f, 1.0f);
        });
        emit("rbf_kmat", "cuda", N, D, /*M=*/-1, cuda_ms, runs);
    }

    // ---- Section: Cholesky factorization -----------------------------------------
    for (int N : {512, 1024, 2048, 4096}) {
        Tensor X = make_X(N, 8, 0xBEEFu + N);
        Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
        K.add_jitter(1e-3f);
        if (N <= kCpuCap) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                Tensor L; (void)cholesky_cpu(K, L);
            });
            emit("cholesky", "cpu", N, 0, /*M=*/-1, cpu_ms, runs);
        }
        const double cuda_ms = bench::median_ms(runs, [&]() {
            Tensor L; (void)cholesky_cuda(K, L);
        });
        emit("cholesky", "cuda", N, 0, /*M=*/-1, cuda_ms, runs);
    }

    // ---- Section: matrix-free RBF matvec ----------------------------------------
    for (int N : {2000, 5000, 10000, 20000}) {
        Tensor X = make_X(N, 8, 0xF00Du + N);
        Tensor v = Tensor::randn(N, 1, 0xCAFEu + N);
        if (N <= kCpuCap) {
            const double cpu_explicit_ms = bench::median_ms(runs, [&]() {
                Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
                K.add_jitter(1e-2f);
                (void)K.matmul(v);
            });
            emit("rbf_matvec_explicit", "cpu", N, 8, /*M=*/-1, cpu_explicit_ms, runs);
        }
        const double cuda_ms = bench::median_ms(runs, [&]() {
            (void)rbf_matvec_cuda(X, v, 1.0f, 1.0f, 1e-2f);
        });
        emit("rbf_matvec", "cuda", N, 8, /*M=*/-1, cuda_ms, runs,
             "matrix-free, no N^2 allocation");
    }

    // ---- Section: end-to-end GPExact (Cholesky) --------------------------------
    for (auto cfg : std::initializer_list<std::pair<int, int>>{
             {1024, 8}, {2048, 4}, {2048, 16}, {4096, 8}}) {
        const int N = cfg.first, D = cfg.second;
        Tensor X = make_X(N, D, 0x12345u + N);
        Tensor y = make_y(X);
        GPHyperparams hp;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-2f;
        Tensor X_test = make_X(128, D, 0x6789u + N);
        if (N <= kCpuCap) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                GPExact g(hp, Backend::CPU, Solver::Cholesky);
                g.fit(X, y);
                Tensor m, v; g.predict(X_test, m, v);
            });
            emit("gpexact_chol", "cpu", N, D, /*M=*/-1, cpu_ms, runs);
        }
        const double cuda_ms = bench::median_ms(runs, [&]() {
            GPExact g(hp, Backend::CUDA, Solver::Cholesky);
            g.fit(X, y);
            Tensor m, v; g.predict(X_test, m, v);
        });
        emit("gpexact_chol", "cuda", N, D, /*M=*/-1, cuda_ms, runs);
    }

    // ---- Section: GPExact CG (matrix-free on CUDA) -----------------------------
    for (int N : {2048, 5000, 10000}) {
        Tensor X = make_X(N, 8, 0x9ABCu + N);
        Tensor y = make_y(X);
        GPHyperparams hp;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-2f;
        Tensor X_test = make_X(128, 8, 0xDEF0u + N);
        if (N <= kCpuCap) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                GPExact g(hp, Backend::CPU, Solver::CG);
                g.fit(X, y);
                Tensor m, v; g.predict(X_test, m, v);
            });
            emit("gpexact_cg", "cpu", N, 8, /*M=*/-1, cpu_ms, runs, "explicit K_y");
        }
        const double cuda_ms = bench::median_ms(runs, [&]() {
            GPExact g(hp, Backend::CUDA, Solver::CG);
            g.fit(X, y);
            Tensor m, v; g.predict(X_test, m, v);
        });
        emit("gpexact_cg", "cuda", N, 8, /*M=*/-1, cuda_ms, runs, "matrix-free CUDA matvec");
    }

    // ---- Section: GPSparse VFE --------------------------------------------------
    for (auto cfg : std::initializer_list<std::tuple<int, int, int>>{
             {2000, 2, 100}, {10000, 2, 100}, {20000, 4, 200}, {50000, 4, 200}}) {
        const int N = std::get<0>(cfg), D = std::get<1>(cfg), M = std::get<2>(cfg);
        Tensor X = make_X(N, D, 0xACE0u + N);
        Tensor y = make_y(X);
        GPSparseHyperparams hp;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 0.05f;
        Tensor X_test = make_X(128, D, 0xBEE1u + N);
        if (N <= kCpuCap) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                GPSparse g(hp, Backend::CPU);
                g.fit(X, y, M);
                Tensor m, v; g.predict(X_test, m, v);
            });
            emit("gpsparse", "cpu", N, D, /*M=*/M, cpu_ms, runs);
        }
        const double cuda_ms = bench::median_ms(runs, [&]() {
            GPSparse g(hp, Backend::CUDA);
            g.fit(X, y, M);
            Tensor m, v; g.predict(X_test, m, v);
        });
        emit("gpsparse", "cuda", N, D, /*M=*/M, cuda_ms, runs);
    }

    return 0;
#endif
}
