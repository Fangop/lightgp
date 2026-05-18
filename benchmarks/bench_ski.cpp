// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

// bench_ski — SKI matvec + end-to-end fit/predict timings, JSON-per-line output.
//
// Compares SKI matvec to matrix-free CUDA matvec to dense CPU matvec, and SKI e2e
// to matrix-free CG e2e and sparse VFE e2e. SKI shines once N exceeds ~50k in 1D.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "../core/backend.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"
#include "../inference/ski.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../kernels/rbf_kernel.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#include "../kernels/cuda/rbf_matvec_cuda.h"
#endif

namespace {

void emit(const std::string& method, const std::string& device,
          long N, long M, double total_ms, int runs,
          const std::string& notes = "") {
    std::printf("{\"method\":\"%s\",\"device\":\"%s\",\"N\":%ld,\"M\":%ld,",
                method.c_str(), device.c_str(), N, M);
    std::printf("\"total_ms\":%.4f,\"runs\":%d,\"version\":\"lightgp/ski\"",
                total_ms, runs);
    if (!notes.empty()) std::printf(",\"notes\":\"%s\"", notes.c_str());
    std::printf("}\n");
}

lightgp::Tensor make_X_1d(int N, std::uint64_t seed) {
    lightgp::Tensor X(N, 1);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(0.0f, 10.0f);
    for (int i = 0; i < N; ++i) X(i, 0) = d(rng);
    return X;
}

lightgp::Tensor make_y_sin(const lightgp::Tensor& X) {
    lightgp::Tensor y(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) y(i, 0) = std::sin(X(i, 0));
    return y;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace lightgp;
    const int runs = (argc > 1) ? std::atoi(argv[1]) : 3;

    const Backend backend =
#ifdef LIGHTGP_HAS_CUDA
        CudaContext::instance().available() ? Backend::CUDA :
#endif
        Backend::CPU;

    // ---- SKI matvec scaling -------------------------------------------------
    for (int N : {1000, 5000, 10000, 50000, 100000, 200000}) {
        Tensor X = make_X_1d(N, 0xC0DE + N);
        Tensor v = Tensor::randn(N, 1, 0xBEEF + N);
        const int M = std::max(64, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N)) * 4.0)));
        RBFKernel kernel(0.5f, 1.0f);

        // Build once (cost amortised across many matvecs in a CG solve).
        SKIData ski = build_ski(X, kernel, /*sn2=*/1e-2f, M, backend);
        const double ski_ms = bench::median_ms(runs, [&]() {
            (void)ski.matvec(v);
        });
        emit("ski_matvec", backend == Backend::CUDA ? "cuda" : "cpu",
             N, ski.M(), ski_ms, runs, "amortised, build excluded");

#ifdef LIGHTGP_HAS_CUDA
        if (backend == Backend::CUDA && N <= 50000) {
            // Matrix-free RBF matvec on CUDA (O(N^2)) for comparison.
            const double mf_ms = bench::median_ms(runs, [&]() {
                (void)rbf_matvec_cuda(X, v, 0.5f, 1.0f, 1e-2f);
            });
            emit("rbf_matvec_cuda", "cuda", N, /*M=*/-1, mf_ms, runs, "O(N^2)");
        }
#endif
        if (N <= 5000) {
            // Dense explicit matvec on CPU (baseline).
            const double dense_ms = bench::median_ms(runs, [&]() {
                Tensor K = rbf_kernel_cpu(X, X, 0.5f, 1.0f);
                K.add_jitter(1e-2f);
                (void)K.matmul(v);
            });
            emit("rbf_matvec_explicit", "cpu", N, /*M=*/-1, dense_ms, runs);
        }
    }

    // ---- Section H: SKI end-to-end fit + predict, CPU vs CUDA ----------------
    // 1D heuristic gives M = N grid points, so CPU SKI through kron_toeplitz_matvec_cpu
    // is O(N^2) per CG matvec; the Hutchinson 30-probe variance estimator at predict
    // time multiplies that by ~30 × 100 = 3000 matvecs, putting even N=10000 past the
    // 60s/configuration budget. Skip CPU for all section H sizes and rely on the
    // CPU dense Toeplitz vs SKI cross-check in tests/test_ski.cpp for correctness.
    for (int N : {10000, 50000, 100000, 200000, 500000}) {
        Tensor X = make_X_1d(N, 0xACE + N);
        Tensor y = make_y_sin(X);
        Tensor X_test = make_X_1d(128, 0xBADu + N);

        GPHyperparams hp;
        hp.length_scale = 0.5f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-2f;

        std::printf("{\"section\":\"H\",\"method\":\"gpexact_ski_e2e\",\"device\":\"cpu\","
                    "\"N\":%d,\"D\":1,\"M\":null,\"total_ms\":-1.0,\"runs\":0,"
                    "\"version\":\"lightgp/ski\","
                    "\"notes\":\"skipped: CPU Toeplitz matvec is O(M^2) with M=N for D=1; "
                    "Hutchinson predict pushes median-of-5 past 60s budget\"}\n", N);
        std::fflush(stdout);

        if (backend == Backend::CUDA) {
            const double cuda_ms = bench::median_ms(runs, [&]() {
                GPExact g(hp, Backend::CUDA, Solver::SKI);
                g.fit(X, y);
                Tensor m, v; g.predict(X_test, m, v);
            });
            std::printf("{\"section\":\"H\",\"method\":\"gpexact_ski_e2e\",\"device\":\"cuda\","
                        "\"N\":%d,\"D\":1,\"M\":null,\"total_ms\":%.4f,\"runs\":%d,"
                        "\"version\":\"lightgp/ski\"}\n", N, cuda_ms, runs);
            std::fflush(stdout);
        }

        // Optional comparison: sparse VFE at the same N. Keep emitting with the old
        // (legacy "method") schema for the standalone bench_ski summary.
        if (N <= 50000) {
            const double sparse_ms = bench::median_ms(runs, [&]() {
                GPSparseHyperparams shp;
                shp.length_scale = 0.5f;
                shp.signal_variance = 1.0f;
                shp.noise_variance = 1e-2f;
                GPSparse g(shp, backend);
                g.fit(X, y, /*num_inducing=*/200);
                Tensor m, v; g.predict(X_test, m, v);
            });
            emit("gpsparse_vfe", backend == Backend::CUDA ? "cuda" : "cpu",
                 N, /*M=*/200, sparse_ms, runs);
        }
    }

    return 0;
}
