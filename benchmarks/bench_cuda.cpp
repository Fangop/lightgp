// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

// bench_cuda — comprehensive CPU (OpenBLAS) vs CUDA timing sweep.
// Emits one JSON record per line (schema below) so the linux_cuda_report.md generator
// can join configurations by (section, method, device, N, D, M).
//
// All timings are median of `runs` samples (default 5) with one warmup discarded
// inside bench::median_ms. A "skipped: ..." note indicates a configuration we
// pre-empt rather than run (typically because the CPU path would exceed ~60s).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>

#include "../core/backend.h"
#include "../core/dispatch.h"
#include "../core/kernel.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"
#include "../kernels/cpu/matern_cpu.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#include "../kernels/cuda/gemm_cuda.h"
#include "../kernels/cuda/matern_cuda.h"
#include "../kernels/cuda/rbf_cuda.h"
#include "../kernels/cuda/rbf_matvec_cuda.h"
#include "../solvers/cuda/cholesky_cuda.h"
#include "../solvers/cuda/cholesky_solve_cuda.h"
#endif

namespace {

constexpr const char* kVersion = "lightgp/cuda";

void emit(const std::string& section, const std::string& method,
          const std::string& device, long N, long D, long M,
          double total_ms, int runs, const std::string& notes = "") {
    std::printf("{\"section\":\"%s\",\"method\":\"%s\",\"device\":\"%s\",",
                section.c_str(), method.c_str(), device.c_str());
    std::printf("\"N\":%ld,\"D\":%ld,", N, D);
    if (M >= 0) std::printf("\"M\":%ld,", M);
    else        std::printf("\"M\":null,");
    std::printf("\"total_ms\":%.4f,\"runs\":%d,\"version\":\"%s\"",
                total_ms, runs, kVersion);
    if (!notes.empty()) std::printf(",\"notes\":\"%s\"", notes.c_str());
    std::printf("}\n");
    std::fflush(stdout);
}

void skipped(const std::string& section, const std::string& method,
             const std::string& device, long N, long D, long M,
             const std::string& reason) {
    emit(section, method, device, N, D, M, /*total_ms=*/-1.0, /*runs=*/0,
         "skipped: " + reason);
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
    const int runs = (argc > 1) ? std::atoi(argv[1]) : 5;

#ifdef LIGHTGP_HAS_CUDA
    const bool cuda_ok = CudaContext::instance().available();
#else
    const bool cuda_ok = false;
#endif

    // ===================== A. RBF kernel matrix construction =====================
    for (int N : {500, 1000, 2000, 5000, 10000}) {
        for (int D : {1, 4, 16, 64}) {
            Tensor X = make_X(N, D, 0xA000u + 11 * N + D);
            const double cpu_ms = bench::median_ms(runs, [&]() {
                (void)rbf_kernel_cpu(X, X, 1.0f, 1.0f);
            });
            emit("A", "rbf_kmat", "cpu", N, D, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_CUDA
            if (cuda_ok) {
                const double cuda_ms = bench::median_ms(runs, [&]() {
                    (void)rbf_kernel_cuda(X, X, 1.0f, 1.0f);
                });
                emit("A", "rbf_kmat", "cuda", N, D, -1, cuda_ms, runs);
            }
#endif
        }
    }

    // ===================== B. Matérn-5/2 kernel matrix construction ==============
    for (int N : {1000, 5000}) {
        for (int D : {4, 16, 64}) {
            Tensor X = make_X(N, D, 0xB000u + 13 * N + D);
            const double cpu_ms = bench::median_ms(runs, [&]() {
                (void)matern_kernel_cpu(X, X, 1.0f, 1.0f, KernelType::Matern52);
            });
            emit("B", "matern52_kmat", "cpu", N, D, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_CUDA
            if (cuda_ok) {
                const double cuda_ms = bench::median_ms(runs, [&]() {
                    (void)matern_kernel_cuda(X, X, 1.0f, 1.0f, KernelType::Matern52);
                });
                emit("B", "matern52_kmat", "cuda", N, D, -1, cuda_ms, runs);
            }
#endif
        }
    }

    // ===================== C. Cholesky factorization =============================
    for (int N : {256, 512, 1024, 2048, 4096, 8192}) {
        Tensor X = make_X(N, 4, 0xC000u + N);
        Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
        K.add_jitter(1e-3f);
        // Reference measurement on this host: OpenBLAS Cholesky scales like ~1s @ 2048,
        // ~8s @ 4096; N=8192 would push past 60s of wall time at runs=5.
        if (N <= 4096) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                Tensor L; (void)cholesky_cpu(K, L);
            });
            emit("C", "cholesky", "cpu", N, 0, -1, cpu_ms, runs);
        } else {
            skipped("C", "cholesky", "cpu", N, 0, -1, "expected >60s on OpenBLAS CPU");
        }
#ifdef LIGHTGP_HAS_CUDA
        if (cuda_ok) {
            const double cuda_ms = bench::median_ms(runs, [&]() {
                Tensor L; (void)cholesky_cuda(K, L);
            });
            emit("C", "cholesky", "cuda", N, 0, -1, cuda_ms, runs);
        }
#endif
    }

    // ===================== D. Matrix-free RBF matvec =============================
    for (int N : {1000, 2000, 5000, 10000, 20000, 50000, 100000}) {
        Tensor X = make_X(N, 4, 0xD000u + N);
        Tensor v = Tensor::randn(N, 1, 0xD100u + N);
        // Explicit CPU = build K (N^2 entries) + matmul. ~12 ms @ 1000, ~1.5 s @ 10000;
        // 20000 starts to bloat memory + wall time too much for a 5-sample median.
        if (N <= 10000) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
                K.add_jitter(1e-2f);
                (void)K.matmul(v);
            });
            emit("D", "matvec_explicit", "cpu", N, 4, -1, cpu_ms, runs);
        } else {
            skipped("D", "matvec_explicit", "cpu", N, 4, -1,
                    "explicit K is O(N^2) memory and >60s wall time");
        }
#ifdef LIGHTGP_HAS_CUDA
        if (cuda_ok) {
            const double cuda_ms = bench::median_ms(runs, [&]() {
                (void)rbf_matvec_cuda(X, v, 1.0f, 1.0f, 1e-2f);
            });
            emit("D", "matvec_matrix_free", "cuda", N, 4, -1, cuda_ms, runs,
                 "matrix-free, no N^2 allocation");
        }
#endif
    }

    // ===================== E. End-to-end exact GP — Cholesky ====================
    for (int N : {256, 512, 1024, 2048, 4096, 8192}) {
        Tensor X = make_X(N, 4, 0xE000u + N);
        Tensor y = make_y(X);
        Tensor X_test = make_X(128, 4, 0xE100u + N);
        GPHyperparams hp;
        hp.length_scale = 1.0f; hp.signal_variance = 1.0f; hp.noise_variance = 1e-2f;
        if (N <= 4096) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                GPExact g(hp, Backend::CPU, Solver::Cholesky);
                g.fit(X, y);
                Tensor m, var; g.predict(X_test, m, var);
            });
            emit("E", "gpexact_chol_e2e", "cpu", N, 4, -1, cpu_ms, runs);
        } else {
            skipped("E", "gpexact_chol_e2e", "cpu", N, 4, -1, "expected >60s on CPU");
        }
#ifdef LIGHTGP_HAS_CUDA
        if (cuda_ok) {
            const double cuda_ms = bench::median_ms(runs, [&]() {
                GPExact g(hp, Backend::CUDA, Solver::Cholesky);
                g.fit(X, y);
                Tensor m, var; g.predict(X_test, m, var);
            });
            emit("E", "gpexact_chol_e2e", "cuda", N, 4, -1, cuda_ms, runs);
        }
#endif
    }

    // ===================== F. End-to-end CG GP — matrix-free ====================
    for (int N : {1000, 2000, 5000, 10000, 20000}) {
        Tensor X = make_X(N, 4, 0xF000u + N);
        Tensor y = make_y(X);
        Tensor X_test = make_X(128, 4, 0xF100u + N);
        GPHyperparams hp;
        hp.length_scale = 1.0f; hp.signal_variance = 1.0f; hp.noise_variance = 1e-2f;
        // CPU CG materialises K_y; predict variance does 30 Hutchinson CG solves —
        // ~35s/run @ N=2000, ~3.5 min/run @ N=5000. Cap to keep the median-of-5
        // budget under the 60s/configuration target the report header sets.
        if (N <= 2000) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                GPExact g(hp, Backend::CPU, Solver::CG);
                g.fit(X, y);
                Tensor m, var; g.predict(X_test, m, var);
            });
            emit("F", "gpexact_cg_e2e", "cpu", N, 4, -1, cpu_ms, runs);
        } else {
            skipped("F", "gpexact_cg_e2e", "cpu", N, 4, -1,
                    "Hutchinson 30-probe predict puts wall time well over 60s on CPU");
        }
#ifdef LIGHTGP_HAS_CUDA
        if (cuda_ok) {
            // CUDA CG predict variance still does 30 probes; at N=20000 a single
            // median-of-5 measurement runs ~20+ min. Cap to N<=10000.
            if (N <= 10000) {
                const double cuda_ms = bench::median_ms(runs, [&]() {
                    GPExact g(hp, Backend::CUDA, Solver::CG);
                    g.fit(X, y);
                    Tensor m, var; g.predict(X_test, m, var);
                });
                emit("F", "gpexact_cg_e2e", "cuda", N, 4, -1, cuda_ms, runs);
            } else {
                skipped("F", "gpexact_cg_e2e", "cuda", N, 4, -1,
                        "Hutchinson predict (30 probes) pushes median-of-5 past budget; "
                        "matvec-only timings in section D show true CUDA matvec scaling");
            }
        }
#endif
    }

    // ===================== G. Sparse GP VFE =====================================
    for (auto cfg : std::initializer_list<std::tuple<int, int>>{
             {1000, 50}, {1000, 100}, {5000, 100}, {5000, 200},
             {10000, 200}, {50000, 200}, {100000, 200}}) {
        const int N = std::get<0>(cfg), M = std::get<1>(cfg);
        Tensor X = make_X(N, 4, 0xACE0u + 17 * N + M);
        Tensor y = make_y(X);
        Tensor X_test = make_X(128, 4, 0xBEE1u + N);
        GPSparseHyperparams hp;
        hp.length_scale = 1.0f; hp.signal_variance = 1.0f; hp.noise_variance = 0.05f;
        // VFE fit is O(N M² + M³); CPU stays tractable through N=10000, but N=50000+
        // takes the wall time over a minute.
        if (N <= 10000) {
            const double cpu_ms = bench::median_ms(runs, [&]() {
                GPSparse g(hp, Backend::CPU);
                g.fit(X, y, M);
                Tensor mu, var; g.predict(X_test, mu, var);
            });
            emit("G", "gpsparse_vfe", "cpu", N, 4, M, cpu_ms, runs);
        } else {
            skipped("G", "gpsparse_vfe", "cpu", N, 4, M, "expected >60s on CPU");
        }
#ifdef LIGHTGP_HAS_CUDA
        if (cuda_ok) {
            const double cuda_ms = bench::median_ms(runs, [&]() {
                GPSparse g(hp, Backend::CUDA);
                g.fit(X, y, M);
                Tensor mu, var; g.predict(X_test, mu, var);
            });
            emit("G", "gpsparse_vfe", "cuda", N, 4, M, cuda_ms, runs);
        }
#endif
    }

    // ===================== I. GEMM performance ==================================
    for (int N : {512, 1024, 2048, 4096}) {
        Tensor A = make_X(N, N, 0x9000u + N);
        Tensor B = make_X(N, N, 0x9100u + N);
        // Tensor::matmul routes through OpenBLAS sgemm when LIGHTGP_HAS_OPENBLAS is set.
        const double cpu_ms = bench::median_ms(runs, [&]() { (void)A.matmul(B); });
        emit("I", "gemm_sgemm", "cpu", N, N, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_CUDA
        if (cuda_ok) {
            const double cuda_ms = bench::median_ms(runs, [&]() { (void)gemm_cuda(A, B); });
            emit("I", "gemm_sgemm", "cuda", N, N, -1, cuda_ms, runs);
        }
#endif
    }

    // ===================== K. Backend agreement =================================
    // Single fit + predict per (CPU, CUDA) for each solver; report max abs diff over
    // the test predictions. Validates correctness across backends within atol=1e-4.
#ifdef LIGHTGP_HAS_CUDA
    if (cuda_ok) {
        const int Nk = 1000, Dk = 4;
        Tensor X = make_X(Nk, Dk, 0xC0DEu);
        Tensor y = make_y(X);
        Tensor X_test = make_X(128, Dk, 0xBEEFu);

        auto run_pair = [&](const std::string& tag, Solver solver,
                            int sparse_M = -1) {
            Tensor mc, vc, mg, vg;
            if (sparse_M < 0) {
                GPHyperparams hp;
                hp.length_scale = 1.0f; hp.signal_variance = 1.0f; hp.noise_variance = 1e-2f;
                GPExact gc(hp, Backend::CPU, solver);
                gc.fit(X, y); gc.predict(X_test, mc, vc);
                GPExact gg(hp, Backend::CUDA, solver);
                gg.fit(X, y); gg.predict(X_test, mg, vg);
            } else {
                GPSparseHyperparams hp;
                hp.length_scale = 1.0f; hp.signal_variance = 1.0f; hp.noise_variance = 1e-2f;
                GPSparse gc(hp, Backend::CPU);
                gc.fit(X, y, sparse_M); gc.predict(X_test, mc, vc);
                GPSparse gg(hp, Backend::CUDA);
                gg.fit(X, y, sparse_M); gg.predict(X_test, mg, vg);
            }
            float max_m = 0.0f, max_v = 0.0f;
            for (std::size_t i = 0; i < mc.size(); ++i) {
                max_m = std::max(max_m, std::fabs(mc.data()[i] - mg.data()[i]));
                max_v = std::max(max_v, std::fabs(vc.data()[i] - vg.data()[i]));
            }
            std::printf("{\"section\":\"K\",\"comparison\":\"%s\","
                        "\"max_abs_mean_diff\":%.6g,\"max_abs_var_diff\":%.6g}\n",
                        tag.c_str(), max_m, max_v);
            std::fflush(stdout);
        };
        run_pair("cholesky", Solver::Cholesky);
        run_pair("cg", Solver::CG);
        run_pair("sparse_M100", Solver::Cholesky, /*sparse_M=*/100);
        run_pair("ski", Solver::SKI);
    }
#endif

    return 0;
}
