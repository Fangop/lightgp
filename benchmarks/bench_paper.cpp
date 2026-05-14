// bench_paper — single binary that produces every timing number in the lightgp paper.
// Emits one JSON record per line to stdout. Schema matches benchmarks/python/bench_gpytorch.py
// so the two outputs can be joined on (method, device, N, D, M) for direct comparison.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/gemm_metal.h"
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_matvec.h"
#include "../kernels/metal/rbf_metal.h"
#include "../solvers/metal/cholesky_metal.h"
#endif

namespace {

constexpr const char* kVersion = "lightgp/week5";

void emit(const std::string& method, const std::string& device,
          long N, long D, long M, double total_ms, int runs,
          const std::string& notes = "") {
    // Inline JSON to avoid pulling in a dependency.
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

lightgp::Tensor random_spd(std::size_t n, std::uint64_t seed) {
    lightgp::Tensor M = lightgp::Tensor::randn(n, n, seed);
    lightgp::Tensor A = M.matmul(M.transpose());
    A.add_jitter(1e-2f);
    return A;
}

// ---------- Section 1: component benchmarks ----------

void section_components(bool metal_ok, int runs) {
    using namespace lightgp;

    // RBF kernel matrix
    const long rbf_Ns[] = {1000, 2000, 5000, 10000};
    const long rbf_Ds[] = {4, 16, 64};
    for (long D : rbf_Ds) {
        for (long N : rbf_Ns) {
            if (static_cast<std::size_t>(N) * N > 80'000'000) continue;
            Tensor X = make_X(N, D, static_cast<std::uint64_t>(N) * 100 + D);
            const double cpu_ms = bench::median_ms(runs, [&]() {
                Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
                (void)K;
            });
            emit("rbf_kernel", "cpu", N, D, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_METAL
            if (metal_ok) {
                const double m_ms = bench::median_ms(runs, [&]() {
                    Tensor K = rbf_kernel_metal(X, X, 1.0f, 1.0f);
                    (void)K;
                });
                emit("rbf_kernel", "metal", N, D, -1, m_ms, runs);
            }
#else
            (void)metal_ok;
#endif
        }
    }

    // Matern-5/2 kernel matrix
    for (long D : rbf_Ds) {
        for (long N : rbf_Ns) {
            if (static_cast<std::size_t>(N) * N > 80'000'000) continue;
            Tensor X = make_X(N, D, static_cast<std::uint64_t>(N) * 200 + D);
            const double cpu_ms = bench::median_ms(runs, [&]() {
                Tensor K = dispatch_kernel(X, X, 1.0f, 1.0f, KernelType::Matern52, Backend::CPU);
                (void)K;
            });
            emit("matern52_kernel", "cpu", N, D, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_METAL
            if (metal_ok) {
                const double m_ms = bench::median_ms(runs, [&]() {
                    Tensor K = dispatch_kernel(X, X, 1.0f, 1.0f, KernelType::Matern52, Backend::Metal);
                    (void)K;
                });
                emit("matern52_kernel", "metal", N, D, -1, m_ms, runs);
            }
#endif
        }
    }

    // Cholesky
    const long ch_Ns[] = {512, 1024, 2048, 4096};
    for (long N : ch_Ns) {
        Tensor A = random_spd(N, static_cast<std::uint64_t>(N) * 7);
        const double cpu_ms = bench::median_ms(runs, [&]() {
            Tensor L;
            (void)cholesky_cpu(A, L);
        });
        emit("cholesky", "cpu", N, -1, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_METAL
        if (metal_ok) {
            const double m_ms = bench::median_ms(runs, [&]() {
                Tensor L;
                (void)cholesky_metal(A, L, 128);
            });
            emit("cholesky", "metal", N, -1, -1, m_ms, runs);
        }
#endif
    }

    // GEMM — naive vs tiled vs simdgroup (Metal dispatches automatically)
    for (long N : ch_Ns) {
        Tensor A = Tensor::randn(N, N, static_cast<std::uint64_t>(N) * 11);
        Tensor B = Tensor::randn(N, N, static_cast<std::uint64_t>(N) * 13);
        const double cpu_ms = bench::median_ms(runs, [&]() {
            Tensor C = A.matmul(B);
            (void)C;
        });
        emit("gemm", "cpu", N, -1, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_METAL
        if (metal_ok) {
            const double m_ms = bench::median_ms(runs, [&]() {
                Tensor C = gemm_metal(A, B);
                (void)C;
            });
            emit("gemm", "metal", N, -1, -1, m_ms, runs);
        }
#endif
    }

    // Matrix-free matvec — only meaningful with Metal
#ifdef LIGHTGP_HAS_METAL
    if (metal_ok) {
        const long mv_Ns[] = {5000, 10000, 20000, 50000};
        for (long N : mv_Ns) {
            Tensor X = make_X(N, 4, static_cast<std::uint64_t>(N) * 19);
            Tensor v = Tensor::randn(N, 1, static_cast<std::uint64_t>(N) * 23);
            const double m_ms = bench::median_ms(runs, [&]() {
                Tensor w = rbf_matvec_metal(X, v, 1.0f, 1.0f, 1e-2f);
                (void)w;
            });
            emit("matvec_free", "metal", N, 4, -1, m_ms, runs);
        }
    }
#endif
}

// ---------- Section 2: end-to-end exact GP (Cholesky) ----------

double run_exact(long N, long D, lightgp::Backend backend, lightgp::KernelType kt, int runs) {
    using namespace lightgp;
    Tensor X = make_X(N, D, static_cast<std::uint64_t>(N) * 31);
    Tensor y = make_y(X);
    Tensor Xt = make_X(N / 4, D, static_cast<std::uint64_t>(N) * 37);
    GPHyperparams hp;
    hp.kernel = kt;
    hp.length_scale = 1.0f;
    hp.signal_variance = 1.0f;
    hp.noise_variance = 1e-2f;
    return bench::median_ms(runs, [&]() {
        GPExact g(hp, backend, Solver::Cholesky);
        g.fit(X, y);
        Tensor m, v;
        g.predict(Xt, m, v);
    });
}

void section_exact(bool metal_ok, int runs) {
    using namespace lightgp;
    const long Ns[] = {256, 512, 1024, 2048};
    for (long N : Ns) {
        for (KernelType kt : {KernelType::RBF, KernelType::Matern52}) {
            const std::string mname = (kt == KernelType::RBF) ? "exact_rbf" : "exact_matern52";
            const double cpu_ms = run_exact(N, 4, Backend::CPU, kt, runs);
            emit(mname, "cpu", N, 4, -1, cpu_ms, runs);
#ifdef LIGHTGP_HAS_METAL
            if (metal_ok) {
                const double m_ms = run_exact(N, 4, Backend::Metal, kt, runs);
                emit(mname, "metal", N, 4, -1, m_ms, runs);
            }
#endif
        }
    }
}

// ---------- Section 3: CG GP — small N grid to keep wall time bounded ----------

double run_cg(long N, long D, lightgp::Backend backend, int runs) {
    using namespace lightgp;
    Tensor X = make_X(N, D, static_cast<std::uint64_t>(N) * 41);
    Tensor y = make_y(X);
    Tensor Xt = make_X(16, D, static_cast<std::uint64_t>(N) * 43);  // 16 test points (probe variance independent of M_test)
    GPHyperparams hp;
    hp.length_scale = 1.0f;
    hp.signal_variance = 1.0f;
    hp.noise_variance = 1e-2f;
    return bench::median_ms(runs, [&]() {
        GPExact g(hp, backend, Solver::CG);
        g.fit(X, y);
        Tensor m, v;
        g.predict(Xt, m, v);
    });
}

void section_cg(bool metal_ok, int runs) {
    using namespace lightgp;
    // Keep CG section small — N=2000 already takes minutes on CPU.
    const long Ns[] = {1000, 2000};
    for (long N : Ns) {
#ifdef LIGHTGP_HAS_METAL
        if (metal_ok) {
            const double m_ms = run_cg(N, 4, Backend::Metal, runs);
            emit("cg_rbf", "metal", N, 4, -1, m_ms, runs,
                 "matrix-free matvec + Hutchinson predictive variance");
        }
#endif
        const double cpu_ms = run_cg(N, 4, Backend::CPU, runs);
        emit("cg_rbf", "cpu", N, 4, -1, cpu_ms, runs);
    }
}

// ---------- Section 4: sparse GP ----------

double run_sparse(long N, long D, long M, lightgp::Backend backend, int runs) {
    using namespace lightgp;
    Tensor X = make_X(N, D, static_cast<std::uint64_t>(N) * 53 + M);
    Tensor y = make_y(X);
    Tensor Xt = make_X(64, D, static_cast<std::uint64_t>(N) * 59);
    GPSparseHyperparams hp;
    hp.length_scale = 1.0f;
    hp.signal_variance = 1.0f;
    hp.noise_variance = 1e-2f;
    return bench::median_ms(runs, [&]() {
        GPSparse g(hp, backend);
        g.fit(X, y, M);
        Tensor m, v;
        g.predict(Xt, m, v);
    });
}

void section_sparse(bool metal_ok, int runs) {
    using namespace lightgp;
    struct Cfg { long N; long M; };
    const Cfg cfgs[] = {
        {1000, 50}, {1000, 100},
        {5000, 100}, {5000, 200},
        {10000, 100}, {10000, 200},
        {50000, 200},
    };
    for (const Cfg& c : cfgs) {
        const double cpu_ms = run_sparse(c.N, 4, c.M, Backend::CPU, runs);
        emit("sparse_rbf", "cpu", c.N, 4, c.M, cpu_ms, runs);
#ifdef LIGHTGP_HAS_METAL
        if (metal_ok) {
            const double m_ms = run_sparse(c.N, 4, c.M, Backend::Metal, runs);
            emit("sparse_rbf", "metal", c.N, 4, c.M, m_ms, runs);
        }
#endif
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace lightgp;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    bool run_components = true, run_exact = true, run_cg = true, run_sparse = true;
    int runs = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--skip-components") run_components = false;
        else if (a == "--skip-exact") run_exact = false;
        else if (a == "--skip-cg") run_cg = false;
        else if (a == "--skip-sparse") run_sparse = false;
        else if (a == "--only-sparse") {
            run_components = run_exact = run_cg = false;
        } else if (a == "--help") {
            std::fprintf(stderr,
                "bench_paper: comprehensive paper benchmarks → JSON-per-line on stdout\n"
                "  --runs N           runs per cell (default 3)\n"
                "  --skip-components  skip section 1\n"
                "  --skip-exact       skip section 2\n"
                "  --skip-cg          skip section 3 (long-running)\n"
                "  --skip-sparse      skip section 4\n"
                "  --only-sparse      shorthand for --skip-{components,exact,cg}\n");
            return 0;
        }
    }

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::fprintf(stderr, "# bench_paper: runs=%d  metal=%s\n",
                 runs, metal_ok ? "yes" : "no");

    if (run_components) section_components(metal_ok, runs);
    if (run_exact)      section_exact(metal_ok, runs);
    if (run_cg)         section_cg(metal_ok, runs);
    if (run_sparse)     section_sparse(metal_ok, runs);

    return 0;
}
