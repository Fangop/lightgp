#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../core/backend.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#endif

namespace {

double run_e2e(int N, int D, lightgp::Backend backend, lightgp::Solver solver, int runs) {
    using namespace lightgp;
    Tensor X = Tensor::randn(N, D, static_cast<std::uint64_t>(N) * 13);
    Tensor y(N, 1);
    for (int i = 0; i < N; ++i) {
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += X(i, d);
        y(i, 0) = std::sin(s);
    }
    // CG predict variance does one CG solve per test point — keep M_test small in CG mode
    // so the benchmark finishes in reasonable time. Cholesky predict scales fine with M_test.
    const int m_test = (solver == Solver::CG) ? 16 : (N / 4);
    Tensor X_test = Tensor::randn(m_test, D, static_cast<std::uint64_t>(N) * 17);

    GPHyperparams hp;
    hp.length_scale = 1.0f;
    hp.signal_variance = 1.0f;
    hp.noise_variance = 1e-2f;

    return bench::median_ms(runs, [&]() {
        GPExact g(hp, backend, solver);
        g.fit(X, y);
        Tensor m, v;
        g.predict(X_test, m, v);
    });
}

}  // namespace

int main() {
    using namespace lightgp;
    // Unbuffered so long-running rows stream out as they finish.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int D = 4;
    constexpr int kRuns = 3;

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_gp_e2e  median over %d runs (ms), D=%d, fit+predict, warmup discarded\n",
                kRuns, D);
    std::printf("# Cholesky (exact) for moderate N; CG (matrix-free Metal) for scale.\n");
    std::printf("# N\tcpu_chol_ms\tmetal_chol_ms\tchol_speedup\tcg_cpu_ms\tcg_metal_ms\n");

    // Cholesky mode at moderate N.
    const int Ns_chol[] = {128, 256, 512, 1024, 2048};
    for (int N : Ns_chol) {
        const double cpu_ms = run_e2e(N, D, Backend::CPU, Solver::Cholesky, kRuns);
        const double metal_ms = metal_ok ? run_e2e(N, D, Backend::Metal, Solver::Cholesky, kRuns) : 0.0;
        if (metal_ok) {
            std::printf("%d\t%.2f\t%.2f\t%.2fx\t-\t-\n", N, cpu_ms, metal_ms, cpu_ms / metal_ms);
        } else {
            std::printf("%d\t%.2f\t-\t-\t-\t-\n", N, cpu_ms);
        }
    }

    // CG mode at large N where Cholesky becomes prohibitive.
    const int Ns_cg[] = {1000, 2000, 5000};
    for (int N : Ns_cg) {
        const double cg_cpu_ms = run_e2e(N, D, Backend::CPU, Solver::CG, kRuns);
        const double cg_metal_ms = metal_ok ? run_e2e(N, D, Backend::Metal, Solver::CG, kRuns) : 0.0;
        if (metal_ok) {
            std::printf("%d\t-\t-\t-\t%.2f\t%.2f\n", N, cg_cpu_ms, cg_metal_ms);
        } else {
            std::printf("%d\t-\t-\t-\t%.2f\t-\n", N, cg_cpu_ms);
        }
    }

    return 0;
}
