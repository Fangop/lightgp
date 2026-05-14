#include <cstdint>
#include <cstdio>

#include "../core/tensor.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#include "../solvers/metal/cholesky_metal.h"
#endif

namespace {
lightgp::Tensor random_spd(std::size_t n, std::uint64_t seed) {
    lightgp::Tensor M = lightgp::Tensor::randn(n, n, seed);
    lightgp::Tensor A = M.matmul(M.transpose());
    A.add_jitter(1e-2f);
    return A;
}
}  // namespace

int main() {
    using namespace lightgp;

    const int Ns[] = {128, 256, 512, 1024, 2048, 4096};
    constexpr int kRuns = 3;
    constexpr int kBlock = 128;

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_cholesky  median over %d runs (ms), block=%d, warmup discarded\n",
                kRuns, kBlock);
    std::printf("# N\tcpu_ms\tmetal_ms\tspeedup\n");
    for (int N : Ns) {
        Tensor A = random_spd(N, static_cast<std::uint64_t>(N) * 7);

        const double cpu_ms = bench::median_ms(kRuns, [&]() {
            Tensor L;
            (void)cholesky_cpu(A, L);
        });

        double metal_ms = 0.0;
        if (metal_ok) {
            metal_ms = bench::median_ms(kRuns, [&]() {
#ifdef LIGHTGP_HAS_METAL
                Tensor L;
                (void)cholesky_metal(A, L, kBlock);
#endif
            });
        }
        if (metal_ok) {
            std::printf("%d\t%.3f\t%.3f\t%.2fx\n", N, cpu_ms, metal_ms, cpu_ms / metal_ms);
        } else {
            std::printf("%d\t%.3f\t-\t-\n", N, cpu_ms);
        }
    }
    return 0;
}
