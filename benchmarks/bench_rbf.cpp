#include <cstdint>
#include <cstdio>

#include "../core/dispatch.h"
#include "../core/tensor.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_metal.h"
#endif

int main() {
    using namespace lightgp;

    const int Ns[] = {100, 500, 1000, 2000, 5000, 10000};
    const int Ds[] = {1, 4, 16, 64};
    constexpr int kRuns = 5;

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_rbf  median over %d runs (ms), warmup discarded\n", kRuns);
    std::printf("# N\tD\tcpu_ms\tmetal_ms\tspeedup\n");
    for (int D : Ds) {
        for (int N : Ns) {
            // Skip huge configurations (N=10000 D=64 ≈ 800MB of K alone).
            if (static_cast<std::size_t>(N) * N > 80'000'000) continue;

            Tensor X = Tensor::randn(N, D, static_cast<std::uint64_t>(N) * 1000 + D);

            const double cpu_ms = bench::median_ms(kRuns, [&]() {
                Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
                (void)K;
            });

            double metal_ms = 0.0;
            if (metal_ok) {
                metal_ms = bench::median_ms(kRuns, [&]() {
#ifdef LIGHTGP_HAS_METAL
                    Tensor K = rbf_kernel_metal(X, X, 1.0f, 1.0f);
                    (void)K;
#endif
                });
            }
            if (metal_ok) {
                std::printf("%d\t%d\t%.3f\t%.3f\t%.2fx\n",
                            N, D, cpu_ms, metal_ms, cpu_ms / metal_ms);
            } else {
                std::printf("%d\t%d\t%.3f\t-\t-\n", N, D, cpu_ms);
            }
        }
    }
    return 0;
}
