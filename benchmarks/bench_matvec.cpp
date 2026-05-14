#include <cstdint>
#include <cstdio>

#include "../core/tensor.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_matvec.h"
#endif

int main() {
    using namespace lightgp;

    const int Ns[] = {1000, 5000, 10000, 20000};
    const int D = 4;
    constexpr int kRuns = 3;

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_matvec  median over %d runs (ms), D=%d, w = (K + sn2 I) v\n",
                kRuns, D);
    std::printf("# Compares explicit K (kernel matrix materialized, then matmul) vs matrix-free.\n");
    std::printf("# N\texplicit_ms\tmatrix_free_ms\tspeedup\n");
    for (int N : Ns) {
        Tensor X = Tensor::randn(N, D, static_cast<std::uint64_t>(N) * 13);
        Tensor v = Tensor::randn(N, 1, static_cast<std::uint64_t>(N) * 17);

        const double explicit_ms = bench::median_ms(kRuns, [&]() {
            Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
            K.add_jitter(1e-2f);
            Tensor w = K.matmul(v);
            (void)w;
        });

        double mf_ms = 0.0;
        if (metal_ok) {
            mf_ms = bench::median_ms(kRuns, [&]() {
#ifdef LIGHTGP_HAS_METAL
                Tensor w = rbf_matvec_metal(X, v, 1.0f, 1.0f, 1e-2f);
                (void)w;
#endif
            });
        }
        if (metal_ok) {
            std::printf("%d\t%.3f\t%.3f\t%.2fx\n",
                        N, explicit_ms, mf_ms, explicit_ms / mf_ms);
        } else {
            std::printf("%d\t%.3f\t-\t-\n", N, explicit_ms);
        }
    }
    return 0;
}
