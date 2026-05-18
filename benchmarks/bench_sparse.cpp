// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../core/backend.h"
#include "../core/tensor.h"
#include "../inference/gp_sparse.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#endif

int main() {
    using namespace lightgp;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const int D = 4;
    constexpr int kRuns = 3;
    struct Cfg { int N; int M; };
    const Cfg cfgs[] = {
        {1000, 50}, {1000, 100},
        {5000, 100}, {5000, 200},
        {10000, 100}, {10000, 200},
        {50000, 200},
    };

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_sparse  median over %d runs (ms), D=%d, fit+predict\n", kRuns, D);
    std::printf("# N\tM\tcpu_ms\tmetal_ms\tspeedup\n");

    for (const Cfg& c : cfgs) {
        Tensor X = Tensor::randn(c.N, D, static_cast<std::uint64_t>(c.N) * 31 + c.M);
        Tensor y(c.N, 1);
        for (int i = 0; i < c.N; ++i) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += X(i, d);
            y(i, 0) = std::sin(s);
        }
        Tensor X_test = Tensor::randn(64, D, static_cast<std::uint64_t>(c.N) * 41);

        GPSparseHyperparams hp;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-2f;

        const double cpu_ms = bench::median_ms(kRuns, [&]() {
            GPSparse g(hp, Backend::CPU);
            g.fit(X, y, c.M);
            Tensor m, v;
            g.predict(X_test, m, v);
        });

        double metal_ms = 0.0;
        if (metal_ok) {
            metal_ms = bench::median_ms(kRuns, [&]() {
                GPSparse g(hp, Backend::Metal);
                g.fit(X, y, c.M);
                Tensor m, v;
                g.predict(X_test, m, v);
            });
        }

        if (metal_ok) {
            std::printf("%d\t%d\t%.2f\t%.2f\t%.2fx\n",
                        c.N, c.M, cpu_ms, metal_ms, cpu_ms / metal_ms);
        } else {
            std::printf("%d\t%d\t%.2f\t-\t-\n", c.N, c.M, cpu_ms);
        }
    }
    return 0;
}
