#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../kernels/cpu/rbf_cpu.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_metal.h"
#endif

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

lightgp::Tensor make_grid(int n, float lo, float hi) {
    lightgp::Tensor X(n, 1);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(n - 1);
    }
    return X;
}

}  // namespace

int main() {
    constexpr float kPi = 3.14159265358979323846f;

    // 1) Fit GP on y = sin(x) + small noise, predict on a fine grid, print a sample.
    {
        const int N_train = 30;
        lightgp::Tensor X_train = make_grid(N_train, 0.0f, 2.0f * kPi);
        lightgp::Tensor y_train(N_train, 1);
        std::mt19937 rng(42);
        std::normal_distribution<float> noise(0.0f, 0.05f);
        for (int i = 0; i < N_train; ++i) {
            y_train(i, 0) = std::sin(X_train(i, 0)) + noise(rng);
        }

        lightgp::GPHyperparams hp;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-2f;
        lightgp::GPExact gp(hp);

        if (!gp.fit(X_train, y_train)) {
            std::fprintf(stderr, "fit failed\n");
            return 1;
        }

        std::printf("Optimizing hyperparameters:\n");
        gp.optimize_hyperparameters(/*num_steps=*/60, /*lr=*/0.05f, /*verbose=*/true);

        const auto& h = gp.hyperparams();
        std::printf("Final hp: l=%.4f sf2=%.4f sn2=%.5f  ll=%.4f\n",
                    h.length_scale, h.signal_variance, h.noise_variance,
                    gp.log_marginal_likelihood());

        lightgp::Tensor X_test = make_grid(11, 0.0f, 2.0f * kPi);
        lightgp::Tensor mean, var;
        gp.predict(X_test, mean, var);
        std::printf("\n  x         mean      ±sd       truth=sin(x)\n");
        for (std::size_t i = 0; i < X_test.rows(); ++i) {
            const float x = X_test(i, 0);
            std::printf("  %6.3f    %+7.4f   %.4f    %+7.4f\n",
                        x, mean(i, 0), std::sqrt(std::max(0.0f, var(i, 0))),
                        std::sin(x));
        }
    }

    // 2) Time CPU vs Metal RBF kernel matrix at increasing N.
    std::printf("\nRBF kernel timing (square N x N, D=4):\n");

#ifdef LIGHTGP_HAS_METAL
    auto& mctx = lightgp::MetalContext::instance();
    if (mctx.available()) {
        std::printf("  Metal: available\n");
    } else {
        std::printf("  Metal: unavailable (%s)\n", mctx.error().c_str());
    }
#else
    std::printf("  Metal: not compiled in\n");
#endif

    std::printf("  %6s  %4s  %12s  %12s  %8s\n", "N", "D", "CPU (ms)", "Metal (ms)", "speedup");
    const int Ns[] = {100, 500, 1000, 2000, 5000};
    const int Ds[] = {4, 16, 64};
    for (int D : Ds) {
        for (int N : Ns) {
            lightgp::Tensor X = lightgp::Tensor::randn(N, D,
                static_cast<std::uint64_t>(N) * 11 + D);

            auto t0 = std::chrono::steady_clock::now();
            lightgp::Tensor Kc = lightgp::rbf_kernel_cpu(X, X, 1.0f, 1.0f);
            const double cpu_ms = seconds_since(t0) * 1000.0;
            (void)Kc;

#ifdef LIGHTGP_HAS_METAL
            // Warmup once at this size to avoid first-dispatch overhead.
            lightgp::Tensor warm = lightgp::rbf_kernel_metal(X, X, 1.0f, 1.0f);
            (void)warm;
            auto t1 = std::chrono::steady_clock::now();
            lightgp::Tensor Km = lightgp::rbf_kernel_metal(X, X, 1.0f, 1.0f);
            const double gpu_ms = seconds_since(t1) * 1000.0;
            (void)Km;
            std::printf("  %6d  %4d  %12.2f  %12.2f  %7.2fx\n",
                        N, D, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
#else
            std::printf("  %6d  %4d  %12.2f  %12s  %8s\n", N, D, cpu_ms, "n/a", "n/a");
#endif
        }
    }

    return 0;
}
