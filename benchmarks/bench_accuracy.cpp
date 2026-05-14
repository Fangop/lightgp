// bench_accuracy — numerical correctness validation across (method × kernel × backend).
// Reports RMSE, mean NLL, and 2σ calibration on a synthetic regression task.
// Also cross-checks Metal vs CPU prediction agreement.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>

#include "../core/backend.h"
#include "../core/kernel.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;

float true_fn(float x) {
    return std::sin(x) + 0.1f * std::cos(3.0f * x);
}

void make_data(int N, float lo, float hi, float noise_std, std::uint64_t seed,
               lightgp::Tensor& X, lightgp::Tensor& y, lightgp::Tensor& y_clean) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    std::normal_distribution<float> n(0.0f, noise_std);
    X = lightgp::Tensor(N, 1);
    y = lightgp::Tensor(N, 1);
    y_clean = lightgp::Tensor(N, 1);
    for (int i = 0; i < N; ++i) {
        const float x = u(rng);
        X(i, 0) = x;
        y_clean(i, 0) = true_fn(x);
        y(i, 0) = y_clean(i, 0) + n(rng);
    }
}

struct Metrics {
    float rmse;
    float mean_nll;
    float frac_within_2sigma;
};

// Standard GP regression metrics:
//   RMSE: against the noise-free target (function-recovery quality).
//   NLL + calibration: against the noisy observation, with noise variance added
//   to the predicted *latent* variance — GPExact/GPSparse return f* variance, not y* variance.
Metrics evaluate(const lightgp::Tensor& mean, const lightgp::Tensor& var,
                 const lightgp::Tensor& y_clean, const lightgp::Tensor& y_noisy,
                 float noise_var) {
    const std::size_t N = mean.rows();
    float sse = 0.0f;
    double nll_sum = 0.0;
    int within_2 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const float err_clean = mean(i, 0) - y_clean(i, 0);
        sse += err_clean * err_clean;
        const float err_noisy = mean(i, 0) - y_noisy(i, 0);
        // Predictive variance for y* (observation) = var_f* + sigma_n^2.
        const float v = std::max(var(i, 0) + noise_var, 1e-6f);
        nll_sum += 0.5 * (std::log(2.0 * kPi * v) + (err_noisy * err_noisy) / v);
        if (std::fabs(err_noisy) <= 2.0f * std::sqrt(v)) ++within_2;
    }
    Metrics m;
    m.rmse = std::sqrt(sse / static_cast<float>(N));
    m.mean_nll = static_cast<float>(nll_sum / N);
    m.frac_within_2sigma = static_cast<float>(within_2) / static_cast<float>(N);
    return m;
}

void emit(const char* method, const char* kernel, const char* backend,
          int N_train, int M, const Metrics& m) {
    std::printf("%-26s %-10s %-7s  N=%4d  M=%4s  RMSE=%.4f  meanNLL=%+8.4f  cal2σ=%5.1f%%\n",
                method, kernel, backend, N_train,
                (M > 0 ? std::to_string(M).c_str() : "-"),
                m.rmse, m.mean_nll, 100.0f * m.frac_within_2sigma);
}

const char* kernel_name(lightgp::KernelType k) {
    switch (k) {
        case lightgp::KernelType::RBF:      return "RBF";
        case lightgp::KernelType::Matern12: return "Matern12";
        case lightgp::KernelType::Matern32: return "Matern32";
        case lightgp::KernelType::Matern52: return "Matern52";
    }
    return "?";
}

const char* backend_name(lightgp::Backend b) {
    switch (b) {
        case lightgp::Backend::CPU:   return "cpu";
        case lightgp::Backend::Metal: return "metal";
        case lightgp::Backend::CUDA:  return "cuda";
    }
    return "?";
}

}  // namespace

int main() {
    using namespace lightgp;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    constexpr int N_train = 200;
    constexpr int N_test = 100;
    constexpr float lo = 0.0f, hi = 2.0f * kPi;
    constexpr float noise = 0.1f;

    Tensor X_train, y_train, y_train_clean;
    Tensor X_test, y_test, y_test_clean;
    make_data(N_train, lo, hi, noise, /*seed=*/42, X_train, y_train, y_train_clean);
    make_data(N_test,  lo, hi, noise, /*seed=*/43, X_test,  y_test,  y_test_clean);

#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_accuracy: f(x) = sin(x) + 0.1 cos(3x) + N(0, %.2f^2),  "
                "N_train=%d  N_test=%d\n", noise, N_train, N_test);
    std::printf("# RMSE / mean NLL / 2σ calibration computed against noisy test targets.\n");
    std::printf("# %-26s %-10s %-7s\n", "method", "kernel", "backend");

    const Backend backends[] = {Backend::CPU, Backend::Metal};
    const KernelType kernels[] = {KernelType::RBF, KernelType::Matern52};

    // Exact GP (Cholesky) across kernel × backend.
    for (KernelType kt : kernels) {
        for (Backend bk : backends) {
            if (bk == Backend::Metal && !metal_ok) continue;
            GPHyperparams hp;
            hp.kernel = kt;
            hp.length_scale = 1.0f;
            hp.signal_variance = 1.0f;
            hp.noise_variance = noise * noise;
            GPExact gp(hp, bk, Solver::Cholesky);
            gp.fit(X_train, y_train);
            gp.optimize_hyperparameters(/*steps=*/40, /*lr=*/0.05f, /*verbose=*/false);
            Tensor m, v;
            gp.predict(X_test, m, v);
            emit("exact (Cholesky)", kernel_name(kt), backend_name(bk),
                 N_train, /*M=*/0,
                 evaluate(m, v, y_test_clean, y_test, hp.noise_variance));
        }
    }

    // CG GP — RBF only (Metal matvec is RBF-specific; CG with Matern would use materialized path).
    for (Backend bk : backends) {
        if (bk == Backend::Metal && !metal_ok) continue;
        GPHyperparams hp;
        hp.kernel = KernelType::RBF;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = noise * noise;
        GPExact gp(hp, bk, Solver::CG);
        gp.fit(X_train, y_train);
        Tensor m, v;
        gp.predict(X_test, m, v);
        emit("CG (Hutchinson var)", "RBF", backend_name(bk),
             N_train, /*M=*/0,
             evaluate(m, v, y_test_clean, y_test, hp.noise_variance));
    }

    // Sparse VFE GP — vary M to show fidelity vs num inducing.
    for (int M : {10, 25, 50, 100}) {
        for (Backend bk : backends) {
            if (bk == Backend::Metal && !metal_ok) continue;
            GPSparseHyperparams hp;
            hp.kernel = KernelType::RBF;
            hp.length_scale = 1.0f;
            hp.signal_variance = 1.0f;
            hp.noise_variance = noise * noise;
            GPSparse gp(hp, bk);
            gp.fit(X_train, y_train, M);
            Tensor m, v;
            gp.predict(X_test, m, v);
            emit("sparse VFE", "RBF", backend_name(bk),
                 N_train, M,
                 evaluate(m, v, y_test_clean, y_test, hp.noise_variance));
        }
    }

    // CPU ↔ Metal numerical-agreement cross-check (independent of accuracy metrics).
    if (metal_ok) {
        std::printf("\n# CPU vs Metal prediction agreement (max |Δmean|, max |Δvar|):\n");
        auto agreement = [&](const char* label, KernelType kt, Solver s, int M_sparse) {
            Tensor m_cpu, v_cpu, m_metal, v_metal;
            if (M_sparse > 0) {
                GPSparseHyperparams hp;
                hp.kernel = kt;
                hp.noise_variance = noise * noise;
                GPSparse gc(hp, Backend::CPU), gm(hp, Backend::Metal);
                gc.fit(X_train, y_train, M_sparse);
                gm.fit(X_train, y_train, M_sparse);
                gc.predict(X_test, m_cpu, v_cpu);
                gm.predict(X_test, m_metal, v_metal);
            } else {
                GPHyperparams hp;
                hp.kernel = kt;
                hp.noise_variance = noise * noise;
                GPExact gc(hp, Backend::CPU, s), gm(hp, Backend::Metal, s);
                gc.fit(X_train, y_train);
                gm.fit(X_train, y_train);
                gc.predict(X_test, m_cpu, v_cpu);
                gm.predict(X_test, m_metal, v_metal);
            }
            float dm = 0.0f, dv = 0.0f;
            for (std::size_t i = 0; i < m_cpu.size(); ++i) {
                dm = std::max(dm, std::fabs(m_cpu.data()[i] - m_metal.data()[i]));
                dv = std::max(dv, std::fabs(v_cpu.data()[i] - v_metal.data()[i]));
            }
            std::printf("  %-32s  Δmean<=%.2e  Δvar<=%.2e\n", label, dm, dv);
        };
        agreement("exact Cholesky / RBF",      KernelType::RBF,      Solver::Cholesky, 0);
        agreement("exact Cholesky / Matern52", KernelType::Matern52, Solver::Cholesky, 0);
        agreement("sparse VFE / RBF / M=50",   KernelType::RBF,      Solver::Cholesky, 50);
    }

    return 0;
}
