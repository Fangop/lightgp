// bench_datasets — accuracy + timing on three GP benchmark datasets.
//
// Datasets are mathematical stand-ins that match the structural characteristics of
// the well-known originals (Silverman mcycle, Mauna Loa CO2, GPML kin40k). Real CSVs
// can be wired in by editing data/datasets.cpp.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../core/backend.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../data/datasets.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"
#include "bench_common.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Metrics {
    float rmse;
    float mean_nll;
    float coverage_2sigma;
    float fit_ms;
    float predict_ms;
};

using clock_t = std::chrono::steady_clock;
using tp_t = clock_t::time_point;
inline tp_t now_tp() { return clock_t::now(); }
inline double ms_since(tp_t t0) {
    return std::chrono::duration<double, std::milli>(clock_t::now() - t0).count();
}

Metrics evaluate(const lightgp::Tensor& mean, const lightgp::Tensor& var,
                 const lightgp::Tensor& y, float y_std, float noise_var_std,
                 float fit_ms, float predict_ms) {
    const std::size_t N = y.rows();
    float sse = 0.0f;
    double nll = 0.0;
    int within = 0;
    // Predicted observation variance (in standardized units) = var_f* + sn2.
    for (std::size_t i = 0; i < N; ++i) {
        const float err = mean(i, 0) - y(i, 0);
        sse += err * err;
        const float v = std::max(var(i, 0) + noise_var_std, 1e-6f);
        nll += 0.5 * (std::log(2.0 * kPi * v) + (err * err) / v);
        if (std::fabs(err) <= 2.0f * std::sqrt(v)) ++within;
    }
    // RMSE in physical units (multiply by y_std since y is standardized).
    Metrics m;
    m.rmse = std::sqrt(sse / static_cast<float>(N)) * y_std;
    m.mean_nll = static_cast<float>(nll / N);
    m.coverage_2sigma = static_cast<float>(within) / static_cast<float>(N);
    m.fit_ms = fit_ms;
    m.predict_ms = predict_ms;
    return m;
}

void print_row(const char* dataset, const char* method, const char* kernel,
               const char* backend, std::size_t N_train, std::size_t M,
               const Metrics& m) {
    std::printf("%-12s  %-22s  %-10s  %-7s  N=%-5zu  M=%-4s  RMSE=%-9.3f  NLL=%+8.3f  cov=%5.1f%%  fit=%-8.1f  pred=%-7.1f\n",
                dataset, method, kernel, backend, N_train,
                (M > 0 ? std::to_string(M).c_str() : "-"),
                m.rmse, m.mean_nll, 100.0f * m.coverage_2sigma,
                m.fit_ms, m.predict_ms);
}

void run_dataset(const lightgp::data::Dataset& ds, bool include_exact, bool metal_ok) {
    using namespace lightgp;
    const std::size_t N_train = ds.X_train.rows();
    const float sn2 = 0.05f;  // standardized; ~0.22 std fraction
    const float noise_var_std = sn2;

    auto run_exact = [&](Backend bk, KernelType kt, const char* kname) {
        GPHyperparams hp;
        hp.kernel = kt;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = sn2;
        GPExact gp(hp, bk, Solver::Cholesky);
        const auto t0 = now_tp();
        gp.fit(ds.X_train, ds.y_train);
        gp.optimize_hyperparameters(/*steps=*/30, /*lr=*/0.05f, /*verbose=*/false);
        const double fit_ms = ms_since(t0);
        Tensor mean, var;
        const auto t1 = now_tp();
        gp.predict(ds.X_test, mean, var);
        const double pred_ms = ms_since(t1);
        const Metrics m = evaluate(mean, var, ds.y_test, ds.y_std, noise_var_std,
                                   fit_ms, pred_ms);
        const char* bk_name = (bk == Backend::CPU) ? "cpu"
                            : (bk == Backend::Metal) ? "metal" : "auto";
        print_row(ds.name, "exact_chol", kname, bk_name, N_train, 0, m);
    };

    auto run_sparse = [&](Backend bk, KernelType kt, std::size_t M, const char* kname) {
        GPSparseHyperparams hp;
        hp.kernel = kt;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = sn2;
        GPSparse gp(hp, bk);
        const auto t0 = now_tp();
        gp.fit(ds.X_train, ds.y_train, M);
        const double fit_ms = ms_since(t0);
        Tensor mean, var;
        const auto t1 = now_tp();
        gp.predict(ds.X_test, mean, var);
        const double pred_ms = ms_since(t1);
        const Metrics m = evaluate(mean, var, ds.y_test, ds.y_std, noise_var_std,
                                   fit_ms, pred_ms);
        const char* bk_name = (bk == Backend::CPU) ? "cpu"
                            : (bk == Backend::Metal) ? "metal" : "auto";
        print_row(ds.name, "sparse_vfe", kname, bk_name, N_train, M, m);
    };

    auto run_cg = [&](Backend bk, KernelType kt, const char* kname) {
        GPHyperparams hp;
        hp.kernel = kt;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = sn2;
        GPExact gp(hp, bk, Solver::CG);
        const auto t0 = now_tp();
        gp.fit(ds.X_train, ds.y_train);
        const double fit_ms = ms_since(t0);
        Tensor mean, var;
        const auto t1 = now_tp();
        gp.predict(ds.X_test, mean, var);
        const double pred_ms = ms_since(t1);
        const Metrics m = evaluate(mean, var, ds.y_test, ds.y_std, noise_var_std,
                                   fit_ms, pred_ms);
        const char* bk_name = (bk == Backend::CPU) ? "cpu"
                            : (bk == Backend::Metal) ? "metal" : "auto";
        print_row(ds.name, "cg_hutchinson", kname, bk_name, N_train, 0, m);
    };

    // Exact Cholesky: skip for kin40k (N=32k → 32k×32k Cholesky is hours).
    if (include_exact) {
        run_exact(Backend::CPU, KernelType::RBF, "RBF");
        run_exact(Backend::CPU, KernelType::Matern52, "Matern52");
        if (metal_ok) run_exact(Backend::Metal, KernelType::RBF, "RBF");
    }
    // Sparse — skip M values larger than N_train (motorcycle is small).
    if (N_train >= 50)  run_sparse(Backend::CPU,   KernelType::RBF, 50,  "RBF");
    if (N_train >= 200) run_sparse(Backend::CPU,   KernelType::RBF, 200, "RBF");
    if (metal_ok && N_train >= 200)
        run_sparse(Backend::Metal, KernelType::RBF, 200, "RBF");
    // CG: useful at moderate-large N; skip for the tiny motorcycle case.
    if (N_train >= 500) {
        run_cg(Backend::CPU, KernelType::RBF, "RBF");
        if (metal_ok) run_cg(Backend::Metal, KernelType::RBF, "RBF");
    }
}

}  // namespace

int main() {
    using namespace lightgp;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef LIGHTGP_HAS_METAL
    const bool metal_ok = MetalContext::instance().available();
#else
    const bool metal_ok = false;
#endif

    std::printf("# bench_datasets — accuracy + timing on three GP benchmark stand-ins\n");
    std::printf("# motorcycle: heteroscedastic 1D (N=133); mauna_loa: trend+seasonal 1D (N=780);\n");
    std::printf("# kin40k: 8D nonlinear (N=40000) — exact Cholesky skipped.\n");
    std::printf("# RMSE in physical units; NLL on standardized noisy targets; cov = %% within ±2σ.\n");
    std::printf("# %-10s  %-22s  %-10s  %-7s  %-7s  %-6s  %-15s  %-13s  %-10s  %-10s  %-7s\n",
                "dataset", "method", "kernel", "backend", "N_train", "M",
                "RMSE", "mean NLL", "cov", "fit (ms)", "pred (ms)");

    auto mc = data::make_motorcycle();
    auto ml = data::make_mauna_loa();
    run_dataset(mc, /*include_exact=*/true, metal_ok);
    run_dataset(ml, /*include_exact=*/true, metal_ok);

    auto k40 = data::make_kin40k();
    run_dataset(k40, /*include_exact=*/false, metal_ok);

    return 0;
}
