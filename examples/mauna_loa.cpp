// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

// Mauna Loa CO2 — the canonical GP-composition demonstration.
//
// The signal has three structures: a multi-decade rising trend, an annual
// seasonal cycle, and medium-term irregularities. A single RBF cannot capture
// all three; instead we compose:
//   k(x, x') = scale(RBF_long)(x, x') + scale(Periodic)(x, x') + scale(RBF_short)(x, x')
//
// and use a Linear mean function for the long-term linear trend.

#include <cmath>
#include <cstdio>
#include <memory>

#include "../core/backend.h"
#include "../core/mean.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../data/datasets.h"
#include "../inference/gp_exact.h"
#include "../kernels/composite_kernel.h"
#include "../kernels/periodic_kernel.h"
#include "../kernels/rbf_kernel.h"

int main() {
    using namespace lightgp;

    // Mauna Loa stand-in: trend + seasonal 1D, monthly samples 1958..2022 (N=780).
    auto ds = data::make_mauna_loa();
    std::printf("# Mauna Loa CO2 GP with composed kernel\n");
    std::printf("#   N_train = %zu, N_test = %zu\n", ds.X_train.rows(), ds.X_test.rows());

    // Compose: scale(RBF_long) + scale(Periodic) + scale(RBF_short).
    // Initial hyperparameters chosen to span the relevant scales: long ~10 yr,
    // periodic ~1 yr, short ~0.5 yr.
    auto rbf_long  = scale(std::make_shared<RBFKernel>(/*length_scale=*/10.0f, /*sf2=*/1.0f));
    auto periodic  = scale(std::make_shared<PeriodicKernel>(/*l=*/1.0f, /*p=*/1.0f, /*sf2=*/1.0f));
    auto rbf_short = scale(std::make_shared<RBFKernel>(/*length_scale=*/0.5f, /*sf2=*/1.0f));
    std::shared_ptr<Kernel> kernel = rbf_long + periodic + rbf_short;

    // Linear mean: y ≈ w * t + b for the rising trend.
    auto mean = std::make_shared<LinearMean>(/*input_dim=*/1);

    std::printf("#   kernel: %s\n", kernel->name().c_str());
    std::printf("#   mean:   %s\n", mean->name().c_str());
    std::printf("#   total log-params (kernel): %d\n", kernel->num_params());

    GPExact gp(kernel, mean, /*noise_variance=*/0.05f, Backend::CPU, Solver::Cholesky);
    if (!gp.fit(ds.X_train, ds.y_train)) {
        std::fprintf(stderr, "fit failed\n");
        return 1;
    }
    std::printf("# initial log_marginal_likelihood = %.3f\n",
                gp.log_marginal_likelihood());

    Tensor mean_pred, var_pred;
    gp.predict(ds.X_test, mean_pred, var_pred);

    double sse = 0.0;
    for (std::size_t i = 0; i < ds.y_test.rows(); ++i) {
        const float err = mean_pred(i, 0) - ds.y_test(i, 0);
        sse += err * err;
    }
    const double rmse_std = std::sqrt(sse / ds.y_test.rows());
    std::printf("# test RMSE (standardized) = %.4f\n", rmse_std);
    std::printf("# test RMSE (physical)     = %.4f\n", rmse_std * ds.y_std);

    // Print learned hyperparameters in human-readable form.
    auto params = kernel->get_log_params();
    std::printf("# learned log-hyperparameters (per-kernel-block):\n");
    for (std::size_t i = 0; i < params.size(); ++i) {
        std::printf("#   theta[%zu] = %+.4f  (exp = %.4f)\n",
                    i, params[i], std::exp(params[i]));
    }
    return 0;
}
