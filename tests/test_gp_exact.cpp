// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#include <cmath>

#include "../core/tensor.h"
#include "../inference/gp_exact.h"

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;

Tensor make_x(int n, float lo, float hi) {
    Tensor X(n, 1);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(n - 1);
    }
    return X;
}
}  // namespace

void run_gp_exact_tests() {
    std::printf("[gp_exact] starting...\n");

    // y = sin(x) on [0, 2π], N=20.
    const int N = 20;
    Tensor X = make_x(N, 0.0f, 2.0f * kPi);
    Tensor y(N, 1);
    for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));

    GPHyperparams hp;
    hp.length_scale = 1.0f;
    hp.signal_variance = 1.0f;
    hp.noise_variance = 1e-3f;

    GPExact gp(hp);
    LIGHTGP_CHECK(gp.fit(X, y));
    LIGHTGP_CHECK(gp.fitted());

    // Predictions at training points should be close to training targets (low noise).
    Tensor mean, var;
    LIGHTGP_CHECK(gp.predict(X, mean, var));
    LIGHTGP_CHECK(mean.rows() == static_cast<std::size_t>(N) && mean.cols() == 1);
    for (int i = 0; i < N; ++i) {
        LIGHTGP_CHECK_NEAR(mean(i, 0), y(i, 0), 5e-2f);
        // Variance at training inputs should be small but non-negative.
        LIGHTGP_CHECK(var(i, 0) >= 0.0f);
        LIGHTGP_CHECK(var(i, 0) < 0.5f);
    }

    // Prediction at midpoints (interpolation) should be close to true sin.
    Tensor X_mid = make_x(2 * N - 1, 0.0f, 2.0f * kPi);  // includes original + midpoints
    Tensor mean_mid, var_mid;
    LIGHTGP_CHECK(gp.predict(X_mid, mean_mid, var_mid));
    for (std::size_t i = 0; i < X_mid.rows(); ++i) {
        const float truth = std::sin(X_mid(i, 0));
        LIGHTGP_CHECK_NEAR(mean_mid(i, 0), truth, 1e-1f);
    }

    // Variance grows when we extrapolate well outside training range.
    Tensor X_far(1, 1, {20.0f});
    Tensor mean_far, var_far;
    LIGHTGP_CHECK(gp.predict(X_far, mean_far, var_far));
    LIGHTGP_CHECK(var_far(0, 0) > 0.5f * hp.signal_variance);

    // log marginal likelihood is finite.
    const float ll0 = gp.log_marginal_likelihood();
    LIGHTGP_CHECK(std::isfinite(ll0));

    // Analytical gradients vs finite differences.
    {
        float dl, dsf2, dsn2;
        gp.log_marginal_likelihood_grads(dl, dsf2, dsn2);

        const float eps = 1e-3f;
        const GPHyperparams base = gp.hyperparams();

        auto ll_at = [&](float log_l, float log_sf2, float log_sn2) {
            GPHyperparams h;
            h.length_scale = std::exp(log_l);
            h.signal_variance = std::exp(log_sf2);
            h.noise_variance = std::exp(log_sn2);
            GPExact g(h);
            g.fit(X, y);
            return g.log_marginal_likelihood();
        };

        const float base_log_l = std::log(base.length_scale);
        const float base_log_sf2 = std::log(base.signal_variance);
        const float base_log_sn2 = std::log(base.noise_variance);

        const float fd_l = (ll_at(base_log_l + eps, base_log_sf2, base_log_sn2)
                           - ll_at(base_log_l - eps, base_log_sf2, base_log_sn2)) / (2.0f * eps);
        const float fd_sf2 = (ll_at(base_log_l, base_log_sf2 + eps, base_log_sn2)
                             - ll_at(base_log_l, base_log_sf2 - eps, base_log_sn2)) / (2.0f * eps);
        const float fd_sn2 = (ll_at(base_log_l, base_log_sf2, base_log_sn2 + eps)
                             - ll_at(base_log_l, base_log_sf2, base_log_sn2 - eps)) / (2.0f * eps);

        LIGHTGP_CHECK_NEAR(dl, fd_l, 5e-2f);
        LIGHTGP_CHECK_NEAR(dsf2, fd_sf2, 5e-2f);
        LIGHTGP_CHECK_NEAR(dsn2, fd_sn2, 5e-2f);
    }

    // Hyperparameter optimization should not decrease log marginal likelihood.
    GPExact gp2(hp);
    LIGHTGP_CHECK(gp2.fit(X, y));
    const float ll_before = gp2.log_marginal_likelihood();
    LIGHTGP_CHECK(gp2.optimize_hyperparameters(/*num_steps=*/40, /*lr=*/0.05f, /*verbose=*/false));
    const float ll_after = gp2.log_marginal_likelihood();
    LIGHTGP_CHECK(ll_after >= ll_before - 1e-3f);
}

}  // namespace lightgp
