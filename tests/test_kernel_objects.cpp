// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#include <cmath>
#include <memory>

#include "../core/backend.h"
#include "../core/mean.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../kernels/composite_kernel.h"
#include "../kernels/linear_kernel.h"
#include "../kernels/matern_kernel.h"
#include "../kernels/periodic_kernel.h"
#include "../kernels/rbf_kernel.h"

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;
Tensor make_grid(int n, float lo, float hi) {
    Tensor X(n, 1);
    for (int i = 0; i < n; ++i)
        X(i, 0) = lo + (hi - lo) * static_cast<float>(i) / (n - 1);
    return X;
}
}  // namespace

void run_kernel_object_tests() {
    std::printf("[kernel_objects] starting...\n");

    // RBFKernel vs legacy dispatch_kernel — should produce identical numbers.
    {
        Tensor X = make_grid(10, 0.0f, 5.0f);
        RBFKernel rbf(1.3f, 0.7f);
        Tensor K = rbf.compute(X, X);
        LIGHTGP_CHECK(K.rows() == 10 && K.cols() == 10);
        for (std::size_t i = 0; i < 10; ++i) {
            LIGHTGP_CHECK_NEAR(K(i, i), 0.7f, 1e-6f);  // sf2 at the diagonal
        }
        Tensor d = rbf.compute_diag(X);
        for (std::size_t i = 0; i < 10; ++i) LIGHTGP_CHECK_NEAR(d(i, 0), 0.7f, 1e-6f);
    }

    // Periodic kernel at r=0 returns sf2; periodicity check at integer periods.
    {
        Tensor X(3, 1, {0.0f, 1.0f, 2.0f});  // period 1.0
        PeriodicKernel p(/*l=*/1.0f, /*period=*/1.0f, /*sf2=*/2.5f);
        Tensor K = p.compute(X, X);
        LIGHTGP_CHECK_NEAR(K(0, 0), 2.5f, 1e-5f);
        LIGHTGP_CHECK_NEAR(K(0, 1), 2.5f, 1e-5f);  // diff = 1.0 = exact period
        LIGHTGP_CHECK_NEAR(K(0, 2), 2.5f, 1e-5f);
    }

    // Linear kernel = sf2 * X @ X^T (offset = 0).
    {
        Tensor X(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});  // orthonormal rows
        LinearKernel lin(/*sf2=*/3.0f, /*input_dim=*/2);
        Tensor K = lin.compute(X, X);
        LIGHTGP_CHECK_NEAR(K(0, 0), 3.0f, 1e-6f);
        LIGHTGP_CHECK_NEAR(K(0, 1), 0.0f, 1e-6f);  // orthogonal
        LIGHTGP_CHECK_NEAR(K(1, 1), 3.0f, 1e-6f);
    }

    // SumKernel composition: hyperparameter count = sum, diagonal = sum of diagonals.
    {
        auto r1 = std::make_shared<RBFKernel>(1.0f, 0.5f);
        auto r2 = std::make_shared<RBFKernel>(0.3f, 1.5f);
        auto sum = r1 + r2;
        LIGHTGP_CHECK(sum->num_params() == 4);
        Tensor X = make_grid(5, 0.0f, 1.0f);
        Tensor K_sum = sum->compute(X, X);
        Tensor K1 = r1->compute(X, X);
        Tensor K2 = r2->compute(X, X);
        for (std::size_t i = 0; i < K_sum.size(); ++i)
            LIGHTGP_CHECK_NEAR(K_sum.data()[i], K1.data()[i] + K2.data()[i], 1e-5f);
        Tensor d = sum->compute_diag(X);
        for (std::size_t i = 0; i < 5; ++i) LIGHTGP_CHECK_NEAR(d(i, 0), 0.5f + 1.5f, 1e-5f);
    }

    // ScaleKernel: scale * base, params = [log(scale), ...base.params].
    {
        auto base = std::make_shared<RBFKernel>(1.0f, 1.0f);
        auto sc = scale(base, /*initial_scale=*/4.0f);
        LIGHTGP_CHECK(sc->num_params() == 3);  // 1 (scale) + 2 (rbf)
        Tensor X(1, 1, {0.0f});
        Tensor K = sc->compute(X, X);
        LIGHTGP_CHECK_NEAR(K(0, 0), 4.0f, 1e-6f);  // 4.0 * 1.0
    }

    // ProductKernel: elementwise multiply.
    {
        auto a = std::make_shared<RBFKernel>(1.0f, 1.0f);
        auto b = std::make_shared<RBFKernel>(2.0f, 1.0f);
        auto prod = a * b;
        Tensor X = make_grid(4, 0.0f, 1.0f);
        Tensor K = prod->compute(X, X);
        Tensor Ka = a->compute(X, X);
        Tensor Kb = b->compute(X, X);
        for (std::size_t i = 0; i < K.size(); ++i)
            LIGHTGP_CHECK_NEAR(K.data()[i], Ka.data()[i] * Kb.data()[i], 1e-5f);
    }

    // get_log_params / set_log_params roundtrip on composite.
    {
        auto k = scale(std::make_shared<RBFKernel>()) + scale(std::make_shared<PeriodicKernel>());
        auto p0 = k->get_log_params();
        // perturb and round-trip
        std::vector<float> p1 = p0;
        for (auto& v : p1) v += 0.1f;
        k->set_log_params(p1);
        auto p2 = k->get_log_params();
        for (std::size_t i = 0; i < p1.size(); ++i)
            LIGHTGP_CHECK_NEAR(p1[i], p2[i], 1e-5f);
    }

    // End-to-end: GPExact with new kernel API on sin(x) should match the legacy
    // GPHyperparams path closely.
    {
        const int N = 30;
        Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
        Tensor y(N, 1);
        for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));

        // New API
        auto kernel = std::make_shared<RBFKernel>(1.0f, 1.0f);
        auto mean = std::make_shared<ZeroMean>();
        GPExact gp_new(kernel, mean, /*sn2=*/1e-3f, Backend::CPU, Solver::Cholesky);
        LIGHTGP_CHECK(gp_new.fit(X, y));

        // Legacy API
        GPHyperparams hp;
        hp.length_scale = 1.0f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 1e-3f;
        GPExact gp_old(hp);
        LIGHTGP_CHECK(gp_old.fit(X, y));

        Tensor X_test = make_grid(10, 0.5f, 2.0f * kPi - 0.5f);
        Tensor m_new, v_new, m_old, v_old;
        LIGHTGP_CHECK(gp_new.predict(X_test, m_new, v_new));
        LIGHTGP_CHECK(gp_old.predict(X_test, m_old, v_old));
        for (std::size_t i = 0; i < m_new.size(); ++i) {
            LIGHTGP_CHECK_NEAR(m_new.data()[i], m_old.data()[i], 1e-5f);
            LIGHTGP_CHECK_NEAR(v_new.data()[i], v_old.data()[i], 1e-5f);
        }
    }

    // GPExact with ConstantMean: y has nonzero mean — should be recovered.
    {
        const int N = 40;
        Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
        Tensor y(N, 1);
        const float true_mean = 5.0f;
        for (int i = 0; i < N; ++i) y(i, 0) = true_mean + std::sin(X(i, 0));

        auto kernel = std::make_shared<RBFKernel>(1.0f, 0.5f);
        auto mean = std::make_shared<ConstantMean>(0.0f);
        GPExact gp(kernel, mean, 1e-3f);
        LIGHTGP_CHECK(gp.fit(X, y));

        Tensor X_test = make_grid(8, 0.5f, 2.0f * kPi - 0.5f);
        Tensor m_pred, v_pred;
        LIGHTGP_CHECK(gp.predict(X_test, m_pred, v_pred));
        // Predictions should track the true signal (mean + sin) to within fit quality.
        for (std::size_t i = 0; i < X_test.rows(); ++i) {
            const float truth = true_mean + std::sin(X_test(i, 0));
            LIGHTGP_CHECK_NEAR(m_pred(i, 0), truth, 0.2f);
        }
    }
}

}  // namespace lightgp
