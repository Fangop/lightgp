#include "test_utils.h"

#include <cmath>

#include "../core/backend.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/gp_sparse.h"

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;

Tensor make_grid(int n, float lo, float hi) {
    Tensor X(n, 1);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(n - 1);
    }
    return X;
}
}  // namespace

void run_gp_sparse_tests() {
    std::printf("[gp_sparse] starting...\n");

    // Sin(x) regression: sparse GP with M=10 inducing should track exact GP roughly.
    const int N = 200;
    Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
    Tensor y(N, 1);
    for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));

    GPHyperparams hp_exact;
    hp_exact.length_scale = 1.0f;
    hp_exact.signal_variance = 1.0f;
    hp_exact.noise_variance = 1e-3f;

    GPSparseHyperparams hp_sp;
    hp_sp.length_scale = 1.0f;
    hp_sp.signal_variance = 1.0f;
    hp_sp.noise_variance = 1e-3f;

    GPExact gp_exact(hp_exact);
    GPSparse gp_sp(hp_sp);
    LIGHTGP_CHECK(gp_exact.fit(X, y));
    LIGHTGP_CHECK(gp_sp.fit(X, y, /*num_inducing=*/10));
    LIGHTGP_CHECK(gp_sp.fitted());
    LIGHTGP_CHECK(gp_sp.inducing_points().rows() == 10);
    LIGHTGP_CHECK(gp_sp.inducing_points().cols() == 1);

    // Predictions on a dense grid: sparse mean should track exact mean qualitatively.
    Tensor X_test = make_grid(40, 0.5f, 2.0f * kPi - 0.5f);
    Tensor m_exact, v_exact, m_sp, v_sp;
    LIGHTGP_CHECK(gp_exact.predict(X_test, m_exact, v_exact));
    LIGHTGP_CHECK(gp_sp.predict(X_test, m_sp, v_sp));
    LIGHTGP_CHECK(m_sp.rows() == m_exact.rows() && m_sp.cols() == m_exact.cols());
    for (std::size_t i = 0; i < m_exact.size(); ++i) {
        // M=10 inducing points on N=200 sin(x): mean should be within ~0.15 of exact.
        LIGHTGP_CHECK_NEAR(m_sp.data()[i], m_exact.data()[i], 0.15f);
        LIGHTGP_CHECK(v_sp.data()[i] >= 0.0f);
    }

    // Sparse mean prediction at training points should be close to y.
    Tensor m_train, v_train;
    LIGHTGP_CHECK(gp_sp.predict(X, m_train, v_train));
    float max_abs_err = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float e = std::fabs(m_train.data()[i] - y.data()[i]);
        if (e > max_abs_err) max_abs_err = e;
    }
    LIGHTGP_CHECK(max_abs_err < 0.2f);

    // log marginal likelihood: should be finite, larger N usually → larger absolute value.
    const float ll = gp_sp.log_marginal_likelihood();
    LIGHTGP_CHECK(std::isfinite(ll));

    // More inducing points should improve mean fit.
    GPSparse gp_more(hp_sp);
    LIGHTGP_CHECK(gp_more.fit(X, y, /*num_inducing=*/30));
    Tensor m_more, v_more;
    LIGHTGP_CHECK(gp_more.predict(X, m_more, v_more));
    float max_err_more = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float e = std::fabs(m_more.data()[i] - y.data()[i]);
        if (e > max_err_more) max_err_more = e;
    }
    LIGHTGP_CHECK(max_err_more <= max_abs_err + 1e-3f);
}

}  // namespace lightgp
