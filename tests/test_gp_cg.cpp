#include "test_utils.h"

#include <cmath>

#include "../core/backend.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#endif

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

void run_gp_cg_tests() {
    std::printf("[gp_cg] starting...\n");

    const int N = 30;
    Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
    Tensor y(N, 1);
    for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));

    GPHyperparams hp;
    hp.length_scale = 1.0f;
    hp.signal_variance = 1.0f;
    hp.noise_variance = 1e-2f;

    GPExact gp_chol(hp, Backend::CPU, Solver::Cholesky);
    GPExact gp_cg(hp, Backend::CPU, Solver::CG);
    LIGHTGP_CHECK(gp_chol.fit(X, y));
    LIGHTGP_CHECK(gp_cg.fit(X, y));
    LIGHTGP_CHECK(gp_cg.fitted());
    LIGHTGP_CHECK(gp_cg.solver() == Solver::CG);

    // Mean predictions on a small test grid: CG should match Cholesky to within CG tolerance.
    Tensor X_test = make_grid(8, 0.5f, 2.0f * kPi - 0.5f);
    Tensor m_chol, v_chol, m_cg, v_cg;
    LIGHTGP_CHECK(gp_chol.predict(X_test, m_chol, v_chol));
    LIGHTGP_CHECK(gp_cg.predict(X_test, m_cg, v_cg));
    // Mean is solved by CG to the requested tolerance — tight match.
    // Variance is a stochastic Hutchinson estimator (30 Rademacher probes); its
    // absolute error scales with the Frobenius norm of K_star^T K_y^{-1} K_star,
    // not with the per-point variance, so interpolation points (tiny true variance)
    // can show large *relative* error while extrapolation points are tight.
    // Tolerance below absorbs worst-case probe noise on this small problem AND the
    // cross-platform drift in the probe-by-probe matmul rounding (Accelerate vs
    // OpenBLAS vs reference triple-loop differ by ~1 ULP per add, which shifts the
    // 30-sample mean by up to ~0.5 on the noisy tails).
    for (std::size_t i = 0; i < m_chol.size(); ++i) {
        LIGHTGP_CHECK_NEAR(m_cg.data()[i], m_chol.data()[i], 5e-3f);
        LIGHTGP_CHECK_NEAR(v_cg.data()[i], v_chol.data()[i], 0.6f);
    }

    // log marginal likelihood: SLQ estimate vs Cholesky exact. 15% absorbs SLQ
    // stochastic noise + cross-platform probe drift (Mac/Linux Hutchinson trajectories
    // give 7-11% rel error on this N=30 problem with fixed seed).
    const float ll_chol = gp_chol.log_marginal_likelihood();
    const float ll_cg = gp_cg.log_marginal_likelihood();
    LIGHTGP_CHECK(std::isfinite(ll_cg));
    LIGHTGP_CHECK(std::fabs(ll_cg - ll_chol) / std::fabs(ll_chol) < 0.15f);

    // Gradients: Hutchinson estimator vs exact Cholesky gradients. Loose tolerance since it's stochastic.
    float dl_c, ds_c, dn_c;
    float dl_g, ds_g, dn_g;
    gp_chol.log_marginal_likelihood_grads(dl_c, ds_c, dn_c);
    gp_cg.log_marginal_likelihood_grads(dl_g, ds_g, dn_g);
    const float gscale = std::fabs(dl_c) + std::fabs(ds_c) + std::fabs(dn_c) + 1.0f;
    LIGHTGP_CHECK(std::fabs(dl_g - dl_c) < 0.3f * gscale);
    LIGHTGP_CHECK(std::fabs(ds_g - ds_c) < 0.3f * gscale);
    LIGHTGP_CHECK(std::fabs(dn_g - dn_c) < 0.3f * gscale);

    // Hyperparameter optimization in CG mode should still improve log-marginal likelihood.
    GPExact gp_opt(hp, Backend::CPU, Solver::CG);
    LIGHTGP_CHECK(gp_opt.fit(X, y));
    const float ll_before = gp_opt.log_marginal_likelihood();
    LIGHTGP_CHECK(gp_opt.optimize_hyperparameters(/*num_steps=*/30, /*lr=*/0.05f, /*verbose=*/false));
    const float ll_after = gp_opt.log_marginal_likelihood();
    // SLQ noise can pull this slightly either way; require no major regression.
    LIGHTGP_CHECK(ll_after >= ll_before - 0.5f);

#ifdef LIGHTGP_HAS_METAL
    // Matrix-free Metal CG path: same answer as explicit-matrix CPU CG, no K_y stored.
    if (MetalContext::instance().available()) {
        GPExact gp_metal_cg(hp, Backend::Metal, Solver::CG);
        LIGHTGP_CHECK(gp_metal_cg.fit(X, y));

        Tensor m_metal, v_metal;
        LIGHTGP_CHECK(gp_metal_cg.predict(X_test, m_metal, v_metal));
        for (std::size_t i = 0; i < m_cg.size(); ++i) {
            LIGHTGP_CHECK_NEAR(m_metal.data()[i], m_cg.data()[i], 5e-3f);
            // Same probe seed on CPU and Metal CG → identical probe vectors;
            // only float-32 matvec rounding distinguishes the two paths.
            LIGHTGP_CHECK_NEAR(v_metal.data()[i], v_cg.data()[i], 5e-2f);
        }
        const float ll_metal_cg = gp_metal_cg.log_marginal_likelihood();
        LIGHTGP_CHECK(std::isfinite(ll_metal_cg));
        LIGHTGP_CHECK(std::fabs(ll_metal_cg - ll_chol) / std::fabs(ll_chol) < 0.15f);
    }
#endif
}

}  // namespace lightgp
