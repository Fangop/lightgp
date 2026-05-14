#include "test_utils.h"

#include <cmath>

#include "../core/backend.h"
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

void run_gp_backend_tests() {
    std::printf("[gp_backend] starting...\n");

#ifdef LIGHTGP_HAS_METAL
    if (!MetalContext::instance().available()) {
        std::fprintf(stderr, "[gp_backend] SKIP — Metal unavailable: %s\n",
                     MetalContext::instance().error().c_str());
        return;
    }
#else
    std::printf("[gp_backend] SKIP — built without LIGHTGP_HAS_METAL\n");
    return;
#endif

    const int N = 40;
    Tensor X = make_grid(N, 0.0f, 2.0f * kPi);
    Tensor y(N, 1);
    for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));

    GPHyperparams hp;
    hp.length_scale = 1.3f;
    hp.signal_variance = 0.8f;
    hp.noise_variance = 1e-3f;

    GPExact gp_cpu(hp, Backend::CPU);
    GPExact gp_metal(hp, Backend::Metal);
    LIGHTGP_CHECK(gp_cpu.fit(X, y));
    LIGHTGP_CHECK(gp_metal.fit(X, y));

    // log marginal likelihood agreement (uses Cholesky path on top of kernel matrix).
    const float ll_cpu = gp_cpu.log_marginal_likelihood();
    const float ll_metal = gp_metal.log_marginal_likelihood();
    LIGHTGP_CHECK_NEAR(ll_cpu, ll_metal, 1e-3f);

    // Predict on a dense grid; means and variances must match end-to-end.
    Tensor X_test = make_grid(2 * N - 1, -1.0f, 2.0f * kPi + 1.0f);
    Tensor m_cpu, v_cpu, m_metal, v_metal;
    LIGHTGP_CHECK(gp_cpu.predict(X_test, m_cpu, v_cpu));
    LIGHTGP_CHECK(gp_metal.predict(X_test, m_metal, v_metal));
    LIGHTGP_CHECK(m_cpu.rows() == m_metal.rows() && m_cpu.cols() == m_metal.cols());
    for (std::size_t i = 0; i < m_cpu.size(); ++i) {
        LIGHTGP_CHECK_NEAR(m_cpu.data()[i], m_metal.data()[i], 1e-4f);
        LIGHTGP_CHECK_NEAR(v_cpu.data()[i], v_metal.data()[i], 1e-4f);
    }

    // Gradients: kernel-construction backend shouldn't affect them either.
    float dl_cpu, ds_cpu, dn_cpu;
    float dl_m, ds_m, dn_m;
    gp_cpu.log_marginal_likelihood_grads(dl_cpu, ds_cpu, dn_cpu);
    gp_metal.log_marginal_likelihood_grads(dl_m, ds_m, dn_m);
    LIGHTGP_CHECK_NEAR(dl_cpu, dl_m, 1e-3f);
    LIGHTGP_CHECK_NEAR(ds_cpu, ds_m, 1e-3f);
    LIGHTGP_CHECK_NEAR(dn_cpu, dn_m, 1e-3f);

    // Higher-dim case to exercise the float4 dispatch path through GPExact.
    Tensor X_hi = Tensor::randn(48, 8, 123);
    Tensor y_hi(48, 1);
    for (int i = 0; i < 48; ++i) {
        float s = 0.0f;
        for (std::size_t d = 0; d < X_hi.cols(); ++d) s += X_hi(i, d);
        y_hi(i, 0) = std::sin(s);
    }
    GPExact gp_hi_cpu(hp, Backend::CPU);
    GPExact gp_hi_metal(hp, Backend::Metal);
    LIGHTGP_CHECK(gp_hi_cpu.fit(X_hi, y_hi));
    LIGHTGP_CHECK(gp_hi_metal.fit(X_hi, y_hi));
    Tensor X_hi_test = Tensor::randn(20, 8, 456);
    Tensor mh_c, vh_c, mh_m, vh_m;
    LIGHTGP_CHECK(gp_hi_cpu.predict(X_hi_test, mh_c, vh_c));
    LIGHTGP_CHECK(gp_hi_metal.predict(X_hi_test, mh_m, vh_m));
    for (std::size_t i = 0; i < mh_c.size(); ++i) {
        LIGHTGP_CHECK_NEAR(mh_c.data()[i], mh_m.data()[i], 1e-4f);
        LIGHTGP_CHECK_NEAR(vh_c.data()[i], vh_m.data()[i], 1e-4f);
    }
}

}  // namespace lightgp
