#include "test_utils.h"

#include <cmath>

#include "../core/tensor.h"
#include "../kernels/cpu/rbf_cpu.h"

namespace lightgp {

void run_rbf_cpu_tests() {
    std::printf("[rbf_cpu] starting...\n");

    // 1D: x1=[0], x2=[1], l=1, sf2=1.  K(0,0)=K(1,1)=1, K(0,1)=exp(-0.5).
    Tensor X(2, 1, {0.0f, 1.0f});
    Tensor K = rbf_kernel_cpu(X, X, 1.0f, 1.0f);
    LIGHTGP_CHECK(K.rows() == 2 && K.cols() == 2);
    LIGHTGP_CHECK_NEAR(K(0, 0), 1.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(K(1, 1), 1.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(K(0, 1), std::exp(-0.5f), 1e-6f);
    LIGHTGP_CHECK_NEAR(K(1, 0), std::exp(-0.5f), 1e-6f);

    // signal_variance scales K linearly.
    Tensor K_sf = rbf_kernel_cpu(X, X, 1.0f, 4.0f);
    LIGHTGP_CHECK_NEAR(K_sf(0, 0), 4.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(K_sf(0, 1), 4.0f * std::exp(-0.5f), 1e-5f);

    // length_scale = 2: K(0,1) = exp(-0.5 / 4) = exp(-0.125).
    Tensor K_l = rbf_kernel_cpu(X, X, 2.0f, 1.0f);
    LIGHTGP_CHECK_NEAR(K_l(0, 1), std::exp(-0.125f), 1e-6f);

    // 2D input: X = [[0,0],[1,1]], squared dist = 2; l=1, sf2=1.
    Tensor X2d(2, 2, {0.0f, 0.0f, 1.0f, 1.0f});
    Tensor K_2d = rbf_kernel_cpu(X2d, X2d, 1.0f, 1.0f);
    LIGHTGP_CHECK_NEAR(K_2d(0, 1), std::exp(-1.0f), 1e-6f);

    // Cross kernel: X1 (2x1) x X2 (3x1) -> 2x3 with correct shape.
    Tensor X2(3, 1, {0.0f, 0.5f, 1.0f});
    Tensor Kcross = rbf_kernel_cpu(X, X2, 1.0f, 1.0f);
    LIGHTGP_CHECK(Kcross.rows() == 2 && Kcross.cols() == 3);
    LIGHTGP_CHECK_NEAR(Kcross(0, 0), 1.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(Kcross(0, 1), std::exp(-0.125f), 1e-6f);
    LIGHTGP_CHECK_NEAR(Kcross(0, 2), std::exp(-0.5f), 1e-6f);
    LIGHTGP_CHECK_NEAR(Kcross(1, 2), 1.0f, 1e-6f);

    // Analytical gradients: dK/d(log sf2) = K; dK/d(log l) = K * r^2 / l^2.
    Tensor grad_l, grad_sf2;
    rbf_kernel_gradients_cpu(X, X, 1.0f, 1.0f, grad_l, grad_sf2);
    LIGHTGP_CHECK_NEAR(grad_sf2(0, 1), std::exp(-0.5f), 1e-6f);
    LIGHTGP_CHECK_NEAR(grad_sf2(0, 0), 1.0f, 1e-6f);
    LIGHTGP_CHECK_NEAR(grad_l(0, 1), std::exp(-0.5f), 1e-6f);
    LIGHTGP_CHECK_NEAR(grad_l(0, 0), 0.0f, 1e-6f);

    // Finite-difference check for d(log l) at l=1.3, sf2=1.7, r^2=2.
    const float l = 1.3f, sf2 = 1.7f;
    Tensor Y(2, 1, {0.0f, std::sqrt(2.0f)});
    Tensor gl, gsf;
    rbf_kernel_gradients_cpu(Y, Y, l, sf2, gl, gsf);
    const float eps = 1e-3f;
    Tensor Kplus = rbf_kernel_cpu(Y, Y, l * std::exp(eps), sf2);
    Tensor Kminus = rbf_kernel_cpu(Y, Y, l * std::exp(-eps), sf2);
    const float fd = (Kplus(0, 1) - Kminus(0, 1)) / (2.0f * eps);
    LIGHTGP_CHECK_NEAR(gl(0, 1), fd, 1e-3f);

    Tensor Kp_sf = rbf_kernel_cpu(Y, Y, l, sf2 * std::exp(eps));
    Tensor Km_sf = rbf_kernel_cpu(Y, Y, l, sf2 * std::exp(-eps));
    const float fd_sf = (Kp_sf(0, 1) - Km_sf(0, 1)) / (2.0f * eps);
    LIGHTGP_CHECK_NEAR(gsf(0, 1), fd_sf, 1e-3f);
}

}  // namespace lightgp
