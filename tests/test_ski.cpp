// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../core/backend.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../inference/gp_exact.h"
#include "../inference/ski.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../kernels/rbf_kernel.h"

#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#endif

#ifdef LIGHTGP_HAS_ACCELERATE
#include "../inference/ski_accel.h"
#endif

namespace lightgp {

namespace {

Tensor make_X1(int N, float lo, float hi, std::uint64_t seed = 0) {
    Tensor X(N, 1);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int i = 0; i < N; ++i) X(i, 0) = dist(rng);
    return X;
}

Tensor make_X2_grid(int Nx, int Ny, float lo, float hi) {
    Tensor X(Nx * Ny, 2);
    int k = 0;
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            X(k, 0) = lo + (hi - lo) * static_cast<float>(i) / (Nx - 1);
            X(k, 1) = lo + (hi - lo) * static_cast<float>(j) / (Ny - 1);
            ++k;
        }
    return X;
}

float relative_error(const Tensor& a, const Tensor& b) {
    float num = 0.0f, den = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = a.data()[i] - b.data()[i];
        num += diff * diff;
        den += b.data()[i] * b.data()[i];
    }
    return std::sqrt(num / std::max(den, 1e-30f));
}

}  // namespace

void run_ski_tests() {
    std::printf("[ski] starting...\n");

    // -- Grid construction (1D) -----------------------------------------------
    {
        Tensor X = make_X1(50, -1.5f, 3.0f, 0xA1u);
        SKIGrid g = SKIGrid::from_data(X, /*points_per_dim=*/64, /*padding=*/0.1f);
        LIGHTGP_CHECK(g.D == 1);
        LIGHTGP_CHECK(g.grid_sizes[0] == 64);
        LIGHTGP_CHECK(g.total_points == 64);
        // Padding gives ≥ 0.1 * (max-min) margin on each side of the empirical extent.
        float emp_lo = X(0, 0), emp_hi = X(0, 0);
        for (int i = 1; i < X.rows(); ++i) {
            emp_lo = std::min(emp_lo, X(i, 0));
            emp_hi = std::max(emp_hi, X(i, 0));
        }
        const float emp_range = emp_hi - emp_lo;
        LIGHTGP_CHECK(g.grid_mins[0] <= emp_lo - 0.1f * emp_range + 1e-3f);
        const float gmax = g.grid_mins[0] + g.grid_steps[0] * (g.grid_sizes[0] - 1);
        LIGHTGP_CHECK(gmax >= emp_hi + 0.1f * emp_range - 1e-3f);
    }

    // -- Grid construction (2D) -----------------------------------------------
    {
        Tensor X = make_X2_grid(10, 10, 0.0f, 1.0f);
        SKIGrid g = SKIGrid::from_data(X, /*points_per_dim=*/0);
        LIGHTGP_CHECK(g.D == 2);
        // Default heuristic: M_d = max(50, ceil(N^(1/D))) = max(50, ceil(sqrt(100))) = 50.
        LIGHTGP_CHECK(g.grid_sizes[0] == 50);
        LIGHTGP_CHECK(g.grid_sizes[1] == 50);
        LIGHTGP_CHECK(g.total_points == 2500);
    }

    // -- Interpolation matrix sparsity + weight sum ---------------------------
    {
        Tensor X = make_X1(30, 0.0f, 1.0f, 0xB2u);
        SKIGrid g = SKIGrid::from_data(X, 32);
        SparseMatrix W = build_interpolation_matrix(X, g);
        LIGHTGP_CHECK(W.rows == 30);
        LIGHTGP_CHECK(W.cols == 32);
        for (int i = 0; i < W.rows; ++i) {
            const int nnz_row = W.row_ptr[i + 1] - W.row_ptr[i];
            // Up to 4 nonzero in 1D (4^1 = 4); boundary points might collapse some.
            LIGHTGP_CHECK(nnz_row >= 1 && nnz_row <= 4);
            float sum = 0.0f;
            for (int k = W.row_ptr[i]; k < W.row_ptr[i + 1]; ++k) sum += W.values[k];
            // Cardinal cubic weights sum to exactly 1 for any t (boundary clamping
            // preserves this because we just duplicate indices, not values).
            LIGHTGP_CHECK_NEAR(sum, 1.0f, 1e-5f);
        }
    }

    // -- Cubic interpolation reproduces a linear function exactly -------------
    {
        // f(x) = 2x + 0.5. Interpolating f on a regular grid via cubic weights
        // recovers f exactly (cubics reproduce linears).
        Tensor X = make_X1(40, 0.1f, 0.9f, 0xC3u);
        SKIGrid g = SKIGrid::from_data(X, 64);
        SparseMatrix W = build_interpolation_matrix(X, g);
        Tensor f_grid(g.total_points, 1);
        for (int i = 0; i < g.total_points; ++i) {
            const float gx = g.grid_mins[0] + g.grid_steps[0] * i;
            f_grid(i, 0) = 2.0f * gx + 0.5f;
        }
        Tensor f_interp = W.matvec(f_grid);
        for (int i = 0; i < X.rows(); ++i) {
            const float truth = 2.0f * X(i, 0) + 0.5f;
            LIGHTGP_CHECK_NEAR(f_interp(i, 0), truth, 1e-4f);
        }
    }

    // -- Toeplitz column matches direct kernel evaluation ---------------------
    {
        RBFKernel kernel(0.4f, 1.1f);
        const int M = 20;
        const float h = 0.05f;
        Tensor col = toeplitz_column_1d(kernel, M, h);
        for (int j = 0; j < M; ++j) {
            const float r = j * h;
            const float expected = 1.1f * std::exp(-0.5f * (r * r) / (0.4f * 0.4f));
            LIGHTGP_CHECK_NEAR(col(j, 0), expected, 1e-5f);
        }
    }

    // -- Dense Toeplitz matvec vs explicit dense kmat -------------------------
    {
        const int M = 32;
        const float h = 0.1f;
        RBFKernel kernel(0.5f, 1.0f);
        // Build the M x M Toeplitz K_grid by direct evaluation.
        Tensor X_grid(M, 1);
        for (int i = 0; i < M; ++i) X_grid(i, 0) = h * i;
        Tensor K_dense = rbf_kernel_cpu(X_grid, X_grid, 0.5f, 1.0f);
        Tensor col = toeplitz_column_1d(kernel, M, h);
        Tensor v(M, 1);
        std::mt19937_64 rng(0xD4u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < M; ++i) v(i, 0) = nd(rng);

        Tensor ref = K_dense.matmul(v);
        Tensor got = toeplitz_matvec_cpu(col, v);
        LIGHTGP_CHECK(relative_error(got, ref) < 1e-5f);
    }

    // -- SKI matvec (CPU) vs dense K matvec, RBF, 1D --------------------------
    {
        // SKI is an approximation, so we compare with a moderate tolerance.
        Tensor X = make_X1(200, 0.0f, 4.0f, 0xE5u);
        RBFKernel kernel(0.5f, 1.0f);
        const float sn2 = 1e-3f;
        SKIData ski = build_ski(X, kernel, sn2, /*points_per_dim=*/128, Backend::CPU);

        Tensor K = rbf_kernel_cpu(X, X, 0.5f, 1.0f);
        K.add_jitter(sn2);
        Tensor v(X.rows(), 1);
        std::mt19937_64 rng(0xF6u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < static_cast<int>(X.rows()); ++i) v(i, 0) = nd(rng);

        Tensor ref = K.matmul(v);
        Tensor got = ski.matvec(v);
        const float rel = relative_error(got, ref);
        LIGHTGP_CHECK(rel < 5e-2f);  // SKI approximation tolerance
    }

#ifdef LIGHTGP_HAS_CUDA
    if (CudaContext::instance().available()) {
        // -- SKI CUDA matvec matches CPU SKI matvec ---------------------------
        Tensor X = make_X1(400, -1.0f, 2.0f, 0x107u);
        RBFKernel kernel(0.4f, 0.9f);
        const float sn2 = 5e-3f;
        SKIData ski_cpu = build_ski(X, kernel, sn2, /*points_per_dim=*/128, Backend::CPU);
        SKIData ski_gpu = build_ski(X, kernel, sn2, /*points_per_dim=*/128, Backend::CUDA);
        Tensor v(X.rows(), 1);
        std::mt19937_64 rng(0x208u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < static_cast<int>(X.rows()); ++i) v(i, 0) = nd(rng);
        Tensor c = ski_cpu.matvec(v);
        Tensor g = ski_gpu.matvec(v);
        LIGHTGP_CHECK(relative_error(g, c) < 1e-4f);
    }
#endif

    // -- End-to-end GPExact with Solver::SKI on sin(x) ------------------------
    {
        const int N = 500;
        Tensor X = make_X1(N, 0.0f, 6.0f, 0x309u);
        Tensor y(N, 1);
        for (int i = 0; i < N; ++i) y(i, 0) = std::sin(X(i, 0));
        GPHyperparams hp;
        hp.length_scale = 0.5f;
        hp.signal_variance = 1.0f;
        hp.noise_variance = 0.01f;

        const Backend backend =
#ifdef LIGHTGP_HAS_CUDA
            CudaContext::instance().available() ? Backend::CUDA :
#endif
            Backend::CPU;
        GPExact gp(hp, backend, Solver::SKI);
        LIGHTGP_CHECK(gp.fit(X, y));
        LIGHTGP_CHECK(gp.fitted());

        Tensor X_test(20, 1);
        for (int i = 0; i < 20; ++i) X_test(i, 0) = 0.3f + 0.27f * i;
        Tensor mean_out, var_out;
        LIGHTGP_CHECK(gp.predict(X_test, mean_out, var_out));
        float rmse = 0.0f;
        for (int i = 0; i < 20; ++i) {
            const float err = mean_out(i, 0) - std::sin(X_test(i, 0));
            rmse += err * err;
        }
        rmse = std::sqrt(rmse / 20.0f);
        LIGHTGP_CHECK(rmse < 0.3f);

        const float ll = gp.log_marginal_likelihood();
        LIGHTGP_CHECK(std::isfinite(ll));
    }

#ifdef LIGHTGP_HAS_ACCELERATE
    // -- vDSP FFT Toeplitz matvec matches the dense O(M^2) reference --------
    // Random-but-decaying symmetric Toeplitz column; FFT path must match the
    // dense path on a few random vectors within reasonable float32 tolerance.
    {
        std::mt19937_64 rng(0xF1B0u);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (int M : {8, 16, 32, 63, 128}) {
            Tensor col(M, 1);
            for (int i = 0; i < M; ++i) col(i, 0) = std::exp(-0.05f * i) + 0.01f * gauss(rng);
            ToeplitzFFTCpu plan(col);
            LIGHTGP_CHECK(plan.M() == M);
            for (int trial = 0; trial < 4; ++trial) {
                Tensor v(M, 1);
                for (int i = 0; i < M; ++i) v(i, 0) = gauss(rng);
                Tensor fft_out = plan.matvec(v);
                Tensor dense_out = toeplitz_matvec_cpu(col, v);
                // Allow ~1e-4 relative — FFT round-trip + complex multiply on fp32.
                LIGHTGP_CHECK(relative_error(fft_out, dense_out) < 1e-4f);
            }
        }
    }

    // -- 2D Kronecker-Toeplitz via FFT matches the dense reference ----------
    {
        std::mt19937_64 rng(0xF1B1u);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        const int Mx = 16, My = 32;
        const int total = Mx * My;
        Tensor col_x(Mx, 1), col_y(My, 1);
        for (int i = 0; i < Mx; ++i) col_x(i, 0) = std::exp(-0.10f * i);
        for (int i = 0; i < My; ++i) col_y(i, 0) = std::exp(-0.03f * i);
        std::vector<std::unique_ptr<ToeplitzFFTCpu>> plans;
        plans.emplace_back(std::make_unique<ToeplitzFFTCpu>(col_x));
        plans.emplace_back(std::make_unique<ToeplitzFFTCpu>(col_y));
        std::vector<Tensor> cols{col_x, col_y};
        std::vector<int> grid_sizes{Mx, My};

        Tensor v(total, 1);
        for (int i = 0; i < total; ++i) v(i, 0) = gauss(rng);
        Tensor fft_out = kron_toeplitz_matvec_accelerate(plans, grid_sizes, v);
        Tensor dense_out = kron_toeplitz_matvec_cpu(cols, grid_sizes, v);
        LIGHTGP_CHECK(relative_error(fft_out, dense_out) < 1e-4f);
    }
#endif  // LIGHTGP_HAS_ACCELERATE
}

}  // namespace lightgp
