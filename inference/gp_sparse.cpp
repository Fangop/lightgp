#include "gp_sparse.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "../core/dispatch.h"
#include "../solvers/cpu/cholesky_cpu.h"

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kInducingJitter = 1e-6f;

/// Profile-print helper for GPSparse::fit. Enabled when LIGHTGP_PROFILE_SPARSE=1
/// is set in the environment. Each call prints `[gpsparse] <label>: <ms>` and resets
/// the timer. Resolution: std::chrono::steady_clock (typically 1us on Linux).
class Profiler {
public:
    Profiler() : enabled_(std::getenv("LIGHTGP_PROFILE_SPARSE") != nullptr),
                 start_(clock::now()) {}
    void tick(const char* label) {
        if (!enabled_) return;
        const auto now = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - start_).count();
        std::fprintf(stderr, "[gpsparse] %-26s %7.2f ms\n", label, ms);
        start_ = now;
    }
private:
    using clock = std::chrono::steady_clock;
    bool enabled_;
    clock::time_point start_;
};

// Farthest-point sampling: deterministic given seed. Returns M selected rows of X.
Tensor farthest_point_sample(const Tensor& X, std::size_t M, std::uint64_t seed) {
    const std::size_t N = X.rows();
    const std::size_t D = X.cols();
    assert(M <= N && M > 0);

    std::mt19937_64 rng(seed);
    Tensor Z(M, D);
    std::vector<float> min_dist_sq(N, std::numeric_limits<float>::max());

    const std::size_t first = rng() % N;
    for (std::size_t d = 0; d < D; ++d) Z(0, d) = X(first, d);
    std::size_t last = first;

    for (std::size_t m = 1; m < M; ++m) {
        for (std::size_t i = 0; i < N; ++i) {
            float d2 = 0.0f;
            for (std::size_t k = 0; k < D; ++k) {
                const float diff = X(i, k) - X(last, k);
                d2 += diff * diff;
            }
            if (d2 < min_dist_sq[i]) min_dist_sq[i] = d2;
        }
        std::size_t best = 0;
        float best_d = -1.0f;
        for (std::size_t i = 0; i < N; ++i) {
            if (min_dist_sq[i] > best_d) {
                best_d = min_dist_sq[i];
                best = i;
            }
        }
        for (std::size_t d = 0; d < D; ++d) Z(m, d) = X(best, d);
        last = best;
    }
    return Z;
}

}  // namespace

GPSparse::GPSparse(GPSparseHyperparams hp, Backend backend)
    : hp_(hp), backend_(backend) {}

Backend GPSparse::effective_backend() const {
    if (backend_ != Backend::Auto) return backend_;
    // For sparse GP, Cholesky is over the M x M Sigma; M dominates inner cost.
    // Use the inducing-point count if Z is set, else fall back to N.
    const std::size_t n = (Z_.rows() > 0) ? Z_.rows() : X_train_.rows();
    const std::size_t d = X_train_.cols();
    return resolve_auto_backend(n, d, Solver::Cholesky);
}

bool GPSparse::fit(const Tensor& X_train, const Tensor& y_train,
                   std::size_t M, std::uint64_t seed) {
    assert(X_train.rows() == y_train.rows());
    assert(y_train.cols() == 1);
    assert(M > 0 && M <= X_train.rows());

    Profiler prof;
    X_train_ = X_train;
    y_train_ = y_train;
    Z_ = farthest_point_sample(X_train, M, seed);
    prof.tick("farthest_point_sample");

    const std::size_t N = X_train_.rows();

    // K_uu (M x M) + K_uf (M x N).
    Tensor K_uu = dispatch_kernel(Z_, Z_,
                                  hp_.length_scale, hp_.signal_variance,
                                  hp_.kernel, effective_backend());
    prof.tick("K_uu kernel");
    K_uu.add_jitter(kInducingJitter);
    K_uf_ = dispatch_kernel(Z_, X_train_,
                            hp_.length_scale, hp_.signal_variance,
                            hp_.kernel, effective_backend());
    prof.tick("K_uf kernel");

    float jit = 0.0f;
    if (!dispatch_cholesky_with_jitter(K_uu, L_uu_, jit, effective_backend())) {
        fitted_ = false;
        return false;
    }
    prof.tick("L_uu cholesky");

    // V = L_uu^{-1} K_uf   →   Q_ff_diag(i) = ||V(:, i)||^2.
    Tensor V = dispatch_forward_solve(L_uu_, K_uf_, effective_backend());
    prof.tick("V = L_uu^-1 K_uf");

    Lambda_ = Tensor(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        float q_ii = 0.0f;
        for (std::size_t m = 0; m < M; ++m) q_ii += V(m, i) * V(m, i);
        float lam = hp_.signal_variance - q_ii + hp_.noise_variance;
        if (lam < hp_.noise_variance) lam = hp_.noise_variance;  // numerical safety
        Lambda_(i, 0) = lam;
    }
    prof.tick("Lambda (CPU loop)");

    // Σ = K_uu + K_uf Λ^{-1} K_fu = K_uu + (Λ^{-1/2} K_fu)^T (Λ^{-1/2} K_fu).
    // Form K_fu_scaled = Λ^{-1/2} K_fu (N x M) by row-scaling.
    Tensor K_fu_scaled(N, M);
    for (std::size_t i = 0; i < N; ++i) {
        const float s = 1.0f / std::sqrt(Lambda_(i, 0));
        for (std::size_t m = 0; m < M; ++m) {
            K_fu_scaled(i, m) = K_uf_(m, i) * s;
        }
    }
    prof.tick("K_fu_scaled (CPU loop)");
    Tensor outer = K_fu_scaled.transpose().matmul(K_fu_scaled);
    prof.tick("K_fu^T K_fu (CPU BLAS)");
    Tensor Sigma = K_uu.add(outer);

    if (!dispatch_cholesky_with_jitter(Sigma, L_Sigma_, jit, effective_backend())) {
        fitted_ = false;
        return false;
    }
    prof.tick("L_Sigma cholesky");

    // α = Σ^{-1} K_uf Λ^{-1} y.
    Tensor y_scaled(N, 1);
    for (std::size_t i = 0; i < N; ++i) y_scaled(i, 0) = y_train_(i, 0) / Lambda_(i, 0);
    Tensor Kuf_y = K_uf_.matmul(y_scaled);
    alpha_ = dispatch_cholesky_solve(L_Sigma_, Kuf_y, effective_backend());
    prof.tick("alpha solve");

    fitted_ = true;
    return true;
}

bool GPSparse::predict(const Tensor& X_test, Tensor& mean_out, Tensor& var_out) const {
    if (!fitted_) return false;
    const std::size_t M_test = X_test.rows();
    const std::size_t M = Z_.rows();

    Tensor K_su = dispatch_kernel(X_test, Z_,
                                  hp_.length_scale, hp_.signal_variance,
                                  hp_.kernel, effective_backend());

    mean_out = K_su.matmul(alpha_);

    // var(i) = sf2 - ||L_uu^{-1} K_us(:, i)||^2 + ||L_Σ^{-1} K_us(:, i)||^2.
    Tensor K_us = K_su.transpose();  // M x M_test
    Tensor V = dispatch_forward_solve(L_uu_, K_us, effective_backend());
    Tensor W = dispatch_forward_solve(L_Sigma_, K_us, effective_backend());

    var_out = Tensor(M_test, 1);
    const float prior_var = hp_.signal_variance;
    for (std::size_t j = 0; j < M_test; ++j) {
        float v_sq = 0.0f, w_sq = 0.0f;
        for (std::size_t m = 0; m < M; ++m) {
            v_sq += V(m, j) * V(m, j);
            w_sq += W(m, j) * W(m, j);
        }
        float var = prior_var - v_sq + w_sq;
        if (var < 0.0f) var = 0.0f;
        var_out(j, 0) = var;
    }
    return true;
}

float GPSparse::log_marginal_likelihood() const {
    if (!fitted_) return std::nanf("");
    const std::size_t N = X_train_.rows();

    float log_Lambda = 0.0f;
    for (std::size_t i = 0; i < N; ++i) log_Lambda += std::log(Lambda_(i, 0));

    const float log_K_uu = log_det_from_cholesky(L_uu_);
    const float log_Sigma = log_det_from_cholesky(L_Sigma_);

    // y^T (Q_ff + Λ)^{-1} y = y^T Λ^{-1} y - (K_uf Λ^{-1} y)^T α
    float yty = 0.0f;
    Tensor y_scaled(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        const float ly = y_train_(i, 0) / Lambda_(i, 0);
        yty += y_train_(i, 0) * ly;
        y_scaled(i, 0) = ly;
    }
    Tensor Kuf_y = K_uf_.matmul(y_scaled);

    float quad = 0.0f;
    for (std::size_t m = 0; m < alpha_.rows(); ++m) quad += Kuf_y(m, 0) * alpha_(m, 0);

    const float data_term = yty - quad;

    return -0.5f * (static_cast<float>(N) * std::log(2.0f * kPi)
                    + log_Lambda + log_Sigma - log_K_uu + data_term);
}

}  // namespace lightgp
