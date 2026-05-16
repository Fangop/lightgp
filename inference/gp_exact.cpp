#include "gp_exact.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include <random>

#include "../core/dispatch.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../solvers/cpu/cg_cpu.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "../solvers/cpu/slq_cpu.h"
#include "ski.h"
#include "../kernels/rbf_kernel.h"
#include "../kernels/matern_kernel.h"

#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_matvec.h"
#endif
#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#include "../kernels/cuda/rbf_matvec_cuda.h"
#endif

namespace lightgp {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

GPExact::GPExact(GPHyperparams hp, Backend backend, Solver solver)
    : hp_(hp), backend_(backend), solver_(solver) {}

GPExact::~GPExact() = default;
GPExact::GPExact(GPExact&&) noexcept = default;
GPExact& GPExact::operator=(GPExact&&) noexcept = default;

GPExact::GPExact(std::shared_ptr<Kernel> kernel,
                 std::shared_ptr<MeanFunction> mean,
                 float noise_variance,
                 Backend backend,
                 Solver solver)
    : backend_(backend), solver_(solver),
      kernel_(std::move(kernel)),
      mean_(std::move(mean) ? std::move(mean) : std::make_shared<ZeroMean>()),
      noise_variance_(noise_variance) {
    if (!mean_) mean_ = std::make_shared<ZeroMean>();
    // Mirror noise into hp_ so the legacy accessor still returns it.
    hp_.noise_variance = noise_variance_;
}

Backend GPExact::effective_backend() const {
    if (backend_ != Backend::Auto) return backend_;
    const std::size_t n = X_train_.rows();
    const std::size_t d = X_train_.cols();
    return resolve_auto_backend(n, d, solver_);
}

Tensor GPExact::matvec_impl(const Tensor& v) const {
    if (ski_data_) {
        return ski_data_->matvec(v);
    }
    if (matrix_free_) {
#ifdef LIGHTGP_HAS_METAL
        if (effective_backend() == Backend::Metal) {
            return rbf_matvec_metal(X_train_, v,
                                    hp_.length_scale, hp_.signal_variance, hp_.noise_variance);
        }
#endif
#ifdef LIGHTGP_HAS_CUDA
        if (effective_backend() == Backend::CUDA) {
            return rbf_matvec_cuda(X_train_, v,
                                   hp_.length_scale, hp_.signal_variance, hp_.noise_variance);
        }
#endif
    }
    return K_y_.matmul(v);
}

namespace {

// Build a Kernel object for the SKI Toeplitz column when only the legacy
// (GPHyperparams) API is in use. The returned shared_ptr owns the temporary.
std::shared_ptr<Kernel> kernel_from_legacy(const GPHyperparams& hp) {
    switch (hp.kernel) {
        case KernelType::RBF:
            return std::make_shared<RBFKernel>(hp.length_scale, hp.signal_variance);
        case KernelType::Matern12:
            return std::make_shared<MaternKernel>(0.5f, hp.length_scale, hp.signal_variance);
        case KernelType::Matern32:
            return std::make_shared<MaternKernel>(1.5f, hp.length_scale, hp.signal_variance);
        case KernelType::Matern52:
            return std::make_shared<MaternKernel>(2.5f, hp.length_scale, hp.signal_variance);
    }
    return std::make_shared<RBFKernel>(hp.length_scale, hp.signal_variance);
}

}  // namespace

bool GPExact::fit(const Tensor& X_train, const Tensor& y_train) {
    assert(X_train.rows() == y_train.rows());
    assert(y_train.cols() == 1);
    X_train_ = X_train;
    y_train_ = y_train;
    ski_data_.reset();  // drop any stale SKI state from a previous fit.

    // SKI solver: build the structured-kernel approximation up front, then drive
    // CG through ski_data_->matvec(v). Works for both legacy and new-API
    // constructions; the legacy path synthesises a temporary Kernel from `hp_`.
    if (solver_ == Solver::SKI) {
        std::shared_ptr<Kernel> kernel_for_ski;
        if (kernel_) {
            y_train_orig_ = y_train_;
            Tensor mean_train = mean_->compute(X_train_);
            Tensor y_centered(y_train_.rows(), 1);
            for (std::size_t i = 0; i < y_train_.rows(); ++i)
                y_centered(i, 0) = y_train_(i, 0) - mean_train(i, 0);
            y_train_ = std::move(y_centered);
            kernel_for_ski = kernel_;
        } else {
            kernel_for_ski = kernel_from_legacy(hp_);
        }
        const float sn2 = kernel_ ? noise_variance_ : hp_.noise_variance;
        ski_data_ = std::make_unique<SKIData>(
            build_ski(X_train_, *kernel_for_ski, sn2, /*points_per_dim=*/0,
                      effective_backend()));
        matrix_free_ = true;
        K_y_ = Tensor();

        auto matvec = [this](const Tensor& v) { return matvec_impl(v); };
        Tensor alpha;
        CGResult r = cg_solve_matvec(matvec, X_train_.rows(), y_train_, alpha,
                                     /*tol=*/1e-5f, /*max_iter=*/500);
        if (!r.converged) { fitted_ = false; return false; }
        alpha_ = std::move(alpha);
        jitter_used_ = sn2;
        fitted_ = true;
        return true;
    }

    // New API: when constructed with a Kernel + MeanFunction object, use the
    // composable kernel hierarchy. Center y by the mean function before solving.
    if (kernel_) {
        // Preserve original y so re-fits during optimize() can re-center with
        // updated mean-function parameters.
        y_train_orig_ = y_train_;
        Tensor mean_train = mean_->compute(X_train_);
        Tensor y_centered(y_train_.rows(), 1);
        for (std::size_t i = 0; i < y_train_.rows(); ++i)
            y_centered(i, 0) = y_train_(i, 0) - mean_train(i, 0);
        // Store the centered targets so predict() can add the mean back symmetrically.
        y_train_ = std::move(y_centered);

        if (solver_ == Solver::CG) {
            // If the kernel is a plain RBFKernel and we're on a backend that has a
            // matrix-free RBF matvec (Metal / CUDA), avoid materialising K. This is
            // what the legacy GPHyperparams path does — we extend the parity to the
            // new-API path so Python users get the same scalability.
            auto* rbf = dynamic_cast<RBFKernel*>(kernel_.get());
            const bool gpu_matrix_free =
                rbf != nullptr &&
                (effective_backend() == Backend::Metal || effective_backend() == Backend::CUDA);
            if (gpu_matrix_free) {
                // Mirror kernel params into hp_ so matvec_impl can call rbf_matvec_*().
                hp_.length_scale = rbf->length_scale();
                hp_.signal_variance = rbf->signal_variance();
                hp_.noise_variance = noise_variance_;
                matrix_free_ = true;
                K_y_ = Tensor();
            } else {
                // Composite / unsupported-backend kernels still materialise K.
                Tensor K_new = kernel_->compute(X_train_, X_train_, effective_backend());
                K_new.add_jitter(noise_variance_);
                K_y_ = std::move(K_new);
                matrix_free_ = false;
            }
            auto matvec = [this](const Tensor& v) { return matvec_impl(v); };
            Tensor alpha;
            CGResult r = cg_solve_matvec(matvec, X_train_.rows(), y_train_, alpha,
                                         /*tol=*/1e-6f, /*max_iter=*/1000);
            if (!r.converged) { fitted_ = false; return false; }
            alpha_ = std::move(alpha);
            jitter_used_ = noise_variance_;
            fitted_ = true;
            return true;
        }
        Tensor K_new = kernel_->compute(X_train_, X_train_, effective_backend());
        K_new.add_jitter(noise_variance_);
        float jit_new = 0.0f;
        if (!dispatch_cholesky_with_jitter(K_new, L_, jit_new, effective_backend())) {
            fitted_ = false; return false;
        }
        jitter_used_ = jit_new;
        alpha_ = dispatch_cholesky_solve(L_, y_train_, effective_backend());
        fitted_ = true;
        return true;
    }

    Tensor K = dispatch_kernel(X_train_, X_train_,
                               hp_.length_scale, hp_.signal_variance,
                               hp_.kernel, effective_backend());
    K.add_jitter(hp_.noise_variance);

    if (solver_ == Solver::CG) {
        // Matrix-free path on Metal / CUDA: skip materializing the N x N kernel.
        matrix_free_ = false;
#ifdef LIGHTGP_HAS_METAL
        if (effective_backend() == Backend::Metal && MetalContext::instance().available())
            matrix_free_ = true;
#endif
#ifdef LIGHTGP_HAS_CUDA
        if (effective_backend() == Backend::CUDA && CudaContext::instance().available())
            matrix_free_ = true;
#endif
        if (!matrix_free_) {
            K_y_ = std::move(K);
        } else {
            K_y_ = Tensor();
        }

        auto matvec = [this](const Tensor& v) { return matvec_impl(v); };
        Tensor alpha;
        CGResult r = cg_solve_matvec(matvec, X_train_.rows(), y_train_, alpha,
                                     /*tol=*/1e-6f, /*max_iter=*/1000);
        if (!r.converged) {
            fitted_ = false;
            return false;
        }
        alpha_ = std::move(alpha);
        jitter_used_ = hp_.noise_variance;
        fitted_ = true;
        return true;
    }

    float jit = 0.0f;
    if (!dispatch_cholesky_with_jitter(K, L_, jit, effective_backend())) {
        fitted_ = false;
        return false;
    }
    jitter_used_ = jit;
    alpha_ = dispatch_cholesky_solve(L_, y_train_, effective_backend());
    fitted_ = true;
    return true;
}

bool GPExact::predict(const Tensor& X_test, Tensor& mean_out, Tensor& var_out) const {
    if (!fitted_) return false;

    const std::size_t n = X_train_.rows();
    const std::size_t m = X_test.rows();

    // Pick kernel construction call based on which API was used to construct.
    Tensor K_star = kernel_
        ? kernel_->compute(X_train_, X_test, effective_backend())
        : dispatch_kernel(X_train_, X_test,
                          hp_.length_scale, hp_.signal_variance,
                          hp_.kernel, effective_backend());
    Tensor K_star_t = K_star.transpose();
    mean_out = K_star_t.matmul(alpha_);

    // Add mean function back (new API) — legacy API has implicit zero mean.
    if (kernel_) {
        Tensor mean_test = mean_->compute(X_test);
        for (std::size_t i = 0; i < m; ++i) mean_out(i, 0) += mean_test(i, 0);
    }

    var_out = Tensor::zeros(m, 1);
    // Prior variance per test point: kernel diagonal at x_test. For new API this
    // properly handles composite kernels (sum of diagonals). For legacy: scalar sf2.
    Tensor prior_var_vec;
    if (kernel_) prior_var_vec = kernel_->compute_diag(X_test);
    const float prior_var = hp_.signal_variance;  // legacy: k(x, x) = sf2 for RBF

    if (solver_ == Solver::CG || solver_ == Solver::SKI) {
        // Hutchinson probe estimator for diag(K_star^T K_y^{-1} K_star).
        // For Rademacher z ∈ R^N: E[(K_star^T z) ⊙ (K_star^T K_y^{-1} z)] = the diagonal we want.
        // Cost: n_probes CG solves total, independent of M_test (vs M_test solves before).
        constexpr int n_probes = 30;
        auto matvec = [this](const Tensor& v) { return matvec_impl(v); };
        std::mt19937_64 rng(789);
        std::uniform_int_distribution<int> coin(0, 1);

        Tensor Kst_T = K_star.transpose();  // m x n
        Tensor var_acc = Tensor::zeros(m, 1);
        int valid = 0;
        for (int p = 0; p < n_probes; ++p) {
            Tensor z(n, 1);
            for (std::size_t i = 0; i < n; ++i) z(i, 0) = coin(rng) ? 1.0f : -1.0f;

            Tensor v_probe = Kst_T.matmul(z);  // K_star^T z, deterministic
            Tensor u;
            CGResult r = cg_solve_matvec(matvec, n, z, u, /*tol=*/1e-5f, /*max_iter=*/500);
            if (!r.converged) continue;
            Tensor w = Kst_T.matmul(u);  // K_star^T K_y^{-1} z

            for (std::size_t i = 0; i < m; ++i) var_acc(i, 0) += v_probe(i, 0) * w(i, 0);
            valid++;
        }
        const float denom = static_cast<float>(std::max(1, valid));
        for (std::size_t j = 0; j < m; ++j) {
            float reduction = var_acc(j, 0) / denom;
            const float pv = kernel_ ? prior_var_vec(j, 0) : prior_var;
            float var = pv - reduction;
            if (var < 0.0f) var = 0.0f;
            var_out(j, 0) = var;
        }
        return true;
    }

    // Cholesky path: v = L^{-1} K_star (forward only) then var_j = k(x*,x*) - ||v_j||^2.
    Tensor v = dispatch_forward_solve(L_, K_star, effective_backend());
    for (std::size_t j = 0; j < m; ++j) {
        float reduction = 0.0f;
        for (std::size_t i = 0; i < n; ++i) reduction += v(i, j) * v(i, j);
        const float pv = kernel_ ? prior_var_vec(j, 0) : prior_var;
        float var = pv - reduction;
        if (var < 0.0f) var = 0.0f;
        var_out(j, 0) = var;
    }
    return true;
}

float GPExact::log_marginal_likelihood() const {
    if (!fitted_) return std::nanf("");
    const std::size_t n = y_train_.rows();
    float yTa = 0.0f;
    for (std::size_t i = 0; i < n; ++i) yTa += y_train_(i, 0) * alpha_(i, 0);
    float log_det = 0.0f;
    if (solver_ == Solver::CG || solver_ == Solver::SKI) {
        // SLQ via the same matvec used by CG / SKI (matrix-free on Metal/CUDA, or
        // structured-Toeplitz for SKI; materialized for CG-CPU).
        auto matvec = [this](const Tensor& v) { return matvec_impl(v); };
        log_det = slq_log_det_cpu(matvec, n, /*n_probes=*/20, /*n_iters=*/30, /*seed=*/123);
    } else {
        log_det = log_det_from_cholesky(L_);
    }
    return -0.5f * yTa - 0.5f * log_det
           - 0.5f * static_cast<float>(n) * std::log(2.0f * kPi);
}

void GPExact::log_marginal_likelihood_grads(float& d_log_l,
                                            float& d_log_sf2,
                                            float& d_log_sn2) const {
    assert(fitted_);
    const std::size_t n = X_train_.rows();

    Tensor grad_l, grad_sf2;
    rbf_kernel_gradients_cpu(X_train_, X_train_,
                             hp_.length_scale, hp_.signal_variance,
                             grad_l, grad_sf2);

    if (solver_ == Solver::CG || solver_ == Solver::SKI) {
        // Hutchinson trace estimator with Rademacher probes:
        // tr(K_y^{-1} M) ≈ (1/m) Σ z^T (M (K_y^{-1} z)).
        // One CG solve per probe; reuse the v = K_y^{-1} z across all three M_θ.
        constexpr int n_probes = 20;
        std::mt19937_64 rng(456);
        std::uniform_int_distribution<int> coin(0, 1);
        auto matvec = [this](const Tensor& v) { return matvec_impl(v); };

        double tr_l = 0.0, tr_sf2 = 0.0, tr_sn2 = 0.0;
        int valid = 0;
        for (int p = 0; p < n_probes; ++p) {
            Tensor z(n, 1);
            for (std::size_t i = 0; i < n; ++i) z(i, 0) = coin(rng) ? 1.0f : -1.0f;

            Tensor v;
            CGResult r = cg_solve_matvec(matvec, n, z, v, /*tol=*/1e-5f, /*max_iter=*/500);
            if (!r.converged) continue;

            const Tensor Mv_l = grad_l.matmul(v);
            const Tensor Mv_sf2 = grad_sf2.matmul(v);
            for (std::size_t i = 0; i < n; ++i) {
                tr_l += static_cast<double>(z(i, 0)) * Mv_l(i, 0);
                tr_sf2 += static_cast<double>(z(i, 0)) * Mv_sf2(i, 0);
                // M_sn2 = sn2 * I, so z^T M_sn2 v = sn2 * z^T v.
                tr_sn2 += static_cast<double>(z(i, 0)) * (hp_.noise_variance * v(i, 0));
            }
            valid++;
        }
        const double denom = std::max(1, valid);
        tr_l /= denom; tr_sf2 /= denom; tr_sn2 /= denom;

        auto alpha_M_alpha = [&](const Tensor& M) {
            float s = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                float row = 0.0f;
                for (std::size_t j = 0; j < n; ++j) row += M(i, j) * alpha_(j, 0);
                s += alpha_(i, 0) * row;
            }
            return s;
        };
        float a_sq = 0.0f;
        for (std::size_t i = 0; i < n; ++i) a_sq += alpha_(i, 0) * alpha_(i, 0);

        d_log_l = 0.5f * (alpha_M_alpha(grad_l) - static_cast<float>(tr_l));
        d_log_sf2 = 0.5f * (alpha_M_alpha(grad_sf2) - static_cast<float>(tr_sf2));
        d_log_sn2 = 0.5f * (hp_.noise_variance * a_sq - static_cast<float>(tr_sn2));
        return;
    }

    // Cholesky path: exact gradients via explicit K_y^{-1}.
    Tensor I = Tensor::eye(n);
    Tensor K_inv = dispatch_cholesky_solve(L_, I, effective_backend());

    auto contribution = [&](const Tensor& dK) {
        float aTdKa = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float row = 0.0f;
            for (std::size_t j = 0; j < n; ++j) row += dK(i, j) * alpha_(j, 0);
            aTdKa += alpha_(i, 0) * row;
        }
        // tr(K_inv dK) = sum_{ij} K_inv(i,j) * dK(i,j) since both matrices are symmetric.
        float tr = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                tr += K_inv(i, j) * dK(i, j);
        return 0.5f * (aTdKa - tr);
    };

    d_log_sf2 = contribution(grad_sf2);
    d_log_l = contribution(grad_l);

    float alpha_sq = 0.0f;
    for (std::size_t i = 0; i < n; ++i) alpha_sq += alpha_(i, 0) * alpha_(i, 0);
    float trace_K_inv = 0.0f;
    for (std::size_t i = 0; i < n; ++i) trace_K_inv += K_inv(i, i);
    d_log_sn2 = 0.5f * hp_.noise_variance * (alpha_sq - trace_K_inv);
}

// Finite-difference Adam over [kernel.log_params, log_noise, mean.params].
// Works with any kernel composition because gradients come from re-fitting at
// θ ± ε rather than from analytical derivatives. Each step costs 2*P+1 fits
// (P = total parameter count). At N=624 / Mauna Loa ~0.5 ms per fit so 10
// params → ~10 ms per step. At N=2048 → ~700 ms per step.
static bool optimize_new_api(GPExact* gp, int steps, float lr, bool verbose);

bool GPExact::optimize_hyperparameters(int num_steps, float learning_rate, bool verbose) {
    if (!fitted_) return false;
    if (kernel_) return optimize_new_api(this, num_steps, learning_rate, verbose);

    // Adam optimizer in log-hyperparameter space, ascending log marginal likelihood.
    // We still track the best LL and restore at the end because Adam can overshoot near optima.
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps_adam = 1e-8f;
    constexpr float grad_tol = 1e-4f;

    float theta[3] = {std::log(hp_.length_scale),
                      std::log(hp_.signal_variance),
                      std::log(hp_.noise_variance)};
    float m[3] = {0.0f, 0.0f, 0.0f};
    float v[3] = {0.0f, 0.0f, 0.0f};

    GPHyperparams best_hp = hp_;
    float best_ll = log_marginal_likelihood();

    for (int step = 1; step <= num_steps; ++step) {
        float g[3];
        log_marginal_likelihood_grads(g[0], g[1], g[2]);

        const float g_norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (g_norm < grad_tol) {
            if (verbose) {
                std::printf("  step %3d  |grad|=%.2e < tol; stopping\n", step, g_norm);
            }
            break;
        }

        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        for (int p = 0; p < 3; ++p) {
            m[p] = beta1 * m[p] + (1.0f - beta1) * g[p];
            v[p] = beta2 * v[p] + (1.0f - beta2) * g[p] * g[p];
            const float m_hat = m[p] / bc1;
            const float v_hat = v[p] / bc2;
            theta[p] += learning_rate * m_hat / (std::sqrt(v_hat) + eps_adam);
        }

        hp_.length_scale = std::exp(theta[0]);
        hp_.signal_variance = std::exp(theta[1]);
        hp_.noise_variance = std::exp(theta[2]);

        if (!fit(X_train_, y_train_)) {
            hp_ = best_hp;
            fit(X_train_, y_train_);
            return false;
        }

        const float ll = log_marginal_likelihood();
        if (ll > best_ll) {
            best_ll = ll;
            best_hp = hp_;
        }

        if (verbose && (step % 10 == 0 || step == num_steps)) {
            std::printf("  step %3d  ll=%.4f  |grad|=%.2e  l=%.4f sf2=%.4f sn2=%.5f\n",
                        step, ll, g_norm,
                        hp_.length_scale, hp_.signal_variance, hp_.noise_variance);
        }
    }

    hp_ = best_hp;
    return fit(X_train_, y_train_);
}

namespace {
struct GPExactParamView {
    int n_kernel, n_mean, n_total;
};
}

static bool optimize_new_api(GPExact* gp, int steps, float lr, bool verbose) {
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps_adam = 1e-8f;
    constexpr float fd_eps = 1e-3f;

    // The GPExact private fields we need are not exposed; use friend-free access
    // through public getters wherever possible. We need: kernel_, mean_, noise_variance_,
    // X_train_, y_train_orig_, log_marginal_likelihood(). Add tiny accessors on GPExact
    // by including the header pieces here. (Simpler: extend GPExact's own member, but
    // for time I just rely on the public surface where present.)

    // Extract the parameter vector.
    auto kp = gp->_kernel_log_params();
    auto mp = gp->_mean_params();
    const int n_kernel = static_cast<int>(kp.size());
    const int n_mean = static_cast<int>(mp.size());
    const int n_total = n_kernel + 1 + n_mean;  // +1 for log_noise

    std::vector<float> theta(n_total);
    for (int i = 0; i < n_kernel; ++i) theta[i] = kp[i];
    theta[n_kernel] = std::log(gp->_noise_variance());
    for (int i = 0; i < n_mean; ++i) theta[n_kernel + 1 + i] = mp[i];

    std::vector<float> m(n_total, 0.0f), v(n_total, 0.0f);

    auto apply_theta = [&](const std::vector<float>& t) {
        gp->_set_kernel_log_params(std::vector<float>(t.begin(), t.begin() + n_kernel));
        gp->_set_noise_variance(std::exp(t[n_kernel]));
        gp->_set_mean_params(std::vector<float>(t.begin() + n_kernel + 1, t.end()));
    };
    auto eval_ll = [&](const std::vector<float>& t) -> float {
        apply_theta(t);
        if (!gp->_refit_from_orig()) return -std::numeric_limits<float>::infinity();
        return gp->log_marginal_likelihood();
    };

    // Initial LL.
    float best_ll = eval_ll(theta);
    std::vector<float> best_theta = theta;

    for (int step = 1; step <= steps; ++step) {
        // Finite-difference gradient (central).
        std::vector<float> grad(n_total, 0.0f);
        for (int p = 0; p < n_total; ++p) {
            std::vector<float> tp = theta, tm = theta;
            tp[p] += fd_eps;
            tm[p] -= fd_eps;
            const float ll_plus = eval_ll(tp);
            const float ll_minus = eval_ll(tm);
            grad[p] = (ll_plus - ll_minus) / (2.0f * fd_eps);
        }

        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        for (int p = 0; p < n_total; ++p) {
            m[p] = beta1 * m[p] + (1.0f - beta1) * grad[p];
            v[p] = beta2 * v[p] + (1.0f - beta2) * grad[p] * grad[p];
            const float m_hat = m[p] / bc1;
            const float v_hat = v[p] / bc2;
            theta[p] += lr * m_hat / (std::sqrt(v_hat) + eps_adam);
        }

        const float ll = eval_ll(theta);
        if (ll > best_ll) { best_ll = ll; best_theta = theta; }

        if (verbose && (step % 10 == 0 || step == steps)) {
            std::printf("  step %3d  ll=%.4f  noise=%.5f\n",
                        step, ll, std::exp(theta[n_kernel]));
        }
    }
    // Restore best.
    apply_theta(best_theta);
    return gp->_refit_from_orig();
}

// ---- friend-like accessors for the new-API optimizer ----
std::vector<float> GPExact::_kernel_log_params() const {
    return kernel_ ? kernel_->get_log_params() : std::vector<float>{};
}
std::vector<float> GPExact::_mean_params() const {
    return mean_ ? mean_->get_params() : std::vector<float>{};
}
float GPExact::_noise_variance() const { return noise_variance_; }
void GPExact::_set_kernel_log_params(const std::vector<float>& p) {
    if (kernel_) kernel_->set_log_params(p);
}
void GPExact::_set_mean_params(const std::vector<float>& p) {
    if (mean_) mean_->set_params(p);
}
void GPExact::_set_noise_variance(float v) {
    noise_variance_ = v;
    hp_.noise_variance = v;
}
bool GPExact::_refit_from_orig() {
    if (!kernel_ || y_train_orig_.size() == 0) return false;
    return fit(X_train_, y_train_orig_);
}

}  // namespace lightgp
