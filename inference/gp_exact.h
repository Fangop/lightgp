#pragma once

#include <memory>

#include "../core/backend.h"
#include "../core/kernel.h"
#include "../core/mean.h"
#include "../core/solver.h"
#include "../core/tensor.h"
#include "../kernels/kernel_base.h"

namespace lightgp {

/// Hyperparameters of the exact-GP model in their natural (positive) parameterization.
/// Legacy API — for kernel composition use the Kernel+MeanFunction constructor.
struct GPHyperparams {
    KernelType kernel = KernelType::RBF;
    float length_scale = 1.0f;
    float signal_variance = 1.0f;
    float noise_variance = 1e-2f;
};

/// Gaussian Process regression with an RBF kernel.
/// Backend selects where dense linear algebra runs; Solver selects between exact
/// Cholesky and iterative CG (with stochastic Lanczos log-det). CG mode trades
/// exactness for O(N^2) memory and matrix-free scalability.
class GPExact {
public:
    /// Construct with the given hyperparameters, compute backend, and solver type.
    explicit GPExact(GPHyperparams hp = {},
                     Backend backend = Backend::CPU,
                     Solver solver = Solver::Cholesky);

    /// Construct with a Kernel + MeanFunction object (supports composition).
    /// Use std::make_shared<RBFKernel>() etc. and operator+ / scale() to compose.
    /// noise_variance is separate because the GP likelihood noise is not a kernel
    /// hyperparameter in this design.
    GPExact(std::shared_ptr<Kernel> kernel,
            std::shared_ptr<MeanFunction> mean,
            float noise_variance = 1e-2f,
            Backend backend = Backend::CPU,
            Solver solver = Solver::Cholesky);

    /// Fit on (X_train, y_train) by factorizing K + sn2*I. y_train must be N x 1.
    /// Returns false if Cholesky failed even with the maximum jitter retry.
    bool fit(const Tensor& X_train, const Tensor& y_train);

    /// Predictive posterior mean and (marginal) variance at X_test. Both outputs are M x 1.
    bool predict(const Tensor& X_test, Tensor& mean_out, Tensor& var_out) const;

    /// Log marginal likelihood of the training data under the current hyperparameters.
    float log_marginal_likelihood() const;

    /// Analytical gradients of log_marginal_likelihood w.r.t. the log-hyperparameters.
    void log_marginal_likelihood_grads(float& d_log_l,
                                       float& d_log_sf2,
                                       float& d_log_sn2) const;

    /// Maximize log marginal likelihood. Legacy-API mode uses analytical gradients;
    /// new-API (Kernel+MeanFunction) mode uses finite-difference Adam over the full
    /// parameter vector [kernel.log_params, log_noise, mean.params].
    bool optimize_hyperparameters(int num_steps = 100,
                                  float learning_rate = 0.05f,
                                  bool verbose = false);

    // Friend-like accessors used by the new-API optimizer; not part of the user API.
    std::vector<float> _kernel_log_params() const;
    std::vector<float> _mean_params() const;
    float _noise_variance() const;
    void _set_kernel_log_params(const std::vector<float>& p);
    void _set_mean_params(const std::vector<float>& p);
    void _set_noise_variance(float v);
    bool _refit_from_orig();

    /// Current hyperparameters.
    const GPHyperparams& hyperparams() const { return hp_; }
    /// True once fit() has succeeded.
    bool fitted() const { return fitted_; }
    /// Jitter that was actually added to the diagonal in the last fit.
    float jitter_used() const { return jitter_used_; }
    /// Compute backend used for kernel matrix construction.
    Backend backend() const { return backend_; }
    /// Change the compute backend; takes effect on the next fit()/predict().
    void set_backend(Backend b) { backend_ = b; }
    /// Active inference solver (Cholesky or CG).
    Solver solver() const { return solver_; }
    /// Change the solver; takes effect on the next fit().
    void set_solver(Solver s) { solver_ = s; }

private:
    /// In CG mode, evaluates K_y * v either by reading K_y_ or by calling the
    /// Metal matrix-free kernel-vector product (when matrix_free_ is true).
    Tensor matvec_impl(const Tensor& v) const;
    /// Resolves Backend::Auto to a concrete CPU/Metal choice given the current shape.
    Backend effective_backend() const;

    GPHyperparams hp_;
    Backend backend_ = Backend::CPU;
    Solver solver_ = Solver::Cholesky;
    // New-API state: when kernel_ is non-null, fit/predict use the Kernel object hierarchy
    // (allows composition) instead of dispatch_kernel + GPHyperparams.
    std::shared_ptr<Kernel> kernel_;
    std::shared_ptr<MeanFunction> mean_;
    float noise_variance_ = 1e-2f;
    Tensor X_train_;
    Tensor y_train_;        // centered (y - mean) in new-API mode, raw in legacy mode
    Tensor y_train_orig_;   // raw y, only populated in new-API mode (for re-fits during optimize)
    // Cholesky-mode state:
    Tensor L_;
    // CG-mode state. K_y_ holds the materialized kernel matrix when matrix_free_ is false;
    // otherwise it stays empty and matvecs go through the Metal matrix-free kernel.
    Tensor K_y_;
    bool matrix_free_ = false;
    // Common:
    Tensor alpha_;
    float jitter_used_ = 0.0f;
    bool fitted_ = false;
};

}  // namespace lightgp
