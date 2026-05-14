#pragma once

#include <memory>

#include "kernel_base.h"

namespace lightgp {

/// Sum kernel: k(x, x') = k_a(x, x') + k_b(x, x'). Hyperparameters concatenate.
class SumKernel : public Kernel {
public:
    SumKernel(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b);
    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return a_->num_params() + b_->num_params(); }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override { return "(" + a_->name() + " + " + b_->name() + ")"; }

    const std::shared_ptr<Kernel>& a() const { return a_; }
    const std::shared_ptr<Kernel>& b() const { return b_; }

private:
    std::shared_ptr<Kernel> a_, b_;
};

/// Product kernel: k(x, x') = k_a(x, x') * k_b(x, x'). Hyperparameters concatenate.
class ProductKernel : public Kernel {
public:
    ProductKernel(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b);
    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return a_->num_params() + b_->num_params(); }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override { return "(" + a_->name() + " * " + b_->name() + ")"; }

private:
    std::shared_ptr<Kernel> a_, b_;
};

/// Scale kernel: k(x, x') = exp(log_scale) * base(x, x'). Adds one log-hyperparameter
/// in front of the base kernel's params.
class ScaleKernel : public Kernel {
public:
    explicit ScaleKernel(std::shared_ptr<Kernel> base, float scale = 1.0f);
    Tensor compute(const Tensor& X1, const Tensor& X2, Backend backend = Backend::CPU) const override;
    Tensor compute_diag(const Tensor& X) const override;
    int num_params() const override { return 1 + base_->num_params(); }
    std::vector<float> get_log_params() const override;
    void set_log_params(const std::vector<float>& params) override;
    std::string name() const override { return "Scale(" + base_->name() + ")"; }

    float scale() const { return scale_; }

private:
    std::shared_ptr<Kernel> base_;
    float scale_;
};

// Free factory + operator overloads for ergonomic composition:
//   auto k = scale(std::make_shared<RBFKernel>()) + std::make_shared<PeriodicKernel>();
std::shared_ptr<Kernel> operator+(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b);
std::shared_ptr<Kernel> operator*(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b);
std::shared_ptr<Kernel> scale(std::shared_ptr<Kernel> base, float initial_scale = 1.0f);

}  // namespace lightgp
