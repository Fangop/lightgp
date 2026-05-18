// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "composite_kernel.h"

#include <cassert>
#include <cmath>

namespace lightgp {

// ---- SumKernel ----
SumKernel::SumKernel(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b)
    : a_(std::move(a)), b_(std::move(b)) {}

Tensor SumKernel::compute(const Tensor& X1, const Tensor& X2, Backend backend) const {
    Tensor Ka = a_->compute(X1, X2, backend);
    Tensor Kb = b_->compute(X1, X2, backend);
    return Ka.add(Kb);
}

Tensor SumKernel::compute_diag(const Tensor& X) const {
    Tensor da = a_->compute_diag(X);
    Tensor db = b_->compute_diag(X);
    return da.add(db);
}

std::vector<float> SumKernel::get_log_params() const {
    auto pa = a_->get_log_params();
    auto pb = b_->get_log_params();
    pa.insert(pa.end(), pb.begin(), pb.end());
    return pa;
}

void SumKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == static_cast<std::size_t>(num_params()));
    const std::size_t na = a_->num_params();
    a_->set_log_params(std::vector<float>(p.begin(), p.begin() + na));
    b_->set_log_params(std::vector<float>(p.begin() + na, p.end()));
}

// ---- ProductKernel ----
ProductKernel::ProductKernel(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b)
    : a_(std::move(a)), b_(std::move(b)) {}

Tensor ProductKernel::compute(const Tensor& X1, const Tensor& X2, Backend backend) const {
    Tensor Ka = a_->compute(X1, X2, backend);
    Tensor Kb = b_->compute(X1, X2, backend);
    // Element-wise multiply.
    Tensor K(Ka.rows(), Ka.cols());
    for (std::size_t i = 0; i < K.size(); ++i) K.data()[i] = Ka.data()[i] * Kb.data()[i];
    return K;
}

Tensor ProductKernel::compute_diag(const Tensor& X) const {
    Tensor da = a_->compute_diag(X);
    Tensor db = b_->compute_diag(X);
    Tensor d(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) d(i, 0) = da(i, 0) * db(i, 0);
    return d;
}

std::vector<float> ProductKernel::get_log_params() const {
    auto pa = a_->get_log_params();
    auto pb = b_->get_log_params();
    pa.insert(pa.end(), pb.begin(), pb.end());
    return pa;
}

void ProductKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == static_cast<std::size_t>(num_params()));
    const std::size_t na = a_->num_params();
    a_->set_log_params(std::vector<float>(p.begin(), p.begin() + na));
    b_->set_log_params(std::vector<float>(p.begin() + na, p.end()));
}

// ---- ScaleKernel ----
ScaleKernel::ScaleKernel(std::shared_ptr<Kernel> base, float scale)
    : base_(std::move(base)), scale_(scale) {}

Tensor ScaleKernel::compute(const Tensor& X1, const Tensor& X2, Backend backend) const {
    Tensor K = base_->compute(X1, X2, backend);
    return K.scalar_mul(scale_);
}

Tensor ScaleKernel::compute_diag(const Tensor& X) const {
    Tensor d = base_->compute_diag(X);
    return d.scalar_mul(scale_);
}

std::vector<float> ScaleKernel::get_log_params() const {
    std::vector<float> p = {std::log(scale_)};
    auto pb = base_->get_log_params();
    p.insert(p.end(), pb.begin(), pb.end());
    return p;
}

void ScaleKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == static_cast<std::size_t>(num_params()));
    scale_ = std::exp(p[0]);
    base_->set_log_params(std::vector<float>(p.begin() + 1, p.end()));
}

// ---- Free operators ----
std::shared_ptr<Kernel> operator+(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b) {
    return std::make_shared<SumKernel>(std::move(a), std::move(b));
}
std::shared_ptr<Kernel> operator*(std::shared_ptr<Kernel> a, std::shared_ptr<Kernel> b) {
    return std::make_shared<ProductKernel>(std::move(a), std::move(b));
}
std::shared_ptr<Kernel> scale(std::shared_ptr<Kernel> base, float initial_scale) {
    return std::make_shared<ScaleKernel>(std::move(base), initial_scale);
}

}  // namespace lightgp
