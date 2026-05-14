#include "mean.h"

#include <cassert>

namespace lightgp {

Tensor ZeroMean::compute(const Tensor& X) const {
    return Tensor::zeros(X.rows(), 1);
}

Tensor ConstantMean::compute(const Tensor& X) const {
    Tensor m(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) m(i, 0) = c_;
    return m;
}

void ConstantMean::set_params(const std::vector<float>& p) {
    assert(p.size() == 1);
    c_ = p[0];
}

LinearMean::LinearMean(std::size_t input_dim) : w_(input_dim, 0.0f), b_(0.0f) {}

LinearMean::LinearMean(std::vector<float> w, float b) : w_(std::move(w)), b_(b) {}

Tensor LinearMean::compute(const Tensor& X) const {
    assert(X.cols() == w_.size());
    Tensor m(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        float s = b_;
        for (std::size_t d = 0; d < X.cols(); ++d) s += X(i, d) * w_[d];
        m(i, 0) = s;
    }
    return m;
}

std::vector<float> LinearMean::get_params() const {
    std::vector<float> p = w_;
    p.push_back(b_);
    return p;
}

void LinearMean::set_params(const std::vector<float>& p) {
    assert(p.size() == w_.size() + 1);
    for (std::size_t i = 0; i < w_.size(); ++i) w_[i] = p[i];
    b_ = p.back();
}

}  // namespace lightgp
