#include "rbf_kernel.h"

#include <cassert>
#include <cmath>

#include "../core/dispatch.h"

namespace lightgp {

RBFKernel::RBFKernel(float length_scale, float signal_variance)
    : length_scale_(length_scale), signal_variance_(signal_variance) {}

Tensor RBFKernel::compute(const Tensor& X1, const Tensor& X2, Backend backend) const {
    return dispatch_kernel(X1, X2, length_scale_, signal_variance_,
                           KernelType::RBF, backend);
}

Tensor RBFKernel::compute_diag(const Tensor& X) const {
    // k(x, x) = σ² for RBF (r=0 → exp(0) = 1).
    Tensor d(X.rows(), 1);
    for (std::size_t i = 0; i < X.rows(); ++i) d(i, 0) = signal_variance_;
    return d;
}

std::vector<float> RBFKernel::get_log_params() const {
    return {std::log(length_scale_), std::log(signal_variance_)};
}

void RBFKernel::set_log_params(const std::vector<float>& p) {
    assert(p.size() == 2);
    length_scale_ = std::exp(p[0]);
    signal_variance_ = std::exp(p[1]);
}

}  // namespace lightgp
