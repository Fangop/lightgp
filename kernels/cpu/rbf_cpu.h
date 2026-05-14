#pragma once

#include "../../core/tensor.h"

namespace lightgp {

/// RBF / squared-exponential covariance K(i,j) = sf2 * exp(-0.5 * ||x_i - x_j||^2 / l^2).
Tensor rbf_kernel_cpu(const Tensor& X1, const Tensor& X2,
                      float length_scale, float signal_variance);

/// Gradients of the RBF kernel matrix w.r.t. log(length_scale) and log(signal_variance).
/// Outputs are written into grad_log_l and grad_log_sf2 (both shaped N x M).
void rbf_kernel_gradients_cpu(const Tensor& X1, const Tensor& X2,
                              float length_scale, float signal_variance,
                              Tensor& grad_log_l, Tensor& grad_log_sf2);

}  // namespace lightgp
