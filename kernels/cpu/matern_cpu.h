#pragma once

#include "../../core/kernel.h"
#include "../../core/tensor.h"

namespace lightgp {

/// Matern-1/2, 3/2, or 5/2 kernel matrix between X1 (N x D) and X2 (M x D).
/// Selects the variant via `type` (must be one of the Matern values from KernelType).
Tensor matern_kernel_cpu(const Tensor& X1, const Tensor& X2,
                         float length_scale, float signal_variance,
                         KernelType type);

}  // namespace lightgp
