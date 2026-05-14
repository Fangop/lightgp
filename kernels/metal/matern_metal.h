#pragma once

#include "../../core/kernel.h"
#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_METAL

/// Matern kernel matrix on Metal. `type` must be one of the Matern values.
/// Falls back to matern_kernel_cpu if Metal is unavailable.
Tensor matern_kernel_metal(const Tensor& X1, const Tensor& X2,
                           float length_scale, float signal_variance,
                           KernelType type);

#endif  // LIGHTGP_HAS_METAL

}  // namespace lightgp
