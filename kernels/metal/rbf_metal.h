#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_METAL

/// RBF kernel matrix computed on the Metal GPU; mirrors rbf_kernel_cpu element-wise (atol ~1e-5).
/// Falls back to CPU if Metal initialization failed; check MetalContext::available() if you need to detect.
Tensor rbf_kernel_metal(const Tensor& X1, const Tensor& X2,
                        float length_scale, float signal_variance);

#endif  // LIGHTGP_HAS_METAL

}  // namespace lightgp
