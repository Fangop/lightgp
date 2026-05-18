// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_METAL

/// Compute C = A * B on the Metal GPU. Falls back to CPU matmul if Metal is unavailable.
Tensor gemm_metal(const Tensor& A, const Tensor& B);

#endif  // LIGHTGP_HAS_METAL

}  // namespace lightgp
