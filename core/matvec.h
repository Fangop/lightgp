#pragma once

#include <functional>

#include "tensor.h"

namespace lightgp {

/// Matrix-free matvec callback: given v (N x 1), return A * v (N x 1).
/// Used by CG and SLQ to operate on operators without materializing the full matrix.
using MatvecFn = std::function<Tensor(const Tensor&)>;

}  // namespace lightgp
