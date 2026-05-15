#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// C = A * B via cuBLAS sgemm on the singleton CUDA stream.
/// A is M x K, B is K x N (row-major). Returns C as an M x N row-major Tensor.
/// Allocates device buffers internally and copies the result back to host on completion.
Tensor gemm_cuda(const Tensor& A, const Tensor& B);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
