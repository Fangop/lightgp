// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include "../../core/tensor.h"

namespace lightgp {

#ifdef LIGHTGP_HAS_CUDA

/// C = A * B via cuBLAS sgemm on the singleton CUDA stream.
/// A is M x K, B is K x N (row-major). Returns C as an M x N row-major Tensor.
/// Allocates device buffers internally and copies the result back to host on completion.
Tensor gemm_cuda(const Tensor& A, const Tensor& B);

/// C = A * A^T via cuBLAS sgemm on the singleton CUDA stream.
/// A is M x K (row-major); returns C as an M x M row-major Tensor. Uses op_B = T
/// inside cuBLAS so the M x K input is reused for the second factor without an
/// explicit transpose copy.
Tensor gemm_AAt_cuda(const Tensor& A);

/// C = A^T * A via cuBLAS sgemm on the singleton CUDA stream.
/// A is N x M (row-major); returns C as an M x M row-major Tensor. Uses op_A = N,
/// op_B = T on the same buffer so no explicit transpose is materialised.
///
/// Sparse VFE Σ assembly is the canonical caller — `K_fu_scaled.transpose().matmul(
/// K_fu_scaled)` at N=50k / M=200 was 481 ms via OpenBLAS (memory-bandwidth bound on
/// the 40 MB transpose buffer); this CUDA path lands under 10 ms including the
/// 40 MB H→D upload of A.
Tensor gemm_AtA_cuda(const Tensor& A);

#endif  // LIGHTGP_HAS_CUDA

}  // namespace lightgp
