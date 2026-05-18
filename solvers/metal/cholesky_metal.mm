// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#ifdef LIGHTGP_HAS_METAL

#include "cholesky_metal.h"

#include <algorithm>
#include <cassert>

#include "../cpu/cholesky_cpu.h"
#include "../../kernels/metal/gemm_metal.h"
#include "../../kernels/metal/metal_context.h"

namespace lightgp {

namespace {

Tensor extract_block(const Tensor& A, std::size_t r0, std::size_t c0,
                     std::size_t nr, std::size_t nc) {
    Tensor B(nr, nc);
    for (std::size_t i = 0; i < nr; ++i)
        for (std::size_t j = 0; j < nc; ++j)
            B(i, j) = A(r0 + i, c0 + j);
    return B;
}

void write_block(Tensor& A, std::size_t r0, std::size_t c0, const Tensor& B) {
    for (std::size_t i = 0; i < B.rows(); ++i)
        for (std::size_t j = 0; j < B.cols(); ++j)
            A(r0 + i, c0 + j) = B(i, j);
}

// Solve X * L^T = B for X where L is lower triangular (b x b), B and X are (m x b).
// Row by row: each row of X satisfies L * X[i,:]^T = B[i,:]^T (forward substitution).
Tensor trsm_lower_transpose(const Tensor& L, const Tensor& B) {
    const std::size_t m = B.rows();
    const std::size_t b = L.rows();
    Tensor X(m, b);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
            float s = B(i, j);
            for (std::size_t k = 0; k < j; ++k) s -= L(j, k) * X(i, k);
            X(i, j) = s / L(j, j);
        }
    }
    return X;
}

}  // namespace

bool cholesky_metal(const Tensor& K, Tensor& L, int block_size) {
    assert(K.rows() == K.cols());
    const std::size_t N = K.rows();
    if (N <= static_cast<std::size_t>(block_size)
        || !MetalContext::instance().available()) {
        return cholesky_cpu(K, L);
    }

    Tensor A = K;  // mutable working copy: lower-trail block stores running Schur complement
    L = Tensor::zeros(N, N);

    for (std::size_t k = 0; k < N; k += static_cast<std::size_t>(block_size)) {
        const std::size_t b = std::min<std::size_t>(block_size, N - k);

        // 1. Factorize the b x b diagonal block on CPU.
        Tensor K_diag = extract_block(A, k, k, b, b);
        Tensor L_diag;
        if (!cholesky_cpu(K_diag, L_diag)) return false;
        write_block(L, k, k, L_diag);

        if (k + b >= N) break;

        // 2. TRSM: solve L_below * L_diag^T = A[k+b:, k:k+b].
        const std::size_t trail = N - k - b;
        Tensor A_below = extract_block(A, k + b, k, trail, b);
        Tensor L_below = trsm_lower_transpose(L_diag, A_below);
        write_block(L, k + b, k, L_below);

        // 3. Trailing Schur update: A[k+b:, k+b:] -= L_below * L_below^T on Metal.
        Tensor L_below_T = L_below.transpose();
        Tensor update = gemm_metal(L_below, L_below_T);
        for (std::size_t i = 0; i < trail; ++i) {
            for (std::size_t j = 0; j < trail; ++j) {
                A(k + b + i, k + b + j) -= update(i, j);
            }
        }
    }
    return true;
}

bool cholesky_metal_with_jitter(const Tensor& K, Tensor& L, float& jitter_used,
                                int block_size) {
    assert(K.rows() == K.cols());
    constexpr float jitters[] = {1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.0f};
    for (float j : jitters) {
        Tensor A = K;
        A.add_jitter(j);
        if (cholesky_metal(A, L, block_size)) {
            jitter_used = j;
            return true;
        }
    }
    return false;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
