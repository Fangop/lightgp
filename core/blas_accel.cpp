// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "blas_accel.h"

#if defined(LIGHTGP_HAS_ACCELERATE) || defined(LIGHTGP_HAS_OPENBLAS)

#include <cassert>

#if defined(LIGHTGP_HAS_ACCELERATE)
// Use the SDK's declarations (LAPACK ints are __LAPACK_int = long on recent macOS SDKs).
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
namespace lightgp { using lapack_int = __LAPACK_int; }
#else  // LIGHTGP_HAS_OPENBLAS
// Debian's /usr/include/x86_64-linux-gnu/cblas.h does not wrap declarations
// in extern "C", while OpenBLAS's manylinux/CentOS cblas.h does. Force C
// linkage on the include so we link against C symbols on both layouts.
extern "C" {
#include <cblas.h>
}
namespace lightgp { using lapack_int = int; }
// LAPACK Fortran symbol — declared here so we don't depend on the optional lapacke headers.
extern "C" void spotrf_(const char* uplo, const lightgp::lapack_int* n,
                        float* a, const lightgp::lapack_int* lda,
                        lightgp::lapack_int* info);
#endif

namespace lightgp {

void gemm_accelerate(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.cols() == B.rows());
    const int M = static_cast<int>(A.rows());
    const int N = static_cast<int>(B.cols());
    const int K = static_cast<int>(A.cols());
    if (C.rows() != static_cast<std::size_t>(M) || C.cols() != static_cast<std::size_t>(N)) {
        C = Tensor(M, N);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A.data(), K,
                B.data(), N,
                0.0f, C.data(), N);
}

bool cholesky_accelerate(Tensor& K) {
    assert(K.rows() == K.cols());
    lapack_int n = static_cast<lapack_int>(K.rows());
    lapack_int info = 0;
    // Row-major SPD interpreted as column-major is its own transpose (= itself, symmetric).
    // spotrf('U', ...) computes upper R s.t. A = R^T R in LAPACK's view; in row-major
    // memory the result lands in the lower triangle and equals our desired L.
    char uplo = 'U';
    spotrf_(&uplo, &n, K.data(), &n, &info);
    if (info != 0) return false;
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = i + 1; j < K.cols(); ++j) {
            K(i, j) = 0.0f;
        }
    }
    return true;
}

void trsm_lower_no_trans_accelerate(const Tensor& L, Tensor& B) {
    assert(L.rows() == L.cols());
    assert(L.rows() == B.rows());
    const int m = static_cast<int>(B.rows());
    const int n = static_cast<int>(B.cols());
    cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                m, n,
                1.0f, L.data(), static_cast<int>(L.cols()),
                B.data(), n);
}

void trsm_lower_trans_accelerate(const Tensor& L, Tensor& B) {
    assert(L.rows() == L.cols());
    assert(L.rows() == B.rows());
    const int m = static_cast<int>(B.rows());
    const int n = static_cast<int>(B.cols());
    cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                m, n,
                1.0f, L.data(), static_cast<int>(L.cols()),
                B.data(), n);
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_ACCELERATE || LIGHTGP_HAS_OPENBLAS
