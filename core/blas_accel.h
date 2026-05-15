#pragma once

#include "tensor.h"

namespace lightgp {

#if defined(LIGHTGP_HAS_ACCELERATE) || defined(LIGHTGP_HAS_OPENBLAS)

/// Tuned BLAS / LAPACK wrappers for the heavy CPU linear algebra.
/// All functions take row-major Tensors and route through the platform's CBLAS / LAPACK
/// (Apple Accelerate on macOS, OpenBLAS + LAPACK on Linux) using either CblasRowMajor
/// (BLAS) or the row-major-on-LAPACK transpose trick (LAPACK Fortran routines are
/// column-major; calling them with `uplo='U'` on a row-major SPD matrix produces the
/// column-major upper factor, which is the row-major lower factor we want — minus a
/// copy step in the strict-upper triangle).
///
/// Function names retain the `_accelerate` suffix for historical reasons; on Linux they
/// dispatch to OpenBLAS / netlib LAPACK with identical row-major semantics.

/// C = A * B via cblas_sgemm.
void gemm_accelerate(const Tensor& A, const Tensor& B, Tensor& C);

/// In-place lower Cholesky factorization of row-major SPD K via LAPACK spotrf.
/// On entry K is the SPD matrix; on exit the lower triangle holds L (strict upper is zeroed).
/// Returns false on non-positive-definite input.
bool cholesky_accelerate(Tensor& K);

/// Solve L X = B (forward substitution) in place: B is overwritten with X.
void trsm_lower_no_trans_accelerate(const Tensor& L, Tensor& B);

/// Solve L^T X = B (back substitution) in place: B is overwritten with X.
void trsm_lower_trans_accelerate(const Tensor& L, Tensor& B);

#endif  // LIGHTGP_HAS_ACCELERATE || LIGHTGP_HAS_OPENBLAS

}  // namespace lightgp
