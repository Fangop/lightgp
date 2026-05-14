#pragma once

#include "../../core/matvec.h"
#include "../../core/tensor.h"

namespace lightgp {

/// Result of a CG solve.
struct CGResult {
    bool converged = false;
    int iterations = 0;
    float final_residual = 0.0f;  // ||r|| / ||b||
};

/// Solve A x = b for symmetric positive-definite A using conjugate gradients
/// with Jacobi (diagonal) preconditioning. x is resized to (A.rows(), 1).
/// b must be (N, 1). Stops when ||r|| / ||b|| < tol or after max_iter iterations.
CGResult cg_solve_cpu(const Tensor& A, const Tensor& b, Tensor& x,
                      float tol = 1e-6f, int max_iter = 1000);

/// Matrix-free CG: caller supplies the matvec callback. No preconditioner
/// (Jacobi preconditioning needs diag(A) which isn't always cheap to extract).
/// Use when A is too large to materialize but supports an efficient A*v.
CGResult cg_solve_matvec(const MatvecFn& matvec,
                         std::size_t n,
                         const Tensor& b, Tensor& x,
                         float tol = 1e-6f, int max_iter = 1000);

}  // namespace lightgp
