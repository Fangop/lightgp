// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "cg_cpu.h"

#include <cassert>
#include <cmath>

namespace lightgp {

namespace {

float dot(const Tensor& a, const Tensor& b) {
    const std::size_t n = a.size();
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += a.data()[i] * b.data()[i];
    return s;
}

void matvec(const Tensor& A, const Tensor& x, Tensor& y) {
    const std::size_t n = A.rows();
    for (std::size_t i = 0; i < n; ++i) {
        float s = 0.0f;
        for (std::size_t j = 0; j < n; ++j) s += A(i, j) * x(j, 0);
        y(i, 0) = s;
    }
}

}  // namespace

CGResult cg_solve_cpu(const Tensor& A, const Tensor& b, Tensor& x,
                      float tol, int max_iter) {
    assert(A.rows() == A.cols());
    assert(b.rows() == A.rows() && b.cols() == 1);

    const std::size_t n = A.rows();
    x = Tensor::zeros(n, 1);

    // Jacobi preconditioner M^{-1} = diag(1 / A_ii). Trivial for RBF (K_ii = sf2 + sn2 constant),
    // but generalizes cleanly to other kernels.
    Tensor m_inv(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        const float a_ii = A(i, i);
        m_inv(i, 0) = (a_ii != 0.0f) ? (1.0f / a_ii) : 1.0f;
    }

    Tensor r(n, 1);
    for (std::size_t i = 0; i < n; ++i) r(i, 0) = b(i, 0);  // r = b - A x = b (since x=0)

    const float b_norm = std::sqrt(dot(b, b));
    if (b_norm == 0.0f) {
        return {true, 0, 0.0f};
    }

    Tensor z(n, 1);
    for (std::size_t i = 0; i < n; ++i) z(i, 0) = m_inv(i, 0) * r(i, 0);

    Tensor p = z;
    float rz = dot(r, z);

    Tensor Ap(n, 1);
    CGResult result;
    for (int k = 1; k <= max_iter; ++k) {
        matvec(A, p, Ap);
        const float p_Ap = dot(p, Ap);
        if (p_Ap <= 0.0f) {
            // A is not positive definite along p (numerical breakdown).
            result.iterations = k - 1;
            result.final_residual = std::sqrt(dot(r, r)) / b_norm;
            return result;
        }
        const float alpha = rz / p_Ap;

        for (std::size_t i = 0; i < n; ++i) {
            x(i, 0) += alpha * p(i, 0);
            r(i, 0) -= alpha * Ap(i, 0);
        }

        const float r_norm = std::sqrt(dot(r, r));
        if (r_norm / b_norm < tol) {
            result.converged = true;
            result.iterations = k;
            result.final_residual = r_norm / b_norm;
            return result;
        }

        for (std::size_t i = 0; i < n; ++i) z(i, 0) = m_inv(i, 0) * r(i, 0);
        const float rz_new = dot(r, z);
        const float beta = rz_new / rz;
        for (std::size_t i = 0; i < n; ++i) p(i, 0) = z(i, 0) + beta * p(i, 0);
        rz = rz_new;
    }

    result.iterations = max_iter;
    result.final_residual = std::sqrt(dot(r, r)) / b_norm;
    return result;
}

CGResult cg_solve_matvec(const MatvecFn& matvec,
                         std::size_t n,
                         const Tensor& b, Tensor& x,
                         float tol, int max_iter) {
    assert(b.rows() == n && b.cols() == 1);

    x = Tensor::zeros(n, 1);
    Tensor r(n, 1);
    for (std::size_t i = 0; i < n; ++i) r(i, 0) = b(i, 0);

    const float b_norm = std::sqrt(dot(b, b));
    if (b_norm == 0.0f) return {true, 0, 0.0f};

    Tensor p = r;
    float rr = dot(r, r);
    CGResult result;
    for (int k = 1; k <= max_iter; ++k) {
        Tensor Ap = matvec(p);
        const float p_Ap = dot(p, Ap);
        if (p_Ap <= 0.0f) {
            result.iterations = k - 1;
            result.final_residual = std::sqrt(dot(r, r)) / b_norm;
            return result;
        }
        const float alpha = rr / p_Ap;
        for (std::size_t i = 0; i < n; ++i) {
            x(i, 0) += alpha * p(i, 0);
            r(i, 0) -= alpha * Ap(i, 0);
        }
        const float r_norm = std::sqrt(dot(r, r));
        if (r_norm / b_norm < tol) {
            result.converged = true;
            result.iterations = k;
            result.final_residual = r_norm / b_norm;
            return result;
        }
        const float rr_new = dot(r, r);
        const float beta = rr_new / rr;
        for (std::size_t i = 0; i < n; ++i) p(i, 0) = r(i, 0) + beta * p(i, 0);
        rr = rr_new;
    }
    result.iterations = max_iter;
    result.final_residual = std::sqrt(dot(r, r)) / b_norm;
    return result;
}

}  // namespace lightgp
