#include "slq_cpu.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace lightgp {

namespace {

float dot_vec(const Tensor& a, const Tensor& b) {
    float s = 0.0f;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) s += a.data()[i] * b.data()[i];
    return s;
}

// Lanczos with full reorthogonalization. Outputs alpha (diag) and beta (sub-diag) of T.
void lanczos_full_reorth(const MatvecFn& matvec,
                         std::size_t n,
                         const Tensor& v0,
                         int max_iters,
                         std::vector<float>& alpha,
                         std::vector<float>& beta) {
    Tensor q(n, 1);
    float n0 = std::sqrt(dot_vec(v0, v0));
    if (n0 < 1e-12f) return;
    for (std::size_t i = 0; i < n; ++i) q(i, 0) = v0(i, 0) / n0;

    std::vector<Tensor> Q;
    Q.reserve(max_iters);
    Q.push_back(q);

    for (int j = 0; j < max_iters; ++j) {
        Tensor w = matvec(Q[j]);

        const float a = dot_vec(Q[j], w);
        alpha.push_back(a);

        for (std::size_t i = 0; i < n; ++i) w(i, 0) -= a * Q[j](i, 0);
        if (j > 0) {
            const float bp = beta.back();
            for (std::size_t i = 0; i < n; ++i) w(i, 0) -= bp * Q[j - 1](i, 0);
        }
        // Full reorthogonalization against all prior basis vectors.
        for (int k = 0; k <= j; ++k) {
            const float c = dot_vec(Q[k], w);
            for (std::size_t i = 0; i < n; ++i) w(i, 0) -= c * Q[k](i, 0);
        }

        const float bj = std::sqrt(dot_vec(w, w));
        if (bj < 1e-10f) break;  // Lanczos converged on invariant subspace

        if (j < max_iters - 1) {
            beta.push_back(bj);
            Tensor q_new(n, 1);
            for (std::size_t i = 0; i < n; ++i) q_new(i, 0) = w(i, 0) / bj;
            Q.push_back(q_new);
        }
    }
}

// Jacobi eigendecomposition of a symmetric matrix in place.
// On exit, the diagonal of A holds the eigenvalues; U holds eigenvectors as columns.
void jacobi_eigh(Tensor& A, Tensor& U) {
    const std::size_t n = A.rows();
    U = Tensor::eye(n);
    constexpr int max_sweeps = 80;
    constexpr float tol = 1e-7f;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float off2 = 0.0f;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q)
                off2 += A(p, q) * A(p, q);
        if (off2 < tol * tol) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const float apq = A(p, q);
                if (std::fabs(apq) < 1e-14f) continue;

                // Rotation J = [[c, s], [-s, c]]. For the corresponding diagonal updates
                // (c² App - 2sc Apq + s² Aqq, etc.) to zero the off-diagonal,
                // tan(2θ) = 2 Apq / (Aqq - App).
                float c, s;
                const float diff = A(q, q) - A(p, p);
                if (std::fabs(diff) < 1e-14f) {
                    const float pi4 = 0.785398163f;
                    c = std::cos(pi4);
                    s = std::sin(pi4);
                } else {
                    const float theta = 0.5f * std::atan2(2.0f * apq, diff);
                    c = std::cos(theta);
                    s = std::sin(theta);
                }

                const float app = A(p, p);
                const float aqq = A(q, q);
                A(p, p) = c * c * app - 2.0f * s * c * apq + s * s * aqq;
                A(q, q) = s * s * app + 2.0f * s * c * apq + c * c * aqq;
                A(p, q) = 0.0f;
                A(q, p) = 0.0f;
                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    const float aip = A(i, p);
                    const float aiq = A(i, q);
                    A(i, p) = c * aip - s * aiq;
                    A(p, i) = A(i, p);
                    A(i, q) = s * aip + c * aiq;
                    A(q, i) = A(i, q);
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const float uip = U(i, p);
                    const float uiq = U(i, q);
                    U(i, p) = c * uip - s * uiq;
                    U(i, q) = s * uip + c * uiq;
                }
            }
        }
    }
}

}  // namespace

float slq_log_det_cpu(const MatvecFn& matvec,
                      std::size_t n,
                      int n_probes,
                      int n_iters,
                      std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    double sum = 0.0;
    int valid = 0;

    for (int p = 0; p < n_probes; ++p) {
        Tensor z(n, 1);
        for (std::size_t i = 0; i < n; ++i) z(i, 0) = coin(rng) ? 1.0f : -1.0f;

        std::vector<float> alpha, beta;
        lanczos_full_reorth(matvec, n, z, n_iters, alpha, beta);
        if (alpha.empty()) continue;

        const std::size_t k = alpha.size();
        Tensor T = Tensor::zeros(k, k);
        for (std::size_t i = 0; i < k; ++i) T(i, i) = alpha[i];
        for (std::size_t i = 0; i + 1 < k; ++i) {
            T(i, i + 1) = beta[i];
            T(i + 1, i) = beta[i];
        }

        Tensor U;
        jacobi_eigh(T, U);

        // ||z||^2 = n for Rademacher; quadrature: z^T log(A) z ≈ n * Σ U[0, i]² log(D_ii).
        float contrib = 0.0f;
        for (std::size_t i = 0; i < k; ++i) {
            const float d = T(i, i);
            if (d > 0.0f) {
                contrib += U(0, i) * U(0, i) * std::log(d);
            }
        }
        sum += static_cast<double>(n) * contrib;
        valid++;
    }

    if (valid == 0) return 0.0f;
    return static_cast<float>(sum / static_cast<double>(valid));
}

float slq_log_det_cpu(const Tensor& A, int n_probes, int n_iters, std::uint64_t seed) {
    return slq_log_det_cpu(
        [&A](const Tensor& v) { return A.matmul(v); },
        A.rows(), n_probes, n_iters, seed);
}

}  // namespace lightgp
