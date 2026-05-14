#include "cholesky_cpu.h"

#include <cassert>
#include <cmath>

#ifdef LIGHTGP_HAS_ACCELERATE
#include "../../core/blas_accel.h"
#endif

namespace lightgp {

bool cholesky_cpu(const Tensor& K, Tensor& L) {
    assert(K.rows() == K.cols());
#ifdef LIGHTGP_HAS_ACCELERATE
    // LAPACK spotrf via Accelerate: column-major LAPACK + row-major data → lower factor in-place.
    L = K;
    return cholesky_accelerate(L);
#else
    const std::size_t n = K.rows();
    L = Tensor::zeros(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            float s = K(i, j);
            for (std::size_t k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            if (i == j) {
                if (s <= 0.0f) return false;
                L(i, i) = std::sqrt(s);
            } else {
                L(i, j) = s / L(j, j);
            }
        }
    }
    return true;
#endif
}

float log_det_from_cholesky(const Tensor& L) {
    assert(L.rows() == L.cols());
    float s = 0.0f;
    for (std::size_t i = 0; i < L.rows(); ++i) s += std::log(L(i, i));
    return 2.0f * s;
}

Tensor cholesky_solve(const Tensor& L, const Tensor& b) {
    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());
#ifdef LIGHTGP_HAS_ACCELERATE
    Tensor x = b;
    trsm_lower_no_trans_accelerate(L, x);  // L y = b → x holds y
    trsm_lower_trans_accelerate(L, x);     // L^T x = y → x holds final
    return x;
#else
    const std::size_t n = L.rows();
    const std::size_t k = b.cols();
    Tensor y(n, k);
    for (std::size_t col = 0; col < k; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            float s = b(i, col);
            for (std::size_t j = 0; j < i; ++j) s -= L(i, j) * y(j, col);
            y(i, col) = s / L(i, i);
        }
    }
    Tensor x(n, k);
    for (std::size_t col = 0; col < k; ++col) {
        for (std::size_t ii = n; ii-- > 0; ) {
            float s = y(ii, col);
            for (std::size_t j = ii + 1; j < n; ++j) s -= L(j, ii) * x(j, col);
            x(ii, col) = s / L(ii, ii);
        }
    }
    return x;
#endif
}

bool cholesky_with_jitter(const Tensor& K, Tensor& L, float& jitter_used) {
    assert(K.rows() == K.cols());
    constexpr float jitters[] = {1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.0f};
    for (float j : jitters) {
        Tensor A = K;
        A.add_jitter(j);
        if (cholesky_cpu(A, L)) {
            jitter_used = j;
            return true;
        }
    }
    return false;
}

}  // namespace lightgp
