#include "test_utils.h"

#include "../core/tensor.h"

namespace lightgp {

void run_tensor_tests() {
    std::printf("[tensor] starting...\n");

    Tensor z = Tensor::zeros(2, 3);
    LIGHTGP_CHECK(z.rows() == 2 && z.cols() == 3 && z.size() == 6);
    for (std::size_t i = 0; i < z.size(); ++i) LIGHTGP_CHECK(z.data()[i] == 0.0f);

    Tensor o = Tensor::ones(3, 2);
    LIGHTGP_CHECK(o.rows() == 3 && o.cols() == 2);
    for (std::size_t i = 0; i < o.size(); ++i) LIGHTGP_CHECK(o.data()[i] == 1.0f);

    Tensor I = Tensor::eye(3);
    LIGHTGP_CHECK_NEAR(I(0, 0), 1.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(I(1, 1), 1.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(I(2, 2), 1.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(I(0, 1), 0.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(I(2, 0), 0.0f, 0.0f);

    // transpose
    Tensor A(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor At = A.transpose();
    LIGHTGP_CHECK(At.rows() == 3 && At.cols() == 2);
    LIGHTGP_CHECK_NEAR(At(0, 0), 1.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(At(1, 0), 2.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(At(2, 1), 6.0f, 0.0f);

    // matmul: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    Tensor B(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor C(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor D = B.matmul(C);
    LIGHTGP_CHECK_NEAR(D(0, 0), 19.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(D(0, 1), 22.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(D(1, 0), 43.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(D(1, 1), 50.0f, 1e-5f);

    // I * X = X
    Tensor X(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor IX = Tensor::eye(2).matmul(X);
    for (std::size_t i = 0; i < X.size(); ++i) {
        LIGHTGP_CHECK_NEAR(IX.data()[i], X.data()[i], 1e-6f);
    }

    // add / sub / scalar_mul
    Tensor E = B.add(C);
    LIGHTGP_CHECK_NEAR(E(0, 0), 6.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(E(1, 1), 12.0f, 1e-5f);
    Tensor F = C.sub(B);
    LIGHTGP_CHECK_NEAR(F(0, 0), 4.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(F(1, 1), 4.0f, 1e-5f);
    Tensor G = B.scalar_mul(2.0f);
    LIGHTGP_CHECK_NEAR(G(1, 0), 6.0f, 1e-5f);
    LIGHTGP_CHECK_NEAR(G(0, 1), 4.0f, 1e-5f);

    // add_jitter (square)
    Tensor H = Tensor::zeros(3, 3);
    H.add_jitter(0.5f);
    LIGHTGP_CHECK_NEAR(H(0, 0), 0.5f, 0.0f);
    LIGHTGP_CHECK_NEAR(H(1, 1), 0.5f, 0.0f);
    LIGHTGP_CHECK_NEAR(H(2, 2), 0.5f, 0.0f);
    LIGHTGP_CHECK_NEAR(H(0, 1), 0.0f, 0.0f);
    LIGHTGP_CHECK_NEAR(H(2, 0), 0.0f, 0.0f);

    // randn determinism
    Tensor R1 = Tensor::randn(4, 4, 42);
    Tensor R2 = Tensor::randn(4, 4, 42);
    for (std::size_t i = 0; i < R1.size(); ++i) {
        LIGHTGP_CHECK_NEAR(R1.data()[i], R2.data()[i], 0.0f);
    }
}

}  // namespace lightgp
