#include "cholesky_solve_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../../kernels/cuda/cuda_context.h"
#include "../cpu/cholesky_cpu.h"

namespace lightgp {

namespace {

/// Run two strsm calls (forward + back) for full Cholesky solve, or just the forward
/// call when `back == false`. Acts in-place on dB which holds the row-major
/// (N x m) right-hand side. The buffer's col-major view is an (m x N) matrix = B^T;
/// after the calls it holds X^T (col-major) which equals X in row-major.
///
/// L is N x N lower triangular row-major. Its col-major view is L^T (upper triangular),
/// hence the side=RIGHT + UPPER trick:
///   forward: L X = B   ⇔   X^T L^T = B^T   ⇔   X^T L_c = B^T   →   op=N
///   back:    L^T X = B ⇔   X^T L   = B^T   ⇔   X^T L_c^T = B^T →   op=T
void strsm_solve(cublasHandle_t blas, const float* dL, float* dB, int n, int m, bool back) {
    const float alpha = 1.0f;
    cublasStrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                back ? CUBLAS_OP_T : CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                m, n, &alpha, dL, n, dB, m);
}

}  // namespace

Tensor cholesky_solve_cuda(const Tensor& L, const Tensor& b) {
    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());

    auto& ctx = CudaContext::instance();
    if (!ctx.available()) return cholesky_solve(L, b);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(ctx.cublas());

    const int n = static_cast<int>(L.rows());
    const int m = static_cast<int>(b.cols());

    float *dL = nullptr, *dB = nullptr;
    cudaMalloc(&dL, sizeof(float) * n * n);
    cudaMalloc(&dB, sizeof(float) * n * m);

    cudaMemcpyAsync(dL, L.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, b.data(), sizeof(float) * n * m, cudaMemcpyHostToDevice, stream);

    strsm_solve(blas, dL, dB, n, m, /*back=*/false);
    strsm_solve(blas, dL, dB, n, m, /*back=*/true);

    Tensor x(n, m);
    cudaMemcpyAsync(x.data(), dB, sizeof(float) * n * m, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dL);
    cudaFree(dB);
    return x;
}

Tensor forward_solve_cuda(const Tensor& L, const Tensor& b) {
    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());

    auto& ctx = CudaContext::instance();
    if (!ctx.available()) {
        // CPU forward-only path mirrors solvers/cpu/cholesky_cpu.cpp's reference branch.
        const std::size_t n = L.rows();
        const std::size_t m = b.cols();
        Tensor Y(n, m);
        for (std::size_t col = 0; col < m; ++col) {
            for (std::size_t i = 0; i < n; ++i) {
                float s = b(i, col);
                for (std::size_t j = 0; j < i; ++j) s -= L(i, j) * Y(j, col);
                Y(i, col) = s / L(i, i);
            }
        }
        return Y;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(ctx.cublas());

    const int n = static_cast<int>(L.rows());
    const int m = static_cast<int>(b.cols());

    float *dL = nullptr, *dB = nullptr;
    cudaMalloc(&dL, sizeof(float) * n * n);
    cudaMalloc(&dB, sizeof(float) * n * m);

    cudaMemcpyAsync(dL, L.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, b.data(), sizeof(float) * n * m, cudaMemcpyHostToDevice, stream);

    strsm_solve(blas, dL, dB, n, m, /*back=*/false);

    Tensor Y(n, m);
    cudaMemcpyAsync(Y.data(), dB, sizeof(float) * n * m, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dL);
    cudaFree(dB);
    return Y;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
