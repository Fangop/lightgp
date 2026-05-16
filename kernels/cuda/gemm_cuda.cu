#include "gemm_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cuda_context.h"

namespace lightgp {

Tensor gemm_cuda(const Tensor& A, const Tensor& B) {
    assert(A.cols() == B.rows());
    auto& ctx = CudaContext::instance();
    if (!ctx.available()) {
        // Caller (dispatch.cpp) should have routed around an unavailable device,
        // but be defensive: fall back to host matmul.
        return A.matmul(B);
    }
    const int M = static_cast<int>(A.rows());
    const int N = static_cast<int>(B.cols());
    const int K = static_cast<int>(A.cols());

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(ctx.cublas());

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(float) * M * K);
    cudaMalloc(&dB, sizeof(float) * K * N);
    cudaMalloc(&dC, sizeof(float) * M * N);

    cudaMemcpyAsync(dA, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice, stream);

    // Row-major C = A * B using column-major cuBLAS: compute C^T = B^T * A^T.
    // The buffers we hold (row-major A,B,C) are exactly B^T, A^T, C^T in col-major view.
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, dB, N,
                dA, K,
                &beta, dC, N);

    Tensor C(M, N);
    cudaMemcpyAsync(C.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return C;
}

Tensor gemm_AAt_cuda(const Tensor& A) {
    auto& ctx = CudaContext::instance();
    if (!ctx.available()) {
        // Host fallback: explicit transpose + matmul. Caller is expected to dispatch
        // around an unavailable device, so we make this path correct rather than fast.
        return A.matmul(A.transpose());
    }
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(ctx.cublas());

    float *dA = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(float) * M * K);
    cudaMalloc(&dC, sizeof(float) * M * M);
    cudaMemcpyAsync(dA, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice, stream);

    // Row-major C (M x M) = A A^T with A shape (M, K).
    // (A A^T)[i, j] = sum_k A[i, k] A[j, k]. In col-major view, the row-major A buffer
    // is A_col with shape (K, M). The desired sum is (A_col^T A_col)[i, j] in col-major,
    // so we issue a single cublasSgemm with op_A = T, op_B = N on the same dA pointer.
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                M, M, K,
                &alpha, dA, K,
                dA, K,
                &beta, dC, M);

    Tensor C(M, M);
    cudaMemcpyAsync(C.data(), dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dA);
    cudaFree(dC);
    return C;
}

Tensor gemm_AtA_cuda(const Tensor& A) {
    auto& ctx = CudaContext::instance();
    if (!ctx.available()) return A.transpose().matmul(A);

    const int N = static_cast<int>(A.rows());
    const int M = static_cast<int>(A.cols());

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(ctx.cublas());

    float *dA = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(float) * N * M);
    cudaMalloc(&dC, sizeof(float) * M * M);
    cudaMemcpyAsync(dA, A.data(), sizeof(float) * N * M, cudaMemcpyHostToDevice, stream);

    // Row-major C (M x M) = A^T A with A shape (N, M).
    // C[i, j] = sum_k A[k, i] A[k, j]. In col-major view, A_col has shape (M, N) and
    // C col-major (M x M, symmetric so transpose is itself) = A_col A_col^T.
    // → cublasSgemm with op_A = N, op_B = T on the same dA pointer.
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                M, M, N,
                &alpha, dA, M,
                dA, M,
                &beta, dC, M);

    Tensor C(M, M);
    cudaMemcpyAsync(C.data(), dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dA);
    cudaFree(dC);
    return C;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
