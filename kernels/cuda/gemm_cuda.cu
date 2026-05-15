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

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
