#include "cholesky_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../../kernels/cuda/cuda_context.h"
#include "../cpu/cholesky_cpu.h"

namespace lightgp {

namespace {

/// Zero the strict-upper triangle of an n x n row-major matrix on device.
__global__ void zero_upper_kernel(float* M, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) M[i * n + j] = 0.0f;
}

}  // namespace

bool cholesky_cuda(const Tensor& K, Tensor& L) {
    assert(K.rows() == K.cols());
    auto& ctx = CudaContext::instance();
    if (!ctx.available()) return cholesky_cpu(K, L);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cusolverDnHandle_t solver = reinterpret_cast<cusolverDnHandle_t>(ctx.cusolver());

    const int n = static_cast<int>(K.rows());

    float* dK = nullptr;
    cudaMalloc(&dK, sizeof(float) * n * n);
    cudaMemcpyAsync(dK, K.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice, stream);

    int lwork = 0;
    cusolverDnSpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n, dK, n, &lwork);
    float* dWork = nullptr;
    int* dInfo = nullptr;
    cudaMalloc(&dWork, sizeof(float) * lwork);
    cudaMalloc(&dInfo, sizeof(int));

    // Row-major SPD passed to col-major spotrf with UPLO='U' is equivalent to col-major
    // SPD with the same UPLO; the U it returns in col-major lands in the row-major lower
    // triangle (= our L). Same trick as the Accelerate / OpenBLAS path in blas_accel.cpp.
    cusolverDnSpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, dK, n, dWork, lwork, dInfo);
    int info = 0;
    cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Zero strict-upper triangle on device before copying out.
    if (info == 0) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        zero_upper_kernel<<<grid, block, 0, stream>>>(dK, n);
    }

    L = Tensor(n, n);
    cudaMemcpyAsync(L.data(), dK, sizeof(float) * n * n, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dWork);
    cudaFree(dInfo);
    cudaFree(dK);

    return info == 0;
}

bool cholesky_cuda_with_jitter(const Tensor& K, Tensor& L, float& jitter_used) {
    assert(K.rows() == K.cols());
    constexpr float jitters[] = {1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.0f};
    for (float j : jitters) {
        Tensor A = K;
        A.add_jitter(j);
        if (cholesky_cuda(A, L)) {
            jitter_used = j;
            return true;
        }
    }
    return false;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
