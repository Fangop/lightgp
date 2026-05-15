#include "rbf_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../cpu/rbf_cpu.h"
#include "cuda_context.h"

namespace lightgp {

namespace {

/// One thread per matrix element. Each thread reduces D contributions of x_i to ||x_i||²
/// when needed, then assembles K[i, j] from (sq1[i], sq2[j], cross[i, j]).
__global__ void rbf_assemble_kernel(const float* __restrict__ sq1,
                                    const float* __restrict__ sq2,
                                    const float* __restrict__ cross,
                                    float* __restrict__ K,
                                    int n, int m,
                                    float inv_2l2, float signal_variance) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= m) return;
    float raw = sq1[i] + sq2[j] - 2.0f * cross[i * m + j];
    if (raw < 0.0f) raw = 0.0f;
    K[i * m + j] = signal_variance * expf(-inv_2l2 * raw);
}

/// Row-wise reduction: out[i] = sum_k X[i, d_offset + k]² over the whole row.
/// One block per row, blockDim.x threads collaborate on the D-long reduction.
__global__ void row_sqnorm_kernel(const float* __restrict__ X, float* __restrict__ out,
                                  int n, int d) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const float* row = X + i * d;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        const float v = row[k];
        acc += v * v;
    }
    // Block-level reduction in shared memory.
    extern __shared__ float smem[];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[i] = smem[0];
}

}  // namespace

Tensor rbf_kernel_cuda(const Tensor& X1, const Tensor& X2,
                       float length_scale, float signal_variance) {
    assert(X1.cols() == X2.cols());
    auto& ctx = CudaContext::instance();
    if (!ctx.available()) {
        return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
    }
    const int n = static_cast<int>(X1.rows());
    const int m = static_cast<int>(X2.rows());
    const int d = static_cast<int>(X1.cols());
    const float inv_2l2 = 0.5f / (length_scale * length_scale);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(ctx.cublas());

    float *dX1 = nullptr, *dX2 = nullptr;
    float *dSq1 = nullptr, *dSq2 = nullptr;
    float *dCross = nullptr, *dK = nullptr;
    cudaMalloc(&dX1, sizeof(float) * n * d);
    cudaMalloc(&dX2, sizeof(float) * m * d);
    cudaMalloc(&dSq1, sizeof(float) * n);
    cudaMalloc(&dSq2, sizeof(float) * m);
    cudaMalloc(&dCross, sizeof(float) * n * m);
    cudaMalloc(&dK, sizeof(float) * n * m);

    cudaMemcpyAsync(dX1, X1.data(), sizeof(float) * n * d, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX2, X2.data(), sizeof(float) * m * d, cudaMemcpyHostToDevice, stream);

    // Row squared norms via a tiny custom kernel (one block per row).
    const int red_threads = 128;
    const std::size_t shmem = red_threads * sizeof(float);
    row_sqnorm_kernel<<<n, red_threads, shmem, stream>>>(dX1, dSq1, n, d);
    row_sqnorm_kernel<<<m, red_threads, shmem, stream>>>(dX2, dSq2, m, d);

    // Cross = X1 @ X2^T  (n x m, row-major). In column-major cuBLAS view we compute
    // Cross^T = X2 @ X1^T using sgemm; the buffers we pass (row-major X1, X2) are
    // X1^T, X2^T in col-major. So with op_A = T on X2_data (n,d viewed as d×n then T → n×d?),
    // simpler: row-major result Cross[i,j] = sum_k X1[i,k] * X2[j,k] = (X1 X2^T)[i,j].
    // Call cublasSgemm to produce Cross^T (m x n col-major), reading X2 (m×d row-major
    // = d×m col-major) with op_N and X1 (n×d row-major = d×n col-major) with op_T.
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, d,
                &alpha, dX2, d,   // d×m col-major view; op_T → m×d
                dX1, d,           // d×n col-major view; op_N → d×n
                &beta, dCross, m);

    // Assemble K[i, j] = sf2 * exp(-inv_2l2 * (sq1[i] + sq2[j] - 2 cross[i, j])).
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    rbf_assemble_kernel<<<grid, block, 0, stream>>>(dSq1, dSq2, dCross, dK,
                                                    n, m, inv_2l2, signal_variance);

    Tensor K(n, m);
    cudaMemcpyAsync(K.data(), dK, sizeof(float) * n * m, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dX1);
    cudaFree(dX2);
    cudaFree(dSq1);
    cudaFree(dSq2);
    cudaFree(dCross);
    cudaFree(dK);
    return K;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
