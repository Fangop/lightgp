// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "matern_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../cpu/matern_cpu.h"
#include "cuda_context.h"

namespace lightgp {

namespace {

constexpr float kSqrt3 = 1.7320508075688772f;
constexpr float kSqrt5 = 2.2360679774997896f;

// `variant` ∈ {0,1,2} → Matern12 / Matern32 / Matern52.
__global__ void matern_assemble_kernel(const float* __restrict__ sq1,
                                       const float* __restrict__ sq2,
                                       const float* __restrict__ cross,
                                       float* __restrict__ K,
                                       int n, int m, int variant,
                                       float inv_l, float inv_l2, float signal_variance) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= m) return;
    float raw = sq1[i] + sq2[j] - 2.0f * cross[i * m + j];
    if (raw < 0.0f) raw = 0.0f;
    const float r = sqrtf(raw);
    float k = 0.0f;
    if (variant == 0) {
        k = expf(-r * inv_l);
    } else if (variant == 1) {
        const float u = kSqrt3 * r * inv_l;
        k = (1.0f + u) * expf(-u);
    } else {  // variant == 2
        const float u = kSqrt5 * r * inv_l;
        k = (1.0f + u + (5.0f / 3.0f) * raw * inv_l2) * expf(-u);
    }
    K[i * m + j] = signal_variance * k;
}

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

Tensor matern_kernel_cuda(const Tensor& X1, const Tensor& X2,
                          float length_scale, float signal_variance,
                          KernelType type) {
    assert(X1.cols() == X2.cols());
    assert(type != KernelType::RBF);

    auto& ctx = CudaContext::instance();
    if (!ctx.available()) {
        return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
    }
    const int n = static_cast<int>(X1.rows());
    const int m = static_cast<int>(X2.rows());
    const int d = static_cast<int>(X1.cols());
    const float inv_l = 1.0f / length_scale;
    const float inv_l2 = inv_l * inv_l;

    int variant = -1;
    switch (type) {
        case KernelType::Matern12: variant = 0; break;
        case KernelType::Matern32: variant = 1; break;
        case KernelType::Matern52: variant = 2; break;
        default: return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
    }

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

    const int red_threads = 128;
    const std::size_t shmem = red_threads * sizeof(float);
    row_sqnorm_kernel<<<n, red_threads, shmem, stream>>>(dX1, dSq1, n, d);
    row_sqnorm_kernel<<<m, red_threads, shmem, stream>>>(dX2, dSq2, m, d);

    // Row-major Cross = X1 X2^T via cuBLAS as Cross^T = X2 X1^T  (see rbf_cuda.cu for derivation).
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, d,
                &alpha, dX2, d,
                dX1, d,
                &beta, dCross, m);

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    matern_assemble_kernel<<<grid, block, 0, stream>>>(dSq1, dSq2, dCross, dK,
                                                       n, m, variant,
                                                       inv_l, inv_l2, signal_variance);

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
