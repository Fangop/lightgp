// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "rbf_matvec_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

#include "../cpu/rbf_cpu.h"
#include "cuda_context.h"

namespace lightgp {

namespace {

/// Tiled matrix-free RBF kernel-vector product.
/// One threadblock writes a TB-sized row tile of w; threads cooperate by streaming
/// column-tiles of X (X_j and v_j) through shared memory. Memory traffic is
/// O(N D / TB) global reads of X for each row tile, so total work stays O(N²) but
/// the j-axis bandwidth is amortized within each block.
///
/// Layout: blockDim.x = TB threads, each thread owns one row i of the output tile.
/// We sweep j over the full N in CHUNK-sized tiles; within each chunk every thread
/// loads its share of the X_j[k] entries into shared memory cooperatively, then all
/// threads scan the chunk to accumulate K(x_i, x_j) * v_j.
template <int TB, int CHUNK>
__global__ void rbf_matvec_kernel(const float* __restrict__ X,
                                  const float* __restrict__ v,
                                  float* __restrict__ w,
                                  int N, int D,
                                  float inv_2l2, float signal_variance, float noise_variance) {
    const int i = blockIdx.x * TB + threadIdx.x;
    extern __shared__ float smem[];
    // Layout: first CHUNK*D floats = X_j chunk, next CHUNK floats = v_j chunk.
    float* sX = smem;
    float* sV = smem + CHUNK * D;

    // Load this row x_i into thread-local registers (one register per dim is fine for D<=64;
    // beyond that we re-read from global on the fly).
    constexpr int D_REG = 16;
    float xi_reg[D_REG];
    const bool small_d = (D <= D_REG);
    if (small_d && i < N) {
        const float* row = X + i * D;
        #pragma unroll
        for (int k = 0; k < D_REG; ++k) {
            xi_reg[k] = (k < D) ? row[k] : 0.0f;
        }
    }

    float acc = 0.0f;
    for (int j0 = 0; j0 < N; j0 += CHUNK) {
        const int chunk = (j0 + CHUNK <= N) ? CHUNK : (N - j0);
        // Cooperative load: TB threads, chunk*D floats.
        for (int t = threadIdx.x; t < chunk * D; t += TB) {
            sX[t] = X[(j0 + t / D) * D + (t % D)];
        }
        for (int t = threadIdx.x; t < chunk; t += TB) {
            sV[t] = v[j0 + t];
        }
        __syncthreads();

        if (i < N) {
            for (int jj = 0; jj < chunk; ++jj) {
                float r2 = 0.0f;
                if (small_d) {
                    #pragma unroll
                    for (int k = 0; k < D_REG; ++k) {
                        if (k < D) {
                            const float diff = xi_reg[k] - sX[jj * D + k];
                            r2 += diff * diff;
                        }
                    }
                } else {
                    const float* row = X + i * D;
                    for (int k = 0; k < D; ++k) {
                        const float diff = row[k] - sX[jj * D + k];
                        r2 += diff * diff;
                    }
                }
                if (r2 < 0.0f) r2 = 0.0f;
                acc += signal_variance * expf(-inv_2l2 * r2) * sV[jj];
            }
        }
        __syncthreads();
    }

    if (i < N) w[i] = acc + noise_variance * v[i];
}

}  // namespace

Tensor rbf_matvec_cuda(const Tensor& X, const Tensor& v,
                       float length_scale, float signal_variance, float noise_variance) {
    assert(X.rows() == v.rows());
    assert(v.cols() == 1);

    auto& ctx = CudaContext::instance();
    if (!ctx.available()) {
        // Fallback: form K on the host and matmul.
        Tensor K = rbf_kernel_cpu(X, X, length_scale, signal_variance);
        K.add_jitter(noise_variance);
        return K.matmul(v);
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());
    const float inv_2l2 = 0.5f / (length_scale * length_scale);

    float *dX = nullptr, *dV = nullptr, *dW = nullptr;
    cudaMalloc(&dX, sizeof(float) * N * D);
    cudaMalloc(&dV, sizeof(float) * N);
    cudaMalloc(&dW, sizeof(float) * N);

    cudaMemcpyAsync(dX, X.data(), sizeof(float) * N * D, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dV, v.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream);

    constexpr int TB = 64;
    constexpr int CHUNK = 64;
    const std::size_t shmem = sizeof(float) * (CHUNK * D + CHUNK);
    const int grid = (N + TB - 1) / TB;
    rbf_matvec_kernel<TB, CHUNK><<<grid, TB, shmem, stream>>>(
        dX, dV, dW, N, D, inv_2l2, signal_variance, noise_variance);

    Tensor w(N, 1);
    cudaMemcpyAsync(w.data(), dW, sizeof(float) * N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dX);
    cudaFree(dV);
    cudaFree(dW);
    return w;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
