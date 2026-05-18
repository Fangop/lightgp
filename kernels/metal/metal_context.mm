// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#ifdef LIGHTGP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_context.h"

namespace lightgp {

// Mirror of kernels/metal/shaders/gp_kernels.metal — keep these in sync.
// Embedded so we can compile via newLibraryWithSource: without needing the
// standalone `metal`/`metallib` toolchain (full Xcode only).
static NSString* const kGpKernelsSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define D_TILE_SCALAR 16
#define D_TILE_F4 16
#define D_SMALL_MAX 16

kernel void rbf_kernel_small_d(
    device const float* X1     [[buffer(0)]],
    device const float* X2     [[buffer(1)]],
    device float* K            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& D           [[buffer(5)]],
    constant float& inv_2l2    [[buffer(6)]],
    constant float& sf2        [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float lds_X1[TILE * D_SMALL_MAX];
    threadgroup float lds_X2[TILE * D_SMALL_MAX];

    const uint i_base = tgid.y * TILE;
    const uint j_base = tgid.x * TILE;
    const uint lin = tid.y * TILE + tid.x;

    if (lin < TILE * D) {
        const uint r = lin / D;
        const uint c = lin % D;
        const uint x1_row = i_base + r;
        const uint x2_row = j_base + r;
        lds_X1[r * D_SMALL_MAX + c] = (x1_row < N) ? X1[x1_row * D + c] : 0.0f;
        lds_X2[r * D_SMALL_MAX + c] = (x2_row < M) ? X2[x2_row * D + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.y < N && gid.x < M) {
        float r2 = 0.0f;
        for (uint k = 0; k < D; ++k) {
            const float diff = lds_X1[tid.y * D_SMALL_MAX + k]
                             - lds_X2[tid.x * D_SMALL_MAX + k];
            r2 += diff * diff;
        }
        K[gid.y * M + gid.x] = sf2 * exp(-inv_2l2 * r2);
    }
}

kernel void rbf_kernel_tiled(
    device const float* X1     [[buffer(0)]],
    device const float* X2     [[buffer(1)]],
    device float* K            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& D           [[buffer(5)]],
    constant float& inv_2l2    [[buffer(6)]],
    constant float& sf2        [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float lds_X1[TILE * D_TILE_SCALAR];
    threadgroup float lds_X2[TILE * D_TILE_SCALAR];

    const uint i = gid.y;
    const uint j = gid.x;
    const uint row_t = tid.y;
    const uint col_t = tid.x;
    const uint lin = row_t * TILE + col_t;
    const uint load_row = lin / D_TILE_SCALAR;
    const uint load_col = lin % D_TILE_SCALAR;
    const uint i_base = tgid.y * TILE;
    const uint j_base = tgid.x * TILE;

    float r2 = 0.0f;

    for (uint d_base = 0; d_base < D; d_base += D_TILE_SCALAR) {
        const uint x1_row = i_base + load_row;
        const uint x2_row = j_base + load_row;
        const uint dk = d_base + load_col;
        lds_X1[load_row * D_TILE_SCALAR + load_col] =
            (x1_row < N && dk < D) ? X1[x1_row * D + dk] : 0.0f;
        lds_X2[load_row * D_TILE_SCALAR + load_col] =
            (x2_row < M && dk < D) ? X2[x2_row * D + dk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_max = min((uint)D_TILE_SCALAR, D - d_base);
        for (uint k = 0; k < k_max; ++k) {
            const float diff = lds_X1[row_t * D_TILE_SCALAR + k]
                             - lds_X2[col_t * D_TILE_SCALAR + k];
            r2 += diff * diff;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < N && j < M) {
        K[i * M + j] = sf2 * exp(-inv_2l2 * r2);
    }
}

kernel void rbf_kernel_tiled_f4(
    device const float* X1     [[buffer(0)]],
    device const float* X2     [[buffer(1)]],
    device float* K            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& D           [[buffer(5)]],
    constant float& inv_2l2    [[buffer(6)]],
    constant float& sf2        [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float4 lds_X1[TILE * D_TILE_F4];
    threadgroup float4 lds_X2[TILE * D_TILE_F4];

    device const float4* X1_v = reinterpret_cast<device const float4*>(X1);
    device const float4* X2_v = reinterpret_cast<device const float4*>(X2);

    const uint i = gid.y;
    const uint j = gid.x;
    const uint row_t = tid.y;
    const uint col_t = tid.x;
    const uint lin = row_t * TILE + col_t;
    const uint load_row = lin / D_TILE_F4;
    const uint load_col = lin % D_TILE_F4;
    const uint i_base = tgid.y * TILE;
    const uint j_base = tgid.x * TILE;
    const uint D4 = D / 4;

    float r2 = 0.0f;

    for (uint d4_base = 0; d4_base < D4; d4_base += D_TILE_F4) {
        const uint x1_row = i_base + load_row;
        const uint x2_row = j_base + load_row;
        const uint dk4 = d4_base + load_col;
        lds_X1[load_row * D_TILE_F4 + load_col] =
            (x1_row < N && dk4 < D4) ? X1_v[x1_row * D4 + dk4] : float4(0.0f);
        lds_X2[load_row * D_TILE_F4 + load_col] =
            (x2_row < M && dk4 < D4) ? X2_v[x2_row * D4 + dk4] : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_max = min((uint)D_TILE_F4, D4 - d4_base);
        for (uint k = 0; k < k_max; ++k) {
            const float4 diff = lds_X1[row_t * D_TILE_F4 + k]
                              - lds_X2[col_t * D_TILE_F4 + k];
            r2 += dot(diff, diff);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < N && j < M) {
        K[i * M + j] = sf2 * exp(-inv_2l2 * r2);
    }
}

kernel void gemm_naive(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M_dim       [[buffer(3)]],
    constant uint& N_dim       [[buffer(4)]],
    constant uint& K_dim       [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    if (gid.x >= N_dim || gid.y >= M_dim) return;
    const uint i = gid.y;
    const uint j = gid.x;
    float acc = 0.0f;
    for (uint k = 0; k < K_dim; ++k) {
        acc += A[i * K_dim + k] * B[k * N_dim + j];
    }
    C[i * N_dim + j] = acc;
}

#define GEMM_TILE 64
#define GEMM_TG 16
#define GEMM_MICRO 4
#define GEMM_KTILE 16

// simdgroup_matrix GEMM. Each threadgroup covers a 32x32 output tile using a
// 32x16 thread layout (16 simdgroups of 32 threads each, one simdgroup per row).
// Optional kernel — only loaded if newLibraryWithSource succeeds.
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#define SGEMM_TILE 32
#define SGEMM_KT 8

kernel void gemm_simdgroup(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M_dim       [[buffer(3)]],
    constant uint& N_dim       [[buffer(4)]],
    constant uint& K_dim       [[buffer(5)]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float Asub[SGEMM_TILE * SGEMM_KT];
    threadgroup float Bsub[SGEMM_KT * SGEMM_TILE];
    threadgroup float Csub[SGEMM_TILE * SGEMM_TILE];

    const uint lin = tid.y * 32u + tid.x;  // 0..511 with threadgroup (32,16,1)
    const uint sgid = tid.y;                // each row of the threadgroup = one simdgroup
    const uint sy = sgid / 4u;              // simdgroup row position in the 4x4 tile grid
    const uint sx = sgid % 4u;
    const uint c_r = sy * 8u;
    const uint c_c = sx * 8u;
    const uint row_base = tgid.y * SGEMM_TILE;
    const uint col_base = tgid.x * SGEMM_TILE;

    simdgroup_float8x8 C_acc = simdgroup_float8x8(0.0f);

    for (uint k_base = 0; k_base < K_dim; k_base += SGEMM_KT) {
        // 512 threads split: first 256 load Asub (32x8), second 256 load Bsub (8x32).
        if (lin < 256u) {
            const uint r = lin / SGEMM_KT;
            const uint c = lin % SGEMM_KT;
            const uint gr = row_base + r;
            const uint gc = k_base + c;
            Asub[r * SGEMM_KT + c] =
                (gr < M_dim && gc < K_dim) ? A[gr * K_dim + gc] : 0.0f;
        } else {
            const uint t = lin - 256u;
            const uint r = t / SGEMM_TILE;
            const uint c = t % SGEMM_TILE;
            const uint gr = k_base + r;
            const uint gc = col_base + c;
            Bsub[r * SGEMM_TILE + c] =
                (gr < K_dim && gc < N_dim) ? B[gr * N_dim + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 A_sg, B_sg;
        simdgroup_load(A_sg, &Asub[c_r * SGEMM_KT], SGEMM_KT);
        simdgroup_load(B_sg, &Bsub[c_c], SGEMM_TILE);
        simdgroup_multiply_accumulate(C_acc, A_sg, B_sg, C_acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write through threadgroup memory so we can bounds-check the global store.
    simdgroup_store(C_acc, &Csub[c_r * SGEMM_TILE + c_c], SGEMM_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0; e < 2u; ++e) {  // 1024 elements / 512 threads
        const uint idx = lin + e * 512u;
        const uint r = idx / SGEMM_TILE;
        const uint c = idx % SGEMM_TILE;
        const uint gr = row_base + r;
        const uint gc = col_base + c;
        if (gr < M_dim && gc < N_dim) {
            C[gr * N_dim + gc] = Csub[r * SGEMM_TILE + c];
        }
    }
}

kernel void gemm_tiled(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M_dim       [[buffer(3)]],
    constant uint& N_dim       [[buffer(4)]],
    constant uint& K_dim       [[buffer(5)]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float Asub[GEMM_TILE * GEMM_KTILE];
    threadgroup float Bsub[GEMM_KTILE * GEMM_TILE];

    const uint row_base = tgid.y * GEMM_TILE;
    const uint col_base = tgid.x * GEMM_TILE;

    float acc[GEMM_MICRO][GEMM_MICRO];
    for (uint i = 0; i < GEMM_MICRO; ++i)
        for (uint j = 0; j < GEMM_MICRO; ++j)
            acc[i][j] = 0.0f;

    const uint lin = tid.y * GEMM_TG + tid.x;

    for (uint k_base = 0; k_base < K_dim; k_base += GEMM_KTILE) {
        for (uint e = 0; e < (GEMM_TILE * GEMM_KTILE) / (GEMM_TG * GEMM_TG); ++e) {
            const uint idx = lin + e * GEMM_TG * GEMM_TG;
            const uint r = idx / GEMM_KTILE;
            const uint c = idx % GEMM_KTILE;
            const uint gr = row_base + r;
            const uint gc = k_base + c;
            Asub[r * GEMM_KTILE + c] =
                (gr < M_dim && gc < K_dim) ? A[gr * K_dim + gc] : 0.0f;
        }
        for (uint e = 0; e < (GEMM_KTILE * GEMM_TILE) / (GEMM_TG * GEMM_TG); ++e) {
            const uint idx = lin + e * GEMM_TG * GEMM_TG;
            const uint r = idx / GEMM_TILE;
            const uint c = idx % GEMM_TILE;
            const uint gr = k_base + r;
            const uint gc = col_base + c;
            Bsub[r * GEMM_TILE + c] =
                (gr < K_dim && gc < N_dim) ? B[gr * N_dim + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < GEMM_KTILE; ++kk) {
            float a_reg[GEMM_MICRO];
            float b_reg[GEMM_MICRO];
            for (uint i = 0; i < GEMM_MICRO; ++i) {
                a_reg[i] = Asub[(tid.y * GEMM_MICRO + i) * GEMM_KTILE + kk];
            }
            for (uint j = 0; j < GEMM_MICRO; ++j) {
                b_reg[j] = Bsub[kk * GEMM_TILE + (tid.x * GEMM_MICRO + j)];
            }
            for (uint i = 0; i < GEMM_MICRO; ++i) {
                for (uint j = 0; j < GEMM_MICRO; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < GEMM_MICRO; ++i) {
        const uint gr = row_base + tid.y * GEMM_MICRO + i;
        if (gr >= M_dim) continue;
        for (uint j = 0; j < GEMM_MICRO; ++j) {
            const uint gc = col_base + tid.x * GEMM_MICRO + j;
            if (gc >= N_dim) continue;
            C[gr * N_dim + gc] = acc[i][j];
        }
    }
}

#define TRSV_THREADS 64

kernel void trsv_lower_forward(
    device const float* L      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* X            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tgid                  [[threadgroup_position_in_grid]])
{
    if (tgid >= M) return;
    const uint col = tgid;
    threadgroup float partial[TRSV_THREADS];

    for (uint i = 0; i < N; ++i) {
        float ps = 0.0f;
        for (uint j = tid; j < i; j += TRSV_THREADS) {
            ps += L[i * N + j] * X[j * M + col];
        }
        partial[tid] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TRSV_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            X[i * M + col] = (B[i * M + col] - partial[0]) / L[i * N + i];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void trsv_lower_backward(
    device const float* L      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* X            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tgid                  [[threadgroup_position_in_grid]])
{
    if (tgid >= M) return;
    const uint col = tgid;
    threadgroup float partial[TRSV_THREADS];

    for (int ii = int(N) - 1; ii >= 0; --ii) {
        const uint i = uint(ii);
        float ps = 0.0f;
        for (uint j = i + 1 + tid; j < N; j += TRSV_THREADS) {
            ps += L[j * N + i] * X[j * M + col];
        }
        partial[tid] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TRSV_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            X[i * M + col] = (B[i * M + col] - partial[0]) / L[i * N + i];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

#define MATVEC_TG 64
#define MATVEC_TILE 64

kernel void rbf_matvec(
    device const float* X      [[buffer(0)]],
    device const float* v      [[buffer(1)]],
    device float* w            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& D           [[buffer(4)]],
    constant float& inv_2l2    [[buffer(5)]],
    constant float& sf2        [[buffer(6)]],
    constant float& sn2        [[buffer(7)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tgid                  [[threadgroup_position_in_grid]])
{
    const uint i = tgid * MATVEC_TG + tid;
    threadgroup float v_tile[MATVEC_TILE];

    float acc = 0.0f;
    const bool small_d = (D <= 64);
    float x_i[64];
    if (i < N && small_d) {
        for (uint d = 0; d < D; ++d) x_i[d] = X[i * D + d];
    }

    for (uint j_base = 0; j_base < N; j_base += MATVEC_TILE) {
        const uint j_load = j_base + tid;
        v_tile[tid] = (j_load < N) ? v[j_load] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i < N) {
            const uint j_max = min((uint)MATVEC_TILE, N - j_base);
            for (uint jj = 0; jj < j_max; ++jj) {
                const uint j = j_base + jj;
                float r2 = 0.0f;
                if (small_d) {
                    for (uint d = 0; d < D; ++d) {
                        const float diff = x_i[d] - X[j * D + d];
                        r2 += diff * diff;
                    }
                } else {
                    for (uint d = 0; d < D; ++d) {
                        const float diff = X[i * D + d] - X[j * D + d];
                        r2 += diff * diff;
                    }
                }
                acc += sf2 * exp(-inv_2l2 * r2) * v_tile[jj];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < N) {
        w[i] = acc + sn2 * v[i];
    }
}

// ------- Generic Matern kernels via function constant KERNEL_ID -------
// KERNEL_ID: 1 = Matern12, 2 = Matern32, 3 = Matern52.
// Three pipeline states per shape are built by specializing this constant at PSO time.

constant int KERNEL_ID [[function_constant(0)]];

inline float matern_kernel_value(float r2, float inv_l, float sf2) {
    const float r = sqrt(max(r2, 0.0f));
    if (KERNEL_ID == 1) {
        return sf2 * exp(-r * inv_l);
    } else if (KERNEL_ID == 2) {
        const float u = 1.7320508075688772f * r * inv_l;
        return sf2 * (1.0f + u) * exp(-u);
    } else {  // 3 → Matern52
        const float u = 2.23606797749979f * r * inv_l;
        const float u2 = 5.0f * r2 * inv_l * inv_l;
        return sf2 * (1.0f + u + u2 / 3.0f) * exp(-u);
    }
}

kernel void matern_kernel_small_d(
    device const float* X1     [[buffer(0)]],
    device const float* X2     [[buffer(1)]],
    device float* K            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& D           [[buffer(5)]],
    constant float& inv_l      [[buffer(6)]],
    constant float& sf2        [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float lds_X1[TILE * D_SMALL_MAX];
    threadgroup float lds_X2[TILE * D_SMALL_MAX];

    const uint i_base = tgid.y * TILE;
    const uint j_base = tgid.x * TILE;
    const uint lin = tid.y * TILE + tid.x;

    if (lin < TILE * D) {
        const uint r = lin / D;
        const uint c = lin % D;
        const uint x1_row = i_base + r;
        const uint x2_row = j_base + r;
        lds_X1[r * D_SMALL_MAX + c] = (x1_row < N) ? X1[x1_row * D + c] : 0.0f;
        lds_X2[r * D_SMALL_MAX + c] = (x2_row < M) ? X2[x2_row * D + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.y < N && gid.x < M) {
        float r2 = 0.0f;
        for (uint k = 0; k < D; ++k) {
            const float diff = lds_X1[tid.y * D_SMALL_MAX + k]
                             - lds_X2[tid.x * D_SMALL_MAX + k];
            r2 += diff * diff;
        }
        K[gid.y * M + gid.x] = matern_kernel_value(r2, inv_l, sf2);
    }
}

kernel void matern_kernel_tiled(
    device const float* X1     [[buffer(0)]],
    device const float* X2     [[buffer(1)]],
    device float* K            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& D           [[buffer(5)]],
    constant float& inv_l      [[buffer(6)]],
    constant float& sf2        [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float lds_X1[TILE * D_TILE_SCALAR];
    threadgroup float lds_X2[TILE * D_TILE_SCALAR];

    const uint i = gid.y;
    const uint j = gid.x;
    const uint row_t = tid.y;
    const uint col_t = tid.x;
    const uint lin = row_t * TILE + col_t;
    const uint load_row = lin / D_TILE_SCALAR;
    const uint load_col = lin % D_TILE_SCALAR;
    const uint i_base = tgid.y * TILE;
    const uint j_base = tgid.x * TILE;

    float r2 = 0.0f;
    for (uint d_base = 0; d_base < D; d_base += D_TILE_SCALAR) {
        const uint x1_row = i_base + load_row;
        const uint x2_row = j_base + load_row;
        const uint dk = d_base + load_col;
        lds_X1[load_row * D_TILE_SCALAR + load_col] =
            (x1_row < N && dk < D) ? X1[x1_row * D + dk] : 0.0f;
        lds_X2[load_row * D_TILE_SCALAR + load_col] =
            (x2_row < M && dk < D) ? X2[x2_row * D + dk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_max = min((uint)D_TILE_SCALAR, D - d_base);
        for (uint k = 0; k < k_max; ++k) {
            const float diff = lds_X1[row_t * D_TILE_SCALAR + k]
                             - lds_X2[col_t * D_TILE_SCALAR + k];
            r2 += diff * diff;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < N && j < M) {
        K[i * M + j] = matern_kernel_value(r2, inv_l, sf2);
    }
}

kernel void matern_kernel_tiled_f4(
    device const float* X1     [[buffer(0)]],
    device const float* X2     [[buffer(1)]],
    device float* K            [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& D           [[buffer(5)]],
    constant float& inv_l      [[buffer(6)]],
    constant float& sf2        [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 tid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]])
{
    threadgroup float4 lds_X1[TILE * D_TILE_F4];
    threadgroup float4 lds_X2[TILE * D_TILE_F4];

    device const float4* X1_v = reinterpret_cast<device const float4*>(X1);
    device const float4* X2_v = reinterpret_cast<device const float4*>(X2);

    const uint i = gid.y;
    const uint j = gid.x;
    const uint row_t = tid.y;
    const uint col_t = tid.x;
    const uint lin = row_t * TILE + col_t;
    const uint load_row = lin / D_TILE_F4;
    const uint load_col = lin % D_TILE_F4;
    const uint i_base = tgid.y * TILE;
    const uint j_base = tgid.x * TILE;
    const uint D4 = D / 4;

    float r2 = 0.0f;
    for (uint d4_base = 0; d4_base < D4; d4_base += D_TILE_F4) {
        const uint x1_row = i_base + load_row;
        const uint x2_row = j_base + load_row;
        const uint dk4 = d4_base + load_col;
        lds_X1[load_row * D_TILE_F4 + load_col] =
            (x1_row < N && dk4 < D4) ? X1_v[x1_row * D4 + dk4] : float4(0.0f);
        lds_X2[load_row * D_TILE_F4 + load_col] =
            (x2_row < M && dk4 < D4) ? X2_v[x2_row * D4 + dk4] : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_max = min((uint)D_TILE_F4, D4 - d4_base);
        for (uint k = 0; k < k_max; ++k) {
            const float4 diff = lds_X1[row_t * D_TILE_F4 + k]
                              - lds_X2[col_t * D_TILE_F4 + k];
            r2 += dot(diff, diff);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < N && j < M) {
        K[i * M + j] = matern_kernel_value(r2, inv_l, sf2);
    }
}
)METAL";

namespace {
void* make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name,
                    std::string& err_out) {
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        err_out = std::string("newFunctionWithName(") + name.UTF8String + ") returned nil";
        return nullptr;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        const char* msg = err.localizedDescription.UTF8String;
        err_out = std::string("newComputePipelineStateWithFunction(") + name.UTF8String
                  + "): " + (msg ? msg : "<no error>");
        return nullptr;
    }
    return (__bridge_retained void*)pso;
}

void* make_pipeline_specialized(id<MTLDevice> device, id<MTLLibrary> library,
                                NSString* name, int kernel_id,
                                std::string& err_out) {
    MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
    [fcv setConstantValue:&kernel_id type:MTLDataTypeInt atIndex:0];
    NSError* err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name constantValues:fcv error:&err];
    if (!fn) {
        const char* msg = err.localizedDescription.UTF8String;
        err_out = std::string("newFunctionWithName(") + name.UTF8String
                  + " kid=" + std::to_string(kernel_id) + "): " + (msg ? msg : "<no error>");
        return nullptr;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        const char* msg = err.localizedDescription.UTF8String;
        err_out = std::string("newComputePipelineStateWithFunction(") + name.UTF8String
                  + " kid=" + std::to_string(kernel_id) + "): " + (msg ? msg : "<no error>");
        return nullptr;
    }
    return (__bridge_retained void*)pso;
}
}  // namespace

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            error_ = "MTLCreateSystemDefaultDevice returned nil";
            return;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            error_ = "newCommandQueue returned nil";
            return;
        }

        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:kGpKernelsSource
                                                      options:opts
                                                        error:&err];
        if (!library) {
            const char* msg = err.localizedDescription.UTF8String;
            error_ = std::string("newLibraryWithSource failed: ") + (msg ? msg : "<no error>");
            return;
        }

        std::string e;
        void* rbf_small = make_pipeline(device, library, @"rbf_kernel_small_d", e);
        if (!rbf_small) { error_ = e; return; }
        void* rbf = make_pipeline(device, library, @"rbf_kernel_tiled", e);
        if (!rbf) { error_ = e; return; }
        void* rbf_f4 = make_pipeline(device, library, @"rbf_kernel_tiled_f4", e);
        if (!rbf_f4) { error_ = e; return; }
        void* gemm = make_pipeline(device, library, @"gemm_naive", e);
        if (!gemm) { error_ = e; return; }
        void* gemm_t = make_pipeline(device, library, @"gemm_tiled", e);
        if (!gemm_t) { error_ = e; return; }
        // Simdgroup matmul is optional — needs Apple9 hardware (M3+) and Metal 2.3+ MSL.
        // If compilation or PSO creation fails (older GPU, older driver), leave it null
        // and dispatch falls back to gemm_tiled.
        std::string sg_err;
        void* gemm_sg = make_pipeline(device, library, @"gemm_simdgroup", sg_err);
        // Don't set error_ on simdgroup failure — it's optional.
        void* trsv_fwd = make_pipeline(device, library, @"trsv_lower_forward", e);
        if (!trsv_fwd) { error_ = e; return; }
        void* trsv_bwd = make_pipeline(device, library, @"trsv_lower_backward", e);
        if (!trsv_bwd) { error_ = e; return; }
        void* matvec = make_pipeline(device, library, @"rbf_matvec", e);
        if (!matvec) { error_ = e; return; }

        // Matern kernel pipelines: 3 variants (12, 32, 52) × 3 shapes (small_d, tiled, tiled_f4).
        // Function-constant KERNEL_ID specializes one shader source into three PSOs per shape.
        NSString* matern_shape_names[3] = {
            @"matern_kernel_small_d", @"matern_kernel_tiled", @"matern_kernel_tiled_f4"
        };
        void* matern_pipes[3][3] = {{nullptr}};
        for (int variant = 0; variant < 3; ++variant) {
            const int kid = variant + 1;  // 1=Matern12, 2=Matern32, 3=Matern52
            for (int shape = 0; shape < 3; ++shape) {
                std::string m_err;
                void* p = make_pipeline_specialized(device, library,
                                                    matern_shape_names[shape], kid, m_err);
                if (!p) { error_ = m_err; return; }
                matern_pipes[variant][shape] = p;
            }
        }

        device_ = (__bridge_retained void*)device;
        command_queue_ = (__bridge_retained void*)queue;
        library_ = (__bridge_retained void*)library;
        rbf_pipeline_small_d_ = rbf_small;
        rbf_pipeline_ = rbf;
        rbf_pipeline_f4_ = rbf_f4;
        gemm_pipeline_ = gemm;
        gemm_tiled_pipeline_ = gemm_t;
        gemm_simdgroup_pipeline_ = gemm_sg;
        trsv_forward_pipeline_ = trsv_fwd;
        trsv_backward_pipeline_ = trsv_bwd;
        rbf_matvec_pipeline_ = matvec;
        for (int v = 0; v < 3; ++v)
            for (int s = 0; s < 3; ++s)
                matern_kernel_pipelines_[v][s] = matern_pipes[v][s];
    }
}

MetalContext::~MetalContext() {
    for (int v = 0; v < 3; ++v) {
        for (int s = 0; s < 3; ++s) {
            if (matern_kernel_pipelines_[v][s]) {
                (void)(__bridge_transfer id)matern_kernel_pipelines_[v][s];
                matern_kernel_pipelines_[v][s] = nullptr;
            }
        }
    }
    if (rbf_matvec_pipeline_)   { (void)(__bridge_transfer id)rbf_matvec_pipeline_;   rbf_matvec_pipeline_   = nullptr; }
    if (trsv_backward_pipeline_){ (void)(__bridge_transfer id)trsv_backward_pipeline_;trsv_backward_pipeline_= nullptr; }
    if (trsv_forward_pipeline_) { (void)(__bridge_transfer id)trsv_forward_pipeline_; trsv_forward_pipeline_ = nullptr; }
    if (gemm_simdgroup_pipeline_){ (void)(__bridge_transfer id)gemm_simdgroup_pipeline_; gemm_simdgroup_pipeline_ = nullptr; }
    if (gemm_tiled_pipeline_)   { (void)(__bridge_transfer id)gemm_tiled_pipeline_;   gemm_tiled_pipeline_   = nullptr; }
    if (gemm_pipeline_)         { (void)(__bridge_transfer id)gemm_pipeline_;         gemm_pipeline_         = nullptr; }
    if (rbf_pipeline_f4_)       { (void)(__bridge_transfer id)rbf_pipeline_f4_;       rbf_pipeline_f4_       = nullptr; }
    if (rbf_pipeline_)          { (void)(__bridge_transfer id)rbf_pipeline_;          rbf_pipeline_          = nullptr; }
    if (rbf_pipeline_small_d_)  { (void)(__bridge_transfer id)rbf_pipeline_small_d_;  rbf_pipeline_small_d_  = nullptr; }
    if (library_)               { (void)(__bridge_transfer id)library_;               library_               = nullptr; }
    if (command_queue_)         { (void)(__bridge_transfer id)command_queue_;         command_queue_         = nullptr; }
    if (device_)                { (void)(__bridge_transfer id)device_;                device_                = nullptr; }
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
