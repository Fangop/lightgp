// Canonical source for the GP Metal shaders.
// Kept in sync with the embedded copy in kernels/metal/metal_context.mm,
// which compiles this source at runtime via newLibraryWithSource:.
//
// Three RBF variants are dispatched based on D:
//   * rbf_kernel_small_d  — D <= 16: load entire row strips once, single barrier.
//   * rbf_kernel_tiled    — any D, scalar loads through threadgroup memory.
//   * rbf_kernel_tiled_f4 — D % 4 == 0, float4 loads.

#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define D_TILE_SCALAR 16
#define D_TILE_F4 16   // count in float4s; 16 * 4 = 64 floats
#define D_SMALL_MAX 16 // upper bound for the small-D kernel's LDS layout

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

    // Cooperative load: TILE rows of D scalars each. Coalesced because consecutive
    // linear thread ids map to consecutive (r, c) -> consecutive global addresses.
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

// C(M_dim x N_dim) = A(M_dim x K_dim) * B(K_dim x N_dim). Untiled GEMM, one thread per C element.
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

// Tiled GEMM: 16x16 threadgroup computes a 64x64 block of C, each thread a 4x4 micro-tile.
// Inner loop K-tile width = 16. Bounds-checked for non-multiple-of-64 dims.
#define GEMM_TILE 64
#define GEMM_TG 16
#define GEMM_MICRO 4
#define GEMM_KTILE 16

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
    threadgroup float Asub[GEMM_TILE * GEMM_KTILE];  // 64 x 16
    threadgroup float Bsub[GEMM_KTILE * GEMM_TILE];  // 16 x 64

    const uint row_base = tgid.y * GEMM_TILE;
    const uint col_base = tgid.x * GEMM_TILE;

    float acc[GEMM_MICRO][GEMM_MICRO];
    for (uint i = 0; i < GEMM_MICRO; ++i)
        for (uint j = 0; j < GEMM_MICRO; ++j)
            acc[i][j] = 0.0f;

    const uint lin = tid.y * GEMM_TG + tid.x;  // 0..255

    for (uint k_base = 0; k_base < K_dim; k_base += GEMM_KTILE) {
        // Load 64x16 strip of A: 256 threads × 4 floats = 1024 floats.
        for (uint e = 0; e < (GEMM_TILE * GEMM_KTILE) / (GEMM_TG * GEMM_TG); ++e) {
            const uint idx = lin + e * GEMM_TG * GEMM_TG;
            const uint r = idx / GEMM_KTILE;
            const uint c = idx % GEMM_KTILE;
            const uint gr = row_base + r;
            const uint gc = k_base + c;
            Asub[r * GEMM_KTILE + c] =
                (gr < M_dim && gc < K_dim) ? A[gr * K_dim + gc] : 0.0f;
        }
        // Load 16x64 strip of B: also 1024 floats.
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

        // Each thread accumulates a 4x4 micro-tile.
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

    // Write 4x4 micro-tile with bounds checks.
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

// Forward substitution: solve L * X = B for X, L is N x N lower triangular, B and X are N x M.
// One threadgroup per RHS column; threads in the group cooperate on the per-row dot product.
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

// Backward substitution: solve L^T * X = B for X. Iterate i from N-1 down to 0;
// row dot product is sum over j > i of L[j, i] * x[j].
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

// Matrix-free RBF kernel-vector product: w = (K + sn2 * I) * v.
// Each thread computes w[i] = sum_j K(x_i, x_j) v[j] + sn2 * v[i].
// j-dimension is tiled through threadgroup memory; X1 row and v are reused.
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

    // Cache the current row x_i for fast comparison. Bound D <= 64 for the cache;
    // otherwise read X directly (still correct, slightly slower).
    const bool small_d = (D <= 64);
    float x_i[64];
    if (i < N && small_d) {
        for (uint d = 0; d < D; ++d) x_i[d] = X[i * D + d];
    }

    for (uint j_base = 0; j_base < N; j_base += MATVEC_TILE) {
        // Cooperative load of v[j_base : j_base + MATVEC_TILE] into shared memory.
        const uint j_load = j_base + tid;
        v_tile[tid] = (j_load < N) ? v[j_load] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i < N) {
            const uint j_max = min(MATVEC_TILE, N - j_base);
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
        // Add the diagonal noise contribution (K_y = K + sn2 * I).
        w[i] = acc + sn2 * v[i];
    }
}
