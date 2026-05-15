#include "ski_cuda.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

#include "../kernels/cuda/cuda_context.h"
#include "ski.h"

namespace lightgp {

// -----------------------------------------------------------------------------
// SkiCudaState — everything the CUDA SKI matvec needs after the host-side build.
// -----------------------------------------------------------------------------

struct AxisFFT {
    int M = 0;                     // grid points along this axis
    int padded = 0;                // 2 * M (circulant embedding length)
    int nfreq = 0;                 // padded / 2 + 1 (R2C output length)
    cufftHandle plan_r2c = 0;      // batched length-padded forward plan
    cufftHandle plan_c2r = 0;      // batched length-padded inverse plan
    cufftComplex* d_kernel = nullptr;  // (nfreq) FFT of the circulant first column
};

struct SkiCudaState {
    int N = 0;
    int M = 0;
    int D = 0;
    std::vector<int> grid_sizes;
    std::vector<int> grid_strides;  // mirrors SKIGrid::strides()

    // Host-immutable CSR for W (size N) and W^T (size M).
    int*   d_W_row_ptr  = nullptr;
    int*   d_W_col_idx  = nullptr;
    float* d_W_values   = nullptr;

    int*   d_WT_row_ptr = nullptr;
    int*   d_WT_col_idx = nullptr;
    float* d_WT_values  = nullptr;

    std::vector<AxisFFT> axes;

    // Persistent workspace (sized to total grid points × padded[axis_max] / M_axis fibers).
    float* d_v    = nullptr;   // input host->device (N floats)
    float* d_u    = nullptr;   // W^T v scratch (M floats)
    float* d_pad  = nullptr;   // padded buffer used during axis sweeps (max fiber-count * padded floats)
    cufftComplex* d_freq = nullptr;  // freq-domain scratch (max fiber-count * nfreq complex)
    float* d_y    = nullptr;   // W (K_grid (W^T v)) result (N floats)
    int    pad_capacity = 0;
    int    freq_capacity = 0;
};

void SkiCudaStateDeleter::operator()(SkiCudaState* p) const noexcept {
    if (!p) return;
    cudaFree(p->d_W_row_ptr);
    cudaFree(p->d_W_col_idx);
    cudaFree(p->d_W_values);
    cudaFree(p->d_WT_row_ptr);
    cudaFree(p->d_WT_col_idx);
    cudaFree(p->d_WT_values);
    cudaFree(p->d_v);
    cudaFree(p->d_u);
    cudaFree(p->d_y);
    cudaFree(p->d_pad);
    cudaFree(p->d_freq);
    for (auto& axis : p->axes) {
        if (axis.plan_r2c) cufftDestroy(axis.plan_r2c);
        if (axis.plan_c2r) cufftDestroy(axis.plan_c2r);
        cudaFree(axis.d_kernel);
    }
    delete p;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace {

/// Transpose a CSR (rows x cols) into a CSR for its transpose (cols x rows).
/// Used so W^T v can be evaluated with the same CSR kernel.
void csr_transpose(const SparseMatrix& W,
                   std::vector<int>& Tptr,
                   std::vector<int>& Tidx,
                   std::vector<float>& Tval) {
    const int rows = W.rows, cols = W.cols;
    Tptr.assign(cols + 1, 0);
    for (int n : W.col_idx) Tptr[n + 1]++;
    for (int j = 1; j <= cols; ++j) Tptr[j] += Tptr[j - 1];

    const int nnz = static_cast<int>(W.values.size());
    Tidx.assign(nnz, 0);
    Tval.assign(nnz, 0.0f);
    std::vector<int> cursor(Tptr.begin(), Tptr.end() - 1);  // working positions per col
    for (int i = 0; i < rows; ++i) {
        for (int k = W.row_ptr[i]; k < W.row_ptr[i + 1]; ++k) {
            const int j = W.col_idx[k];
            const int dst = cursor[j]++;
            Tidx[dst] = i;
            Tval[dst] = W.values[k];
        }
    }
}

// CUDA kernels --------------------------------------------------------------

__global__ void csr_spmv_kernel(const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                const float* __restrict__ values,
                                const float* __restrict__ v,
                                float* __restrict__ y,
                                int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float s = 0.0f;
    const int s0 = row_ptr[i], s1 = row_ptr[i + 1];
    for (int k = s0; k < s1; ++k) s += values[k] * v[col_idx[k]];
    y[i] = s;
}

/// Build the length-`padded` real signal that is the first column of the 2M-point
/// circulant embedding of an M-point symmetric Toeplitz matrix with first column `col`.
/// The embedding follows the standard convention:
///   c[0..M-1]     = col[0..M-1]
///   c[M]          = 0           (any value works for the symmetric case; choose 0)
///   c[M+1..2M-1]  = col[M-1..1] (reversed tail, gives a real-symmetric circulant)
__global__ void build_circulant_kernel(const float* __restrict__ col,
                                       float* __restrict__ circ,
                                       int M, int padded) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded) return;
    if (i < M)             circ[i] = col[i];
    else if (i == M)       circ[i] = 0.0f;
    else                    circ[i] = col[padded - i];   // col[2M - i]
}

/// Copy `src` (length total = num_fibers * M_axis) into `dst` (length num_fibers * padded)
/// by writing each fiber of length M_axis into the first M_axis slots of a padded fiber,
/// zeroing the remaining padded - M_axis slots. Strides used to walk the original
/// flat layout where the chosen axis has stride `stride_axis`, and the next fiber starts
/// at increment `stride_outer` (= stride_axis * shape[axis]) — but we want to enumerate
/// fibers, so we encode each fiber's base offset as a precomputed table.
///
/// For simplicity we pass the precomputed base offsets in `bases`, length num_fibers.
__global__ void zero_pad_fibers_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       const int* __restrict__ bases,
                                       int num_fibers, int M_axis, int stride_axis,
                                       int padded) {
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_fibers || j >= padded) return;
    const int dst_idx = f * padded + j;
    if (j < M_axis) {
        const int src_idx = bases[f] + j * stride_axis;
        dst[dst_idx] = src[src_idx];
    } else {
        dst[dst_idx] = 0.0f;
    }
}

/// Scatter the first M_axis samples of each padded fiber back into the original flat
/// layout, normalising by 1/padded (cuFFT's inverse is unnormalised).
__global__ void unpad_and_scale_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       const int* __restrict__ bases,
                                       int num_fibers, int M_axis, int stride_axis,
                                       int padded) {
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_fibers || j >= M_axis) return;
    const float inv = 1.0f / static_cast<float>(padded);
    dst[bases[f] + j * stride_axis] = src[f * padded + j] * inv;
}

/// Pointwise (per-frequency) complex multiplication: freq_out = freq * kernel.
/// freq has shape (num_fibers, nfreq) and kernel has shape (nfreq,) — broadcast over fibers.
__global__ void complex_mul_broadcast_kernel(cufftComplex* __restrict__ freq,
                                             const cufftComplex* __restrict__ kernel,
                                             int num_fibers, int nfreq) {
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_fibers || k >= nfreq) return;
    const int idx = f * nfreq + k;
    const cufftComplex a = freq[idx];
    const cufftComplex b = kernel[k];
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    freq[idx] = c;
}

__global__ void axpy_kernel(const float* __restrict__ x, float a,
                            float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += a * x[i];
}

/// For every base-offset in {0..num_fibers-1}, build the flat base offset given the
/// non-axis coordinate. We compute these on the host (cheap, run once per fit) and
/// upload to device; the kernel just reads `bases`. Returns the table on host.
std::vector<int> compute_fiber_bases(const std::vector<int>& shape, int axis) {
    const int D = static_cast<int>(shape.size());
    std::vector<int> strides(D, 1);
    for (int d = D - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];
    int num_fibers = 1;
    for (int d = 0; d < D; ++d) if (d != axis) num_fibers *= shape[d];
    std::vector<int> bases(num_fibers);
    for (int f = 0; f < num_fibers; ++f) {
        int rem = f;
        int base = 0;
        for (int d = D - 1; d >= 0; --d) {
            if (d == axis) continue;
            const int idx_d = rem % shape[d];
            rem /= shape[d];
            base += idx_d * strides[d];
        }
        bases[f] = base;
    }
    return bases;
}

}  // namespace

// -----------------------------------------------------------------------------
// make_ski_cuda_state
// -----------------------------------------------------------------------------

SkiCudaStatePtr make_ski_cuda_state(const SKIGrid& grid,
                                    const SparseMatrix& W,
                                    const std::vector<Tensor>& toeplitz_cols) {
    if (!CudaContext::instance().available()) return nullptr;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(CudaContext::instance().stream());

    SkiCudaStatePtr st(new SkiCudaState());
    st->N = W.rows;
    st->M = W.cols;
    st->D = grid.D;
    st->grid_sizes = grid.grid_sizes;
    st->grid_strides = grid.strides();

    // Upload CSR for W.
    const int nnz = static_cast<int>(W.values.size());
    cudaMalloc(&st->d_W_row_ptr, sizeof(int) * (W.rows + 1));
    cudaMalloc(&st->d_W_col_idx, sizeof(int) * nnz);
    cudaMalloc(&st->d_W_values, sizeof(float) * nnz);
    cudaMemcpyAsync(st->d_W_row_ptr, W.row_ptr.data(), sizeof(int) * (W.rows + 1),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(st->d_W_col_idx, W.col_idx.data(), sizeof(int) * nnz,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(st->d_W_values, W.values.data(), sizeof(float) * nnz,
                    cudaMemcpyHostToDevice, stream);

    // Build and upload CSR for W^T.
    std::vector<int> Tptr, Tidx;
    std::vector<float> Tval;
    csr_transpose(W, Tptr, Tidx, Tval);
    cudaMalloc(&st->d_WT_row_ptr, sizeof(int) * (W.cols + 1));
    cudaMalloc(&st->d_WT_col_idx, sizeof(int) * nnz);
    cudaMalloc(&st->d_WT_values, sizeof(float) * nnz);
    cudaMemcpyAsync(st->d_WT_row_ptr, Tptr.data(), sizeof(int) * (W.cols + 1),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(st->d_WT_col_idx, Tidx.data(), sizeof(int) * nnz,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(st->d_WT_values, Tval.data(), sizeof(float) * nnz,
                    cudaMemcpyHostToDevice, stream);

    // Per-axis cuFFT setup: plan + FFT of the circulant first column.
    st->axes.resize(grid.D);
    int max_padded = 0;
    int max_nfreq = 0;
    int max_fibers = 0;
    for (int d = 0; d < grid.D; ++d) {
        AxisFFT& ax = st->axes[d];
        ax.M = grid.grid_sizes[d];
        ax.padded = 2 * ax.M;
        ax.nfreq = ax.padded / 2 + 1;
        if (ax.padded > max_padded) max_padded = ax.padded;
        if (ax.nfreq > max_nfreq) max_nfreq = ax.nfreq;
        int fibers = 1;
        for (int dd = 0; dd < grid.D; ++dd) if (dd != d) fibers *= grid.grid_sizes[dd];
        if (fibers > max_fibers) max_fibers = fibers;

        // Upload the host-side Toeplitz column, build circulant on device, FFT it.
        float* d_col = nullptr;
        cudaMalloc(&d_col, sizeof(float) * ax.M);
        cudaMemcpyAsync(d_col, toeplitz_cols[d].data(), sizeof(float) * ax.M,
                        cudaMemcpyHostToDevice, stream);
        float* d_circ = nullptr;
        cudaMalloc(&d_circ, sizeof(float) * ax.padded);
        {
            const int threads = 256;
            const int blocks = (ax.padded + threads - 1) / threads;
            build_circulant_kernel<<<blocks, threads, 0, stream>>>(d_col, d_circ, ax.M, ax.padded);
        }
        cudaMalloc(&ax.d_kernel, sizeof(cufftComplex) * ax.nfreq);

        // A single 1-batch R2C plan just to FFT the kernel; we'll create the batched
        // plans separately below for the per-axis sweep.
        cufftHandle one_r2c = 0;
        cufftPlan1d(&one_r2c, ax.padded, CUFFT_R2C, 1);
        cufftSetStream(one_r2c, stream);
        cufftExecR2C(one_r2c, d_circ, ax.d_kernel);
        cufftDestroy(one_r2c);
        cudaFree(d_col);
        cudaFree(d_circ);

        // Batched plans for the axis sweep (batch size = num fibers on this axis).
        int n[1] = {ax.padded};
        int inembed[1] = {ax.padded};
        int onembed[1] = {ax.nfreq};
        cufftPlanMany(&ax.plan_r2c, /*rank=*/1, n,
                      inembed, /*istride=*/1, /*idist=*/ax.padded,
                      onembed, /*ostride=*/1, /*odist=*/ax.nfreq,
                      CUFFT_R2C, /*batch=*/fibers);
        cufftSetStream(ax.plan_r2c, stream);
        cufftPlanMany(&ax.plan_c2r, /*rank=*/1, n,
                      onembed, /*istride=*/1, /*idist=*/ax.nfreq,
                      inembed, /*ostride=*/1, /*odist=*/ax.padded,
                      CUFFT_C2R, /*batch=*/fibers);
        cufftSetStream(ax.plan_c2r, stream);
    }

    // Persistent workspace sized to the largest axis pass.
    cudaMalloc(&st->d_v, sizeof(float) * st->N);
    cudaMalloc(&st->d_u, sizeof(float) * st->M);
    cudaMalloc(&st->d_y, sizeof(float) * st->N);
    st->pad_capacity = max_fibers * max_padded;
    st->freq_capacity = max_fibers * max_nfreq;
    cudaMalloc(&st->d_pad, sizeof(float) * st->pad_capacity);
    cudaMalloc(&st->d_freq, sizeof(cufftComplex) * st->freq_capacity);

    cudaStreamSynchronize(stream);
    return st;
}

// -----------------------------------------------------------------------------
// ski_matvec_cuda
// -----------------------------------------------------------------------------

Tensor ski_matvec_cuda(const SkiCudaState& s, const Tensor& v, float noise_variance) {
    auto& ctx = CudaContext::instance();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream());
    assert(static_cast<int>(v.rows()) == s.N);

    // 1. Upload v.
    cudaMemcpyAsync(s.d_v, v.data(), sizeof(float) * s.N, cudaMemcpyHostToDevice, stream);

    // 2. u = W^T v  (length M).
    {
        const int threads = 256;
        const int blocks = (s.M + threads - 1) / threads;
        csr_spmv_kernel<<<blocks, threads, 0, stream>>>(
            s.d_WT_row_ptr, s.d_WT_col_idx, s.d_WT_values, s.d_v, s.d_u, s.M);
    }

    // 3. Apply K_d along each axis. We bounce between d_u and (after the last pass)
    //    keep the result in d_u.
    for (int d = 0; d < s.D; ++d) {
        const AxisFFT& ax = s.axes[d];
        const int M_axis = ax.M;
        const int stride_axis = s.grid_strides[d];
        int num_fibers = 1;
        for (int dd = 0; dd < s.D; ++dd) if (dd != d) num_fibers *= s.grid_sizes[dd];

        // Upload bases for this axis (cheap; rebuilds each call — TODO cache in state).
        std::vector<int> bases = compute_fiber_bases(s.grid_sizes, d);
        int* d_bases = nullptr;
        cudaMalloc(&d_bases, sizeof(int) * num_fibers);
        cudaMemcpyAsync(d_bases, bases.data(), sizeof(int) * num_fibers,
                        cudaMemcpyHostToDevice, stream);

        // Zero-pad fibers from d_u into d_pad.
        {
            dim3 block(64, 4);
            dim3 grid((ax.padded + block.x - 1) / block.x,
                      (num_fibers + block.y - 1) / block.y);
            zero_pad_fibers_kernel<<<grid, block, 0, stream>>>(
                s.d_u, s.d_pad, d_bases, num_fibers, M_axis, stride_axis, ax.padded);
        }
        // Forward FFT.
        cufftExecR2C(ax.plan_r2c, s.d_pad, s.d_freq);
        // Pointwise multiply by kernel.
        {
            dim3 block(64, 4);
            dim3 grid((ax.nfreq + block.x - 1) / block.x,
                      (num_fibers + block.y - 1) / block.y);
            complex_mul_broadcast_kernel<<<grid, block, 0, stream>>>(
                s.d_freq, ax.d_kernel, num_fibers, ax.nfreq);
        }
        // Inverse FFT.
        cufftExecC2R(ax.plan_c2r, s.d_freq, s.d_pad);
        // Unpad + scale back into d_u (in place along this axis).
        {
            dim3 block(64, 4);
            dim3 grid((M_axis + block.x - 1) / block.x,
                      (num_fibers + block.y - 1) / block.y);
            unpad_and_scale_kernel<<<grid, block, 0, stream>>>(
                s.d_pad, s.d_u, d_bases, num_fibers, M_axis, stride_axis, ax.padded);
        }
        cudaFree(d_bases);
    }

    // 4. y = W u  (length N).
    {
        const int threads = 256;
        const int blocks = (s.N + threads - 1) / threads;
        csr_spmv_kernel<<<blocks, threads, 0, stream>>>(
            s.d_W_row_ptr, s.d_W_col_idx, s.d_W_values, s.d_u, s.d_y, s.N);
    }

    // 5. y += sn2 * v.
    if (noise_variance != 0.0f) {
        const int threads = 256;
        const int blocks = (s.N + threads - 1) / threads;
        axpy_kernel<<<blocks, threads, 0, stream>>>(s.d_v, noise_variance, s.d_y, s.N);
    }

    // Download.
    Tensor out(s.N, 1);
    cudaMemcpyAsync(out.data(), s.d_y, sizeof(float) * s.N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return out;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
