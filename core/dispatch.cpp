#include "dispatch.h"

#include <cstdio>

#include "../kernels/cpu/matern_cpu.h"
#include "../kernels/cpu/rbf_cpu.h"
#include "../solvers/cpu/cholesky_cpu.h"
#if defined(LIGHTGP_HAS_ACCELERATE) || defined(LIGHTGP_HAS_OPENBLAS)
#include "blas_accel.h"
#define LIGHTGP_HAS_BLAS 1
#endif
#ifdef LIGHTGP_HAS_METAL
#include "../kernels/metal/matern_metal.h"
#include "../kernels/metal/metal_context.h"
#include "../kernels/metal/rbf_metal.h"
#include "../solvers/metal/cholesky_metal.h"
#include "../solvers/metal/cholesky_solve_metal.h"
#endif
#ifdef LIGHTGP_HAS_CUDA
#include "../kernels/cuda/cuda_context.h"
#include "../kernels/cuda/matern_cuda.h"
#include "../kernels/cuda/rbf_cuda.h"
#include "../solvers/cuda/cholesky_cuda.h"
#include "../solvers/cuda/cholesky_solve_cuda.h"
#endif

namespace lightgp {

namespace {

void warn_once(const char* msg, bool& flag) {
    if (!flag) {
        std::fprintf(stderr, "[lightgp] %s\n", msg);
        flag = true;
    }
}

}  // namespace

Tensor dispatch_kernel(const Tensor& X1, const Tensor& X2,
                       float length_scale, float signal_variance,
                       KernelType type, Backend backend) {
    if (type == KernelType::RBF) {
        return dispatch_rbf_kernel(X1, X2, length_scale, signal_variance, backend);
    }
    // Matern: Metal supported via function-constant-specialized PSOs; CUDA still falls back.
    switch (backend) {
        case Backend::Metal: {
#ifdef LIGHTGP_HAS_METAL
            if (MetalContext::instance().available()) {
                return matern_kernel_metal(X1, X2, length_scale, signal_variance, type);
            }
            static bool warned = false;
            warn_once("Metal Matern requested but device unavailable; using CPU", warned);
            return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
#else
            static bool warned = false;
            warn_once("Metal Matern requested but not compiled in; using CPU", warned);
            return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
#endif
        }
        case Backend::CUDA: {
#ifdef LIGHTGP_HAS_CUDA
            if (CudaContext::instance().available()) {
                return matern_kernel_cuda(X1, X2, length_scale, signal_variance, type);
            }
            static bool warned = false;
            warn_once("CUDA Matern requested but device unavailable; using CPU", warned);
            return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
#else
            static bool warned = false;
            warn_once("CUDA Matern requested but not compiled in; using CPU", warned);
            return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
#endif
        }
        case Backend::CPU:
        default:
            return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
    }
}

Tensor dispatch_rbf_kernel(const Tensor& X1, const Tensor& X2,
                           float length_scale, float signal_variance,
                           Backend backend) {
    switch (backend) {
        case Backend::Metal: {
#ifdef LIGHTGP_HAS_METAL
            if (MetalContext::instance().available()) {
                return rbf_kernel_metal(X1, X2, length_scale, signal_variance);
            }
            static bool warned = false;
            warn_once("Metal backend requested but device unavailable; using CPU", warned);
            return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
#else
            static bool warned = false;
            warn_once("Metal backend requested but not compiled in; using CPU", warned);
            return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
#endif
        }
        case Backend::CUDA: {
#ifdef LIGHTGP_HAS_CUDA
            if (CudaContext::instance().available()) {
                return rbf_kernel_cuda(X1, X2, length_scale, signal_variance);
            }
            static bool warned = false;
            warn_once("CUDA RBF requested but device unavailable; using CPU", warned);
            return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
#else
            static bool warned = false;
            warn_once("CUDA RBF requested but not compiled in; using CPU", warned);
            return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
#endif
        }
        case Backend::CPU:
        default:
            return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
    }
}

bool dispatch_cholesky_with_jitter(const Tensor& K, Tensor& L, float& jitter_used,
                                   Backend backend, int block_size) {
    switch (backend) {
        case Backend::Metal: {
#ifdef LIGHTGP_HAS_METAL
            if (MetalContext::instance().available()) {
                return cholesky_metal_with_jitter(K, L, jitter_used, block_size);
            }
            static bool warned = false;
            warn_once("Metal Cholesky requested but device unavailable; using CPU", warned);
            return cholesky_with_jitter(K, L, jitter_used);
#else
            static bool warned = false;
            warn_once("Metal Cholesky requested but not compiled in; using CPU", warned);
            return cholesky_with_jitter(K, L, jitter_used);
#endif
        }
        case Backend::CUDA: {
#ifdef LIGHTGP_HAS_CUDA
            if (CudaContext::instance().available()) {
                return cholesky_cuda_with_jitter(K, L, jitter_used);
            }
            static bool warned = false;
            warn_once("CUDA Cholesky requested but device unavailable; using CPU", warned);
            return cholesky_with_jitter(K, L, jitter_used);
#else
            static bool warned = false;
            warn_once("CUDA Cholesky requested but not compiled in; using CPU", warned);
            return cholesky_with_jitter(K, L, jitter_used);
#endif
        }
        case Backend::CPU:
        default:
            return cholesky_with_jitter(K, L, jitter_used);
    }
}

Tensor dispatch_cholesky_solve(const Tensor& L, const Tensor& b, Backend backend) {
    switch (backend) {
        case Backend::Metal: {
#ifdef LIGHTGP_HAS_METAL
            if (MetalContext::instance().available()) {
                return cholesky_solve_metal(L, b);
            }
            static bool warned = false;
            warn_once("Metal cholesky_solve requested but device unavailable; using CPU", warned);
            return cholesky_solve(L, b);
#else
            static bool warned = false;
            warn_once("Metal cholesky_solve requested but not compiled in; using CPU", warned);
            return cholesky_solve(L, b);
#endif
        }
        case Backend::CUDA: {
#ifdef LIGHTGP_HAS_CUDA
            if (CudaContext::instance().available()) {
                return cholesky_solve_cuda(L, b);
            }
            static bool warned = false;
            warn_once("CUDA cholesky_solve requested but device unavailable; using CPU", warned);
            return cholesky_solve(L, b);
#else
            static bool warned = false;
            warn_once("CUDA cholesky_solve requested but not compiled in; using CPU", warned);
            return cholesky_solve(L, b);
#endif
        }
        case Backend::CPU:
        default:
            return cholesky_solve(L, b);
    }
}

namespace {

Tensor forward_solve_cpu_local(const Tensor& L, const Tensor& B) {
#ifdef LIGHTGP_HAS_BLAS
    Tensor Y = B;
    trsm_lower_no_trans_accelerate(L, Y);
    return Y;
#else
    const std::size_t n = L.rows();
    const std::size_t m = B.cols();
    Tensor Y(n, m);
    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            float s = B(i, col);
            for (std::size_t j = 0; j < i; ++j) s -= L(i, j) * Y(j, col);
            Y(i, col) = s / L(i, i);
        }
    }
    return Y;
#endif
}

}  // namespace

Tensor dispatch_forward_solve(const Tensor& L, const Tensor& b, Backend backend) {
    switch (backend) {
        case Backend::Metal: {
#ifdef LIGHTGP_HAS_METAL
            if (MetalContext::instance().available()) {
                return forward_solve_metal(L, b);
            }
            static bool warned = false;
            warn_once("Metal forward_solve requested but device unavailable; using CPU", warned);
            return forward_solve_cpu_local(L, b);
#else
            static bool warned = false;
            warn_once("Metal forward_solve requested but not compiled in; using CPU", warned);
            return forward_solve_cpu_local(L, b);
#endif
        }
        case Backend::CUDA: {
#ifdef LIGHTGP_HAS_CUDA
            if (CudaContext::instance().available()) {
                return forward_solve_cuda(L, b);
            }
            static bool warned = false;
            warn_once("CUDA forward_solve requested but device unavailable; using CPU", warned);
            return forward_solve_cpu_local(L, b);
#else
            static bool warned = false;
            warn_once("CUDA forward_solve requested but not compiled in; using CPU", warned);
            return forward_solve_cpu_local(L, b);
#endif
        }
        case Backend::CPU:
        default:
            return forward_solve_cpu_local(L, b);
    }
}

}  // namespace lightgp
