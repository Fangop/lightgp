// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#ifdef LIGHTGP_HAS_METAL

#include <string>

namespace lightgp {

/// Singleton holding the Metal device, command queue, and compiled GP kernel pipelines.
/// Stores Objective-C objects as void* (retained under ARC) so this header is pure C++.
class MetalContext {
public:
    /// Lazy-initialized singleton.
    static MetalContext& instance();

    /// True once device + queue + library + all pipelines initialized successfully.
    bool available() const { return rbf_pipeline_ != nullptr; }
    /// id<MTLDevice> as void*.
    void* device() const { return device_; }
    /// id<MTLCommandQueue> as void*.
    void* command_queue() const { return command_queue_; }
    /// Small-D RBF pipeline (single-barrier full-row strip; requires D <= 16).
    void* rbf_pipeline_small_d() const { return rbf_pipeline_small_d_; }
    /// Scalar tiled RBF pipeline (works for any D).
    void* rbf_pipeline() const { return rbf_pipeline_; }
    /// Float4 tiled RBF pipeline (requires D % 4 == 0).
    void* rbf_pipeline_f4() const { return rbf_pipeline_f4_; }
    /// Naive GEMM pipeline (C = A * B); fallback for small matrices.
    void* gemm_pipeline() const { return gemm_pipeline_; }
    /// Tiled GEMM pipeline (64x64 output tile, 4x4 micro-tile per thread).
    void* gemm_tiled_pipeline() const { return gemm_tiled_pipeline_; }
    /// simdgroup_matrix_multiply GEMM (8x8 hardware tiles, Apple9+). May be null on older GPUs.
    void* gemm_simdgroup_pipeline() const { return gemm_simdgroup_pipeline_; }
    /// Forward substitution pipeline (solve L * X = B).
    void* trsv_forward_pipeline() const { return trsv_forward_pipeline_; }
    /// Backward substitution pipeline (solve L^T * X = B).
    void* trsv_backward_pipeline() const { return trsv_backward_pipeline_; }
    /// Matrix-free RBF kernel-vector product pipeline (w = (K + sn2 I) v).
    void* rbf_matvec_pipeline() const { return rbf_matvec_pipeline_; }
    /// Matern kernel matrix PSO for `variant_idx` ∈ {0,1,2} (Matern12/32/52)
    /// and `shape_idx` ∈ {0,1,2} (small_d, tiled, tiled_f4). Null if init failed.
    void* matern_kernel_pipeline(int variant_idx, int shape_idx) const {
        return matern_kernel_pipelines_[variant_idx][shape_idx];
    }
    /// Last error message; empty if initialization succeeded.
    const std::string& error() const { return error_; }

private:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    void* device_ = nullptr;
    void* command_queue_ = nullptr;
    void* library_ = nullptr;
    void* rbf_pipeline_small_d_ = nullptr;
    void* rbf_pipeline_ = nullptr;
    void* rbf_pipeline_f4_ = nullptr;
    void* gemm_pipeline_ = nullptr;
    void* gemm_tiled_pipeline_ = nullptr;
    void* gemm_simdgroup_pipeline_ = nullptr;
    void* trsv_forward_pipeline_ = nullptr;
    void* trsv_backward_pipeline_ = nullptr;
    void* rbf_matvec_pipeline_ = nullptr;
    void* matern_kernel_pipelines_[3][3] = {{nullptr, nullptr, nullptr},
                                            {nullptr, nullptr, nullptr},
                                            {nullptr, nullptr, nullptr}};
    std::string error_;
};

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
