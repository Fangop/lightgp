// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#ifdef LIGHTGP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstring>

#include "gemm_metal.h"
#include "metal_context.h"

namespace lightgp {

Tensor gemm_metal(const Tensor& A, const Tensor& B) {
    assert(A.cols() == B.rows());

    MetalContext& ctx = MetalContext::instance();
    if (!ctx.available()) {
        return A.matmul(B);
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.command_queue();

        const std::uint32_t M_dim = static_cast<std::uint32_t>(A.rows());
        const std::uint32_t N_dim = static_cast<std::uint32_t>(B.cols());
        const std::uint32_t K_dim = static_cast<std::uint32_t>(A.cols());

        // Dispatch tiers:
        //   simdgroup_matrix path: hardware 8x8 fp32 tiles via simdgroup_multiply_accumulate,
        //                          32x32 output per threadgroup, requires Apple9-class GPU + Metal 2.3+.
        //   tiled path:            scalar 4x4 micro-tile, 64x64 output per threadgroup.
        //   naive path:            one thread per C element, for small matrices.
        // Empirically, the 32x32 simdgroup tile only beats the 64x64 scalar tile once
        // both dims pass ~3000 (per bench_cholesky on M4). Below that, dispatch overhead
        // and per-threadgroup work imbalance make it lose to the scalar tiled kernel.
        const bool sg_loaded = (ctx.gemm_simdgroup_pipeline() != nullptr);
        const bool use_simdgroup = (sg_loaded && M_dim >= 3072 && N_dim >= 3072);
        const bool use_tiled = !use_simdgroup && (M_dim >= 64 && N_dim >= 64);

        id<MTLComputePipelineState> pso;
        if (use_simdgroup) {
            pso = (__bridge id<MTLComputePipelineState>)ctx.gemm_simdgroup_pipeline();
        } else if (use_tiled) {
            pso = (__bridge id<MTLComputePipelineState>)ctx.gemm_tiled_pipeline();
        } else {
            pso = (__bridge id<MTLComputePipelineState>)ctx.gemm_pipeline();
        }

        const NSUInteger a_bytes = sizeof(float) * A.size();
        const NSUInteger b_bytes = sizeof(float) * B.size();
        const NSUInteger c_bytes = sizeof(float)
            * static_cast<std::size_t>(M_dim) * N_dim;

        id<MTLBuffer> bA = [device newBufferWithBytes:A.data()
                                                length:a_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [device newBufferWithBytes:B.data()
                                                length:b_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [device newBufferWithLength:c_bytes
                                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bA offset:0 atIndex:0];
        [enc setBuffer:bB offset:0 atIndex:1];
        [enc setBuffer:bC offset:0 atIndex:2];
        [enc setBytes:&M_dim length:sizeof(M_dim) atIndex:3];
        [enc setBytes:&N_dim length:sizeof(N_dim) atIndex:4];
        [enc setBytes:&K_dim length:sizeof(K_dim) atIndex:5];

        MTLSize threadgroup;
        MTLSize groups;
        if (use_simdgroup) {
            // 32 threads/simdgroup × 16 simdgroups = 512 threads, 32x32 output tile.
            threadgroup = MTLSizeMake(32, 16, 1);
            groups = MTLSizeMake((N_dim + 31) / 32, (M_dim + 31) / 32, 1);
        } else if (use_tiled) {
            threadgroup = MTLSizeMake(16, 16, 1);
            groups = MTLSizeMake((N_dim + 63) / 64, (M_dim + 63) / 64, 1);
        } else {
            threadgroup = MTLSizeMake(16, 16, 1);
            groups = MTLSizeMake((N_dim + 15) / 16, (M_dim + 15) / 16, 1);
        }
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        Tensor C(M_dim, N_dim);
        std::memcpy(C.data(), bC.contents, c_bytes);
        return C;
    }
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
