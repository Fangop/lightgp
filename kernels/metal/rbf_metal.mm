// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#ifdef LIGHTGP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstring>

#include "metal_context.h"
#include "rbf_metal.h"
#include "../cpu/rbf_cpu.h"

namespace lightgp {

Tensor rbf_kernel_metal(const Tensor& X1, const Tensor& X2,
                        float length_scale, float signal_variance) {
    assert(X1.cols() == X2.cols());

    MetalContext& ctx = MetalContext::instance();
    if (!ctx.available()) {
        // Transparent fallback so callers don't have to guard every site.
        return rbf_kernel_cpu(X1, X2, length_scale, signal_variance);
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.command_queue();

        // Dispatch: small-D (single barrier) for D <= 16, float4 tiled when D % 4 == 0,
        // otherwise scalar tiled. The small-D path wins for typical low-dim GP problems.
        const std::size_t D_cols = X1.cols();
        id<MTLComputePipelineState> pso;
        if (D_cols <= 16) {
            pso = (__bridge id<MTLComputePipelineState>)ctx.rbf_pipeline_small_d();
        } else if (D_cols % 4u == 0u) {
            pso = (__bridge id<MTLComputePipelineState>)ctx.rbf_pipeline_f4();
        } else {
            pso = (__bridge id<MTLComputePipelineState>)ctx.rbf_pipeline();
        }

        const std::uint32_t N = static_cast<std::uint32_t>(X1.rows());
        const std::uint32_t M = static_cast<std::uint32_t>(X2.rows());
        const std::uint32_t D = static_cast<std::uint32_t>(X1.cols());
        const float inv_2l2 = 0.5f / (length_scale * length_scale);

        const NSUInteger x1_bytes = sizeof(float) * X1.size();
        const NSUInteger x2_bytes = sizeof(float) * X2.size();
        const NSUInteger k_bytes = sizeof(float) * static_cast<std::size_t>(N) * M;

        id<MTLBuffer> bx1 = [device newBufferWithBytes:X1.data()
                                                length:x1_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx2 = [device newBufferWithBytes:X2.data()
                                                length:x2_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [device newBufferWithLength:k_bytes
                                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx1 offset:0 atIndex:0];
        [enc setBuffer:bx2 offset:0 atIndex:1];
        [enc setBuffer:bk  offset:0 atIndex:2];
        [enc setBytes:&N length:sizeof(N) atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        [enc setBytes:&D length:sizeof(D) atIndex:5];
        [enc setBytes:&inv_2l2 length:sizeof(inv_2l2) atIndex:6];
        [enc setBytes:&signal_variance length:sizeof(signal_variance) atIndex:7];

        // Threadgroup = 16x16; grid covers (M, N). The tiled kernels require all 256
        // threads to participate in cooperative loads, so we pad up with dispatchThreadgroups
        // and rely on per-thread bounds checks inside the shader.
        const NSUInteger tg_w = 16;
        const NSUInteger tg_h = 16;
        MTLSize threadgroup = MTLSizeMake(tg_w, tg_h, 1);
        MTLSize groups = MTLSizeMake((M + tg_w - 1) / tg_w,
                                     (N + tg_h - 1) / tg_h, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        Tensor K(N, M);
        std::memcpy(K.data(), bk.contents, k_bytes);
        return K;
    }
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
