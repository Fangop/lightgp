// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#ifdef LIGHTGP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstring>

#include "../cpu/rbf_cpu.h"
#include "metal_context.h"
#include "rbf_matvec.h"

namespace lightgp {

Tensor rbf_matvec_metal(const Tensor& X, const Tensor& v,
                        float length_scale, float signal_variance, float noise_variance) {
    assert(X.rows() == v.rows());
    assert(v.cols() == 1);

    MetalContext& ctx = MetalContext::instance();
    if (!ctx.available()) {
        Tensor K = rbf_kernel_cpu(X, X, length_scale, signal_variance);
        K.add_jitter(noise_variance);
        return K.matmul(v);
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.command_queue();
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)ctx.rbf_matvec_pipeline();

        const std::uint32_t N = static_cast<std::uint32_t>(X.rows());
        const std::uint32_t D = static_cast<std::uint32_t>(X.cols());
        const float inv_2l2 = 0.5f / (length_scale * length_scale);

        const NSUInteger x_bytes = sizeof(float) * X.size();
        const NSUInteger v_bytes = sizeof(float) * v.size();
        const NSUInteger w_bytes = sizeof(float) * N;

        id<MTLBuffer> bX = [device newBufferWithBytes:X.data()
                                                length:x_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV = [device newBufferWithBytes:v.data()
                                                length:v_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bW = [device newBufferWithLength:w_bytes
                                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:0 atIndex:0];
        [enc setBuffer:bV offset:0 atIndex:1];
        [enc setBuffer:bW offset:0 atIndex:2];
        [enc setBytes:&N length:sizeof(N) atIndex:3];
        [enc setBytes:&D length:sizeof(D) atIndex:4];
        [enc setBytes:&inv_2l2 length:sizeof(inv_2l2) atIndex:5];
        [enc setBytes:&signal_variance length:sizeof(signal_variance) atIndex:6];
        [enc setBytes:&noise_variance length:sizeof(noise_variance) atIndex:7];

        // 64 threads per threadgroup, one threadgroup per 64-element output chunk.
        const NSUInteger tg = 64;
        const NSUInteger groups = (N + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        Tensor w(N, 1);
        std::memcpy(w.data(), bW.contents, w_bytes);
        return w;
    }
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
