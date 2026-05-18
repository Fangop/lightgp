// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#ifdef LIGHTGP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstring>

#include "cholesky_solve_metal.h"
#include "../../kernels/metal/metal_context.h"
#include "../cpu/cholesky_cpu.h"

namespace lightgp {

namespace {

Tensor forward_solve_cpu(const Tensor& L, const Tensor& B) {
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
}

}  // namespace

Tensor forward_solve_metal(const Tensor& L, const Tensor& b) {
    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());

    MetalContext& ctx = MetalContext::instance();
    if (!ctx.available()) {
        return forward_solve_cpu(L, b);
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.command_queue();
        id<MTLComputePipelineState> fwd =
            (__bridge id<MTLComputePipelineState>)ctx.trsv_forward_pipeline();

        const std::uint32_t N = static_cast<std::uint32_t>(L.rows());
        const std::uint32_t M = static_cast<std::uint32_t>(b.cols());
        const NSUInteger l_bytes = sizeof(float) * L.size();
        const NSUInteger b_bytes = sizeof(float) * b.size();

        id<MTLBuffer> bL = [device newBufferWithBytes:L.data()
                                                length:l_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [device newBufferWithBytes:b.data()
                                                length:b_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bY = [device newBufferWithLength:b_bytes
                                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:fwd];
        [enc setBuffer:bL offset:0 atIndex:0];
        [enc setBuffer:bB offset:0 atIndex:1];
        [enc setBuffer:bY offset:0 atIndex:2];
        [enc setBytes:&N length:sizeof(N) atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        MTLSize tg = MTLSizeMake(64, 1, 1);
        MTLSize grid = MTLSizeMake(M, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        Tensor Y(N, M);
        std::memcpy(Y.data(), bY.contents, b_bytes);
        return Y;
    }
}

Tensor cholesky_solve_metal(const Tensor& L, const Tensor& b) {
    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());

    MetalContext& ctx = MetalContext::instance();
    if (!ctx.available()) {
        return cholesky_solve(L, b);
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.command_queue();
        id<MTLComputePipelineState> fwd =
            (__bridge id<MTLComputePipelineState>)ctx.trsv_forward_pipeline();
        id<MTLComputePipelineState> bwd =
            (__bridge id<MTLComputePipelineState>)ctx.trsv_backward_pipeline();

        const std::uint32_t N = static_cast<std::uint32_t>(L.rows());
        const std::uint32_t M = static_cast<std::uint32_t>(b.cols());
        const NSUInteger l_bytes = sizeof(float) * L.size();
        const NSUInteger b_bytes = sizeof(float) * b.size();

        id<MTLBuffer> bL = [device newBufferWithBytes:L.data()
                                                length:l_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [device newBufferWithBytes:b.data()
                                                length:b_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bY = [device newBufferWithLength:b_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bX = [device newBufferWithLength:b_bytes
                                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];

        // Forward: L * Y = B.
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:fwd];
            [enc setBuffer:bL offset:0 atIndex:0];
            [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bY offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(N) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            MTLSize tg = MTLSizeMake(64, 1, 1);
            MTLSize grid = MTLSizeMake(M, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        // Backward: L^T * X = Y.
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:bwd];
            [enc setBuffer:bL offset:0 atIndex:0];
            [enc setBuffer:bY offset:0 atIndex:1];
            [enc setBuffer:bX offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(N) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            MTLSize tg = MTLSizeMake(64, 1, 1);
            MTLSize grid = MTLSizeMake(M, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        Tensor X(N, M);
        std::memcpy(X.data(), bX.contents, b_bytes);
        return X;
    }
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_METAL
