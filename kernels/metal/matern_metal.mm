#ifdef LIGHTGP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstring>

#include "matern_metal.h"
#include "metal_context.h"
#include "../cpu/matern_cpu.h"

namespace lightgp {

namespace {

int variant_index(KernelType type) {
    switch (type) {
        case KernelType::Matern12: return 0;
        case KernelType::Matern32: return 1;
        case KernelType::Matern52: return 2;
        default: return -1;
    }
}

}  // namespace

Tensor matern_kernel_metal(const Tensor& X1, const Tensor& X2,
                           float length_scale, float signal_variance,
                           KernelType type) {
    assert(X1.cols() == X2.cols());
    assert(type != KernelType::RBF);

    MetalContext& ctx = MetalContext::instance();
    const int v = variant_index(type);
    if (v < 0 || !ctx.available()) {
        return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
    }

    // Shape dispatch mirrors rbf_kernel_metal: small_d for D <= 16, float4 for D % 4 == 0
    // (else scalar tiled). One PSO per (variant, shape) pre-compiled via function constants.
    const std::size_t D_cols = X1.cols();
    int shape;
    if (D_cols <= 16) shape = 0;
    else if (D_cols % 4u == 0u) shape = 2;
    else shape = 1;

    void* psoptr = ctx.matern_kernel_pipeline(v, shape);
    if (!psoptr) {
        return matern_kernel_cpu(X1, X2, length_scale, signal_variance, type);
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.command_queue();
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)psoptr;

        const std::uint32_t N = static_cast<std::uint32_t>(X1.rows());
        const std::uint32_t M = static_cast<std::uint32_t>(X2.rows());
        const std::uint32_t D = static_cast<std::uint32_t>(D_cols);
        const float inv_l = 1.0f / length_scale;

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
        [enc setBytes:&inv_l length:sizeof(inv_l) atIndex:6];
        [enc setBytes:&signal_variance length:sizeof(signal_variance) atIndex:7];

        MTLSize threadgroup = MTLSizeMake(16, 16, 1);
        MTLSize groups = MTLSizeMake((M + 15) / 16, (N + 15) / 16, 1);
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
