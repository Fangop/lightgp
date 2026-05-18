// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#include "cuda_context.h"

#ifdef LIGHTGP_HAS_CUDA

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace lightgp {

CudaContext& CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

CudaContext::CudaContext() {
    int dev_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&dev_count);
    if (cerr != cudaSuccess || dev_count == 0) {
        error_ = cerr != cudaSuccess ? cudaGetErrorString(cerr)
                                     : "no CUDA devices visible";
        return;
    }
    if ((cerr = cudaSetDevice(device_)) != cudaSuccess) {
        error_ = std::string("cudaSetDevice: ") + cudaGetErrorString(cerr);
        return;
    }
    cudaStream_t stream = nullptr;
    if ((cerr = cudaStreamCreate(&stream)) != cudaSuccess) {
        error_ = std::string("cudaStreamCreate: ") + cudaGetErrorString(cerr);
        return;
    }
    stream_ = stream;

    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        error_ = "cublasCreate failed";
        cudaStreamDestroy(stream);
        stream_ = nullptr;
        return;
    }
    cublasSetStream(blas, stream);
    cublas_ = blas;

    cusolverDnHandle_t solver = nullptr;
    if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) {
        error_ = "cusolverDnCreate failed";
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        cublas_ = nullptr;
        stream_ = nullptr;
        return;
    }
    cusolverDnSetStream(solver, stream);
    cusolver_ = solver;

    available_ = true;
}

CudaContext::~CudaContext() {
    if (cusolver_) cusolverDnDestroy(reinterpret_cast<cusolverDnHandle_t>(cusolver_));
    if (cublas_) cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_));
    if (stream_) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
