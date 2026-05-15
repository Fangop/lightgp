#pragma once

#ifdef LIGHTGP_HAS_CUDA

#include <string>

namespace lightgp {

/// Singleton holding the CUDA device, stream, cuBLAS handle and cuSOLVER handle.
/// Mirrors `MetalContext` so dispatch code can pick the right backend without knowing
/// CUDA types — the handles are exposed as `void*` and reinterpret_cast'd inside the
/// .cu files that actually call cuBLAS / cuSOLVER.
class CudaContext {
public:
    /// Lazy-initialized singleton; safe to call from any thread.
    static CudaContext& instance();

    /// True iff cudaSetDevice, cudaStreamCreate, cublasCreate and cusolverDnCreate all
    /// succeeded. False on hosts without a working CUDA driver / device.
    bool available() const { return available_; }

    /// Device index this context is bound to (always 0 in the current build).
    int device() const { return device_; }

    /// `cudaStream_t` as void*. All cuBLAS / cuSOLVER calls run on this stream.
    void* stream() const { return stream_; }
    /// `cublasHandle_t` as void*.
    void* cublas() const { return cublas_; }
    /// `cusolverDnHandle_t` as void*.
    void* cusolver() const { return cusolver_; }

    /// Last error message; empty when `available() == true`.
    const std::string& error() const { return error_; }

private:
    CudaContext();
    ~CudaContext();
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    bool available_ = false;
    int device_ = 0;
    void* stream_ = nullptr;
    void* cublas_ = nullptr;
    void* cusolver_ = nullptr;
    std::string error_;
};

}  // namespace lightgp

#endif  // LIGHTGP_HAS_CUDA
