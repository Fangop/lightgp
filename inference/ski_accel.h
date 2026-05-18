// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#ifdef LIGHTGP_HAS_ACCELERATE

#include <memory>
#include <vector>

#include "../core/tensor.h"

namespace lightgp {

/// Symmetric-Toeplitz matvec via Accelerate vDSP FFT — O(M log M) per matvec.
///
/// Internally embeds the M×M symmetric Toeplitz matrix as a 2M-point circulant
/// matrix (mirror the first column), precomputes the FFT of that circulant
/// kernel at construction, and at each matvec FFTs the input, multiplies in the
/// frequency domain, inverse-FFTs, and returns the first M elements of the real
/// part.
///
/// The forward and inverse vDSP DFT setups are held until destruction.
class ToeplitzFFTCpu {
public:
    /// Build from the first column of the Toeplitz matrix (length M, symmetric).
    explicit ToeplitzFFTCpu(const Tensor& toeplitz_col);
    ~ToeplitzFFTCpu();

    ToeplitzFFTCpu(const ToeplitzFFTCpu&) = delete;
    ToeplitzFFTCpu& operator=(const ToeplitzFFTCpu&) = delete;
    ToeplitzFFTCpu(ToeplitzFFTCpu&&) noexcept;
    ToeplitzFFTCpu& operator=(ToeplitzFFTCpu&&) noexcept;

    /// y = T v, where T is the symmetric Toeplitz matrix with first column = ctor arg.
    /// v must be (M, 1). Returns (M, 1).
    Tensor matvec(const Tensor& v) const;

    int M() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Multi-axis Kronecker-Toeplitz matvec: apply one ToeplitzFFTCpu per axis to
/// every fiber, in axis order. Matches kron_toeplitz_matvec_cpu's semantics
/// but with FFT-accelerated per-axis Toeplitz matvecs.
Tensor kron_toeplitz_matvec_accelerate(
    const std::vector<std::unique_ptr<ToeplitzFFTCpu>>& fft_plans,
    const std::vector<int>& grid_sizes,
    const Tensor& v);

}  // namespace lightgp

#endif  // LIGHTGP_HAS_ACCELERATE
