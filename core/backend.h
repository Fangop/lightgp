// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <cstddef>

#include "solver.h"

namespace lightgp {

/// Compute backend selector. `Auto` resolves to the empirically-best choice for the
/// problem shape via `resolve_auto_backend()`. CPU is always available, GPU backends
/// are conditional on `LIGHTGP_HAS_METAL` / `LIGHTGP_HAS_CUDA`.
enum class Backend { CPU, Metal, CUDA, Auto };

#ifdef LIGHTGP_HAS_METAL
inline constexpr bool has_metal = true;
#else
inline constexpr bool has_metal = false;
#endif

#ifdef LIGHTGP_HAS_CUDA
inline constexpr bool has_cuda = true;
#else
inline constexpr bool has_cuda = false;
#endif

/// Default backend resolution: prefer Metal if compiled in, else CPU.
inline Backend default_backend() {
    return has_metal ? Backend::Metal : Backend::CPU;
}

/// Empirically-derived dispatch rules (see report.md week 5 numbers):
///   - CG solver: Metal wins (matrix-free matvec) once N exceeds ~2000.
///   - Cholesky solver: Accelerate CPU + AMX wins at all measured sizes for low D.
///     At D >= 16 Metal kernel construction starts to dominate the savings; at
///     D >= 16 and N >= 2000 Metal wins net.
///   - On Linux with CUDA: GPU wins from N ≥ ~1000 for Cholesky (cuSOLVER spotrf
///     beats OpenBLAS spotrf by 100×+ past N=2048) and SKI / CG are always GPU.
inline Backend resolve_auto_backend(std::size_t n, std::size_t d, Solver solver) {
    if (has_cuda) {
        if (solver != Solver::Cholesky) return Backend::CUDA;
        if (n >= 1024) return Backend::CUDA;
        return Backend::CPU;
    }
    // SKI on Mac: vDSP FFT (CPU) is the fast path; Metal doesn't currently host
    // the per-axis Toeplitz FFT, so routing to Metal would silently fall back
    // through the CPU dense path which is much slower.
    if (solver == Solver::SKI) return Backend::CPU;
    if (!has_metal) return Backend::CPU;
    if (solver == Solver::CG && n > 2000) return Backend::Metal;
    if (d >= 16 && n >= 2000) return Backend::Metal;
    return Backend::CPU;
}

/// Concretize a (possibly-Auto) backend choice given the problem shape.
inline Backend concrete_backend(Backend b, std::size_t n, std::size_t d, Solver solver) {
    if (b != Backend::Auto) return b;
    return resolve_auto_backend(n, d, solver);
}

}  // namespace lightgp
