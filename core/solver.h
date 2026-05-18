// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

namespace lightgp {

/// GP inference solver.
///   - Cholesky: exact O(N^3) — best for N <= a few thousand.
///   - CG:       iterative O(N^2 k); supports matrix-free operation on Metal/CUDA.
///   - SKI:      structured Kernel Interpolation (KISS-GP). O(N + M log M) matvec
///               via cubic interpolation + Toeplitz/Kronecker FFT; CG-driven solve
///               on top. Approximation quality scales with grid density.
enum class Solver { Cholesky, CG, SKI };

}  // namespace lightgp
