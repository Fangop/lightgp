#pragma once

namespace lightgp {

/// GP inference solver. Cholesky is exact O(N^3); CG is iterative O(N^2 k)
/// and supports matrix-free operation for large-N scalability.
enum class Solver { Cholesky, CG };

}  // namespace lightgp
