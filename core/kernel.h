#pragma once

namespace lightgp {

/// Stationary kernel families supported by GPExact and GPSparse.
/// All variants share the same (length_scale, signal_variance) parameterization
/// and depend on inputs only through r = ||x_i - x_j||.
///
///   RBF:        k(r) = sf2 * exp(-0.5 r^2 / l^2)
///   Matern12:   k(r) = sf2 * exp(-r/l)
///   Matern32:   k(r) = sf2 * (1 + sqrt(3) r/l) * exp(-sqrt(3) r/l)
///   Matern52:   k(r) = sf2 * (1 + sqrt(5) r/l + 5 r^2 / (3 l^2)) * exp(-sqrt(5) r/l)
enum class KernelType { RBF, Matern12, Matern32, Matern52 };

}  // namespace lightgp
