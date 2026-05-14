#pragma once

#include <cstdint>

#include "../core/tensor.h"

namespace lightgp {
namespace data {

/// Train/test split + name + standardization stats so metrics are in physical units.
struct Dataset {
    Tensor X_train, y_train;
    Tensor X_test, y_test;
    float y_mean = 0.0f;
    float y_std = 1.0f;
    const char* name = "";
};

/// Motorcycle-like heteroscedastic 1D regression (N=133).
/// Stand-in for Silverman's mcycle: zero noise pre-impact, rising noise after,
/// negative peak around t=25, oscillation thereafter.
Dataset make_motorcycle(std::uint64_t seed = 1);

/// Mauna-Loa-like 1D regression (N=780, monthly samples 1958-2022).
/// Stand-in for the NOAA CO2 record: linear+quadratic trend + annual seasonal +
/// small noise. Captures the structure GP papers use to demonstrate kernel composition.
Dataset make_mauna_loa(std::uint64_t seed = 2);

/// kin40k-like 8D nonlinear regression (N=40000).
/// Stand-in for the GPML benchmark: smooth nonlinear function of 8 inputs.
Dataset make_kin40k(std::uint64_t seed = 3);

}  // namespace data
}  // namespace lightgp
