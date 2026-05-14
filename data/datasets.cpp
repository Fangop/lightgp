#include "datasets.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace lightgp {
namespace data {

namespace {
constexpr float kPi = 3.14159265358979323846f;

// Helpers
struct Split {
    Tensor X_train, y_train, X_test, y_test;
    float y_mean = 0.0f, y_std = 1.0f;
};

// 80/20 random split, then standardize y.
Split split_and_standardize(const Tensor& X, const Tensor& y, std::uint64_t seed) {
    const std::size_t N = X.rows();
    std::vector<std::size_t> idx(N);
    for (std::size_t i = 0; i < N; ++i) idx[i] = i;
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    const std::size_t n_train = static_cast<std::size_t>(0.8 * N);
    const std::size_t n_test = N - n_train;
    const std::size_t D = X.cols();

    // Compute y_mean and y_std from training subset.
    float y_sum = 0.0f;
    for (std::size_t i = 0; i < n_train; ++i) y_sum += y(idx[i], 0);
    const float y_mean = y_sum / static_cast<float>(n_train);
    float y_var = 0.0f;
    for (std::size_t i = 0; i < n_train; ++i) {
        const float r = y(idx[i], 0) - y_mean;
        y_var += r * r;
    }
    const float y_std = std::sqrt(y_var / static_cast<float>(n_train)) + 1e-6f;

    Split s;
    s.y_mean = y_mean;
    s.y_std = y_std;
    s.X_train = Tensor(n_train, D);
    s.y_train = Tensor(n_train, 1);
    s.X_test = Tensor(n_test, D);
    s.y_test = Tensor(n_test, 1);
    for (std::size_t i = 0; i < n_train; ++i) {
        for (std::size_t d = 0; d < D; ++d) s.X_train(i, d) = X(idx[i], d);
        s.y_train(i, 0) = (y(idx[i], 0) - y_mean) / y_std;
    }
    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t d = 0; d < D; ++d) s.X_test(i, d) = X(idx[n_train + i], d);
        s.y_test(i, 0) = (y(idx[n_train + i], 0) - y_mean) / y_std;
    }
    return s;
}

}  // namespace

Dataset make_motorcycle(std::uint64_t seed) {
    // 133 points: t in [2.4, 57.6] ms, acceleration in g.
    constexpr int N = 133;
    Tensor X(N, 1), y(N, 1);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        const float t = 2.4f + (57.6f - 2.4f) * static_cast<float>(i) / static_cast<float>(N - 1);
        X(i, 0) = t;
        // Mean: simple decaying oscillation post-impact (impact at t=15).
        const float mean = (t < 15.0f) ? 0.0f
            : -120.0f * std::exp(-0.05f * (t - 15.0f)) * std::cos(0.5f * (t - 15.0f));
        // Heteroscedastic noise: zero-ish pre-impact, ramps up.
        const float sigma = (t < 15.0f) ? 1.5f : 8.0f + 0.5f * (t - 15.0f);
        y(i, 0) = mean + sigma * n(rng);
    }
    auto s = split_and_standardize(X, y, seed * 17 + 1);
    return {std::move(s.X_train), std::move(s.y_train),
            std::move(s.X_test),  std::move(s.y_test),
            s.y_mean, s.y_std, "motorcycle"};
}

Dataset make_mauna_loa(std::uint64_t seed) {
    // Monthly 1958-01 .. 2022-12 → 780 samples.
    constexpr int N = 780;
    Tensor X(N, 1), y(N, 1);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        const float years_since_1958 = static_cast<float>(i) / 12.0f;
        X(i, 0) = years_since_1958;
        // Real CO2 has linear + quadratic trend + annual seasonal + small autocorrelated noise.
        // We model: 315 + 1.5*t + 0.012*t^2 + 3*sin(2π*t) + 0.5*cos(2π*t * 2) + N(0, 0.3).
        const float trend = 315.0f + 1.5f * years_since_1958
                          + 0.012f * years_since_1958 * years_since_1958;
        const float season = 3.0f * std::sin(2.0f * kPi * years_since_1958)
                           + 0.5f * std::cos(4.0f * kPi * years_since_1958);
        y(i, 0) = trend + season + 0.3f * n(rng);
    }
    auto s = split_and_standardize(X, y, seed * 17 + 1);
    return {std::move(s.X_train), std::move(s.y_train),
            std::move(s.X_test),  std::move(s.y_test),
            s.y_mean, s.y_std, "mauna_loa"};
}

Dataset make_kin40k(std::uint64_t seed) {
    // 8D nonlinear regression, 40000 samples — too big for exact Cholesky in any
    // sane GP library; this is the sparse / CG showcase.
    constexpr int N = 40000;
    constexpr int D = 8;
    Tensor X(N, D), y(N, 1);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        float v[D];
        for (int d = 0; d < D; ++d) v[d] = u(rng);
        for (int d = 0; d < D; ++d) X(i, d) = v[d];
        // Friedman-style nonlinear interaction surface.
        const float f = 10.0f * std::sin(kPi * v[0] * v[1])
                      + 20.0f * (v[2] - 0.5f) * (v[2] - 0.5f)
                      + 10.0f * v[3]
                      + 5.0f  * v[4]
                      + 0.5f  * v[5] * v[6]
                      + 0.25f * v[7] * v[0];
        y(i, 0) = f + 1.0f * nrm(rng);
    }
    auto s = split_and_standardize(X, y, seed * 17 + 1);
    return {std::move(s.X_train), std::move(s.y_train),
            std::move(s.X_test),  std::move(s.y_test),
            s.y_mean, s.y_std, "kin40k"};
}

}  // namespace data
}  // namespace lightgp
