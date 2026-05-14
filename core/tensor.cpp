#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#ifdef LIGHTGP_HAS_ACCELERATE
#include "blas_accel.h"
#endif

namespace lightgp {

Tensor::Tensor() = default;

Tensor::Tensor(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Tensor::Tensor(std::size_t rows, std::size_t cols, std::vector<float> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {
    assert(data_.size() == rows * cols);
}

Tensor Tensor::zeros(std::size_t rows, std::size_t cols) {
    return Tensor(rows, cols);
}

Tensor Tensor::ones(std::size_t rows, std::size_t cols) {
    Tensor t(rows, cols);
    std::fill(t.data_.begin(), t.data_.end(), 1.0f);
    return t;
}

Tensor Tensor::eye(std::size_t n) {
    Tensor t(n, n);
    for (std::size_t i = 0; i < n; ++i) t(i, i) = 1.0f;
    return t;
}

Tensor Tensor::randn(std::size_t rows, std::size_t cols, std::uint64_t seed) {
    Tensor t(rows, cols);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& v : t.data_) v = dist(rng);
    return t;
}

Tensor Tensor::transpose() const {
    Tensor r(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            r(j, i) = (*this)(i, j);
        }
    }
    return r;
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(cols_ == other.rows_);
    Tensor r(rows_, other.cols_);
#ifdef LIGHTGP_HAS_ACCELERATE
    // Apple AMX-accelerated sgemm. Beats hand-rolled triple loop by 10-50x at moderate N.
    gemm_accelerate(*this, other, r);
#else
    // i-k-j ordering: B's row and result row swept contiguously.
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const float a = (*this)(i, k);
            for (std::size_t j = 0; j < other.cols_; ++j) {
                r(i, j) += a * other(k, j);
            }
        }
    }
#endif
    return r;
}

Tensor Tensor::add(const Tensor& other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    Tensor r(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + other.data_[i];
    return r;
}

Tensor Tensor::sub(const Tensor& other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    Tensor r(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - other.data_[i];
    return r;
}

Tensor Tensor::scalar_mul(float s) const {
    Tensor r(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * s;
    return r;
}

void Tensor::add_jitter(float jitter) {
    const std::size_t n = std::min(rows_, cols_);
    for (std::size_t i = 0; i < n; ++i) (*this)(i, i) += jitter;
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) std::printf("%s = (%zux%zu)\n", name.c_str(), rows_, cols_);
    const std::size_t max_r = std::min<std::size_t>(rows_, 8);
    const std::size_t max_c = std::min<std::size_t>(cols_, 8);
    for (std::size_t i = 0; i < max_r; ++i) {
        std::printf("  [");
        for (std::size_t j = 0; j < max_c; ++j) {
            std::printf(j == 0 ? "%9.4f" : ", %9.4f", (*this)(i, j));
        }
        if (cols_ > max_c) std::printf(", ...");
        std::printf("]\n");
    }
    if (rows_ > max_r) std::printf("  ...\n");
}

}  // namespace lightgp
