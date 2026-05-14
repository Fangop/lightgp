#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lightgp {

// 2D row-major float32 matrix; the workhorse for all CPU/GPU data exchange.
class Tensor {
public:
    /// Construct an empty (0x0) tensor.
    Tensor();
    /// Construct an uninitialized-zero rows x cols tensor.
    Tensor(std::size_t rows, std::size_t cols);
    /// Construct from explicit row-major data; data.size() must equal rows*cols.
    Tensor(std::size_t rows, std::size_t cols, std::vector<float> data);

    /// All-zeros tensor of shape rows x cols.
    static Tensor zeros(std::size_t rows, std::size_t cols);
    /// All-ones tensor of shape rows x cols.
    static Tensor ones(std::size_t rows, std::size_t cols);
    /// Square identity matrix of size n.
    static Tensor eye(std::size_t n);
    /// Tensor of standard-normal samples; deterministic given seed.
    static Tensor randn(std::size_t rows, std::size_t cols, std::uint64_t seed = 0);

    /// Number of rows.
    std::size_t rows() const { return rows_; }
    /// Number of columns.
    std::size_t cols() const { return cols_; }
    /// Total element count (rows*cols).
    std::size_t size() const { return rows_ * cols_; }

    /// Raw row-major data pointer.
    float* data() { return data_.data(); }
    /// Raw row-major data pointer (const).
    const float* data() const { return data_.data(); }

    /// Mutable element access at (i, j).
    float& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    /// Read-only element access at (i, j).
    float operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    /// Return a new transposed tensor.
    Tensor transpose() const;
    /// Matrix multiply: returns this * other.
    Tensor matmul(const Tensor& other) const;
    /// Elementwise this + other.
    Tensor add(const Tensor& other) const;
    /// Elementwise this - other.
    Tensor sub(const Tensor& other) const;
    /// Elementwise this * s.
    Tensor scalar_mul(float s) const;

    /// Add a scalar to the diagonal in-place (used for GP numerical stability).
    void add_jitter(float jitter);

    /// Print a short summary to stdout; clipped for large tensors.
    void print(const std::string& name = "") const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;
};

}  // namespace lightgp
