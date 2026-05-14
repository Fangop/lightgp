#include "test_utils.h"

#include <cmath>

#include "../core/tensor.h"
#include "../solvers/cpu/cholesky_cpu.h"
#include "../solvers/cpu/slq_cpu.h"

namespace lightgp {

void run_slq_cpu_tests() {
    std::printf("[slq_cpu] starting...\n");

    // Diagonal matrix: log|A| = sum(log(d_i)) is exact.
    {
        const std::size_t n = 100;
        Tensor A = Tensor::zeros(n, n);
        float true_log_det = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            const float d = 1.0f + 0.05f * static_cast<float>(i);
            A(i, i) = d;
            true_log_det += std::log(d);
        }
        const float est = slq_log_det_cpu(A, /*n_probes=*/40, /*n_iters=*/20, /*seed=*/42);
        // Identity-shifted diagonals make Lanczos trivial — should be very tight.
        LIGHTGP_CHECK(std::fabs(est - true_log_det) / std::fabs(true_log_det) < 0.05f);
    }

    // Random SPD: compare against exact log-det from Cholesky.
    {
        const std::size_t n = 80;
        Tensor M = Tensor::randn(n, n, 11);
        Tensor A = M.matmul(M.transpose());
        A.add_jitter(1.0f);  // shift well above 0 to stabilize Lanczos

        Tensor L;
        LIGHTGP_CHECK(cholesky_cpu(A, L));
        const float true_log_det = log_det_from_cholesky(L);

        const float est = slq_log_det_cpu(A, /*n_probes=*/30, /*n_iters=*/30, /*seed=*/7);
        // 30 probes + 30 Lanczos iters: aim for ~10% relative accuracy on N=80.
        LIGHTGP_CHECK(std::fabs(est - true_log_det) / std::fabs(true_log_det) < 0.10f);
    }

    // Scaled identity: log|cI| = N * log(c). Exact regardless of probe count.
    {
        const std::size_t n = 50;
        const float c = 2.0f;
        Tensor A = Tensor::zeros(n, n);
        for (std::size_t i = 0; i < n; ++i) A(i, i) = c;
        const float true_log_det = static_cast<float>(n) * std::log(c);
        const float est = slq_log_det_cpu(A, /*n_probes=*/5, /*n_iters=*/5, /*seed=*/13);
        LIGHTGP_CHECK_NEAR(est, true_log_det, 1e-3f);
    }
}

}  // namespace lightgp
