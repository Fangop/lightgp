#include <cstdio>

#include "test_utils.h"

namespace lightgp {
void run_tensor_tests();
void run_rbf_cpu_tests();
void run_cholesky_cpu_tests();
void run_gp_exact_tests();
void run_rbf_metal_tests();
void run_gp_backend_tests();
void run_cg_cpu_tests();
void run_cholesky_metal_tests();
void run_cholesky_solve_metal_tests();
void run_slq_cpu_tests();
void run_gp_cg_tests();
void run_rbf_matvec_tests();
void run_gp_sparse_tests();
void run_matern_tests();
void run_matern_metal_tests();
void run_kernel_object_tests();
void run_cuda_tests();
void run_ski_tests();
}  // namespace lightgp

int main() {
    lightgp::run_tensor_tests();
    lightgp::run_rbf_cpu_tests();
    lightgp::run_cholesky_cpu_tests();
    lightgp::run_gp_exact_tests();
    lightgp::run_rbf_metal_tests();
    lightgp::run_gp_backend_tests();
    lightgp::run_cg_cpu_tests();
    lightgp::run_cholesky_metal_tests();
    lightgp::run_cholesky_solve_metal_tests();
    lightgp::run_slq_cpu_tests();
    lightgp::run_gp_cg_tests();
    lightgp::run_rbf_matvec_tests();
    lightgp::run_gp_sparse_tests();
    lightgp::run_matern_tests();
    lightgp::run_matern_metal_tests();
    lightgp::run_kernel_object_tests();
    lightgp::run_cuda_tests();
    lightgp::run_ski_tests();

    const int passed = lightgp::test::pass_count();
    const int failed = lightgp::test::fail_count();
    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
