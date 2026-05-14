#!/usr/bin/env bash
# Build the CPU library, tests, and the basic_regression example.
# Metal sources are included automatically on macOS unless LIGHTGP_NO_METAL is set.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build

UNAME=$(uname -s)
ENABLE_METAL=0
ENABLE_ACCELERATE=0
if [[ "${UNAME}" == "Darwin" ]]; then
    [[ -z "${LIGHTGP_NO_METAL:-}" ]] && ENABLE_METAL=1
    [[ -z "${LIGHTGP_NO_ACCELERATE:-}" ]] && ENABLE_ACCELERATE=1
fi

CPP_SRCS=(
    core/tensor.cpp
    core/dispatch.cpp
    core/blas_accel.cpp
    core/mean.cpp
    kernels/kernel_base.cpp
    kernels/rbf_kernel.cpp
    kernels/matern_kernel.cpp
    kernels/periodic_kernel.cpp
    kernels/linear_kernel.cpp
    kernels/composite_kernel.cpp
    kernels/cpu/rbf_cpu.cpp
    kernels/cpu/matern_cpu.cpp
    solvers/cpu/cholesky_cpu.cpp
    solvers/cpu/cg_cpu.cpp
    solvers/cpu/slq_cpu.cpp
    inference/gp_exact.cpp
    inference/gp_sparse.cpp
    data/datasets.cpp
    tests/test_tensor.cpp
    tests/test_rbf_cpu.cpp
    tests/test_cholesky_cpu.cpp
    tests/test_gp_exact.cpp
    tests/test_rbf_metal.cpp
    tests/test_gp_backend.cpp
    tests/test_cg_cpu.cpp
    tests/test_cholesky_metal.cpp
    tests/test_cholesky_solve_metal.cpp
    tests/test_gp_sparse.cpp
    tests/test_matern.cpp
    tests/test_matern_metal.cpp
    tests/test_slq_cpu.cpp
    tests/test_gp_cg.cpp
    tests/test_rbf_matvec.cpp
    tests/test_kernel_objects.cpp
)

COMMON_FLAGS=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -I.)
LINK_FLAGS=()
METAL_FLAGS=()

if [[ "${ENABLE_METAL}" -eq 1 ]]; then
    COMMON_FLAGS+=(-DLIGHTGP_HAS_METAL)
    LINK_FLAGS+=(-framework Metal -framework Foundation)
    METAL_FLAGS+=(-fobjc-arc -x objective-c++)
fi
if [[ "${ENABLE_ACCELERATE}" -eq 1 ]]; then
    COMMON_FLAGS+=(-DLIGHTGP_HAS_ACCELERATE)
    LINK_FLAGS+=(-framework Accelerate)
fi

# Compile each .cpp to an object, plus .mm files when Metal is on.
OBJS=()
for src in "${CPP_SRCS[@]}"; do
    out="build/$(echo "$src" | tr '/' '_').o"
    clang++ "${COMMON_FLAGS[@]}" -c "$src" -o "$out"
    OBJS+=("$out")
done

if [[ "${ENABLE_METAL}" -eq 1 ]]; then
    for src in kernels/metal/metal_context.mm kernels/metal/rbf_metal.mm kernels/metal/matern_metal.mm kernels/metal/gemm_metal.mm kernels/metal/rbf_matvec.mm solvers/metal/cholesky_metal.mm solvers/metal/cholesky_solve_metal.mm; do
        out="build/$(echo "$src" | tr '/' '_').o"
        clang++ "${COMMON_FLAGS[@]}" "${METAL_FLAGS[@]}" -c "$src" -o "$out"
        OBJS+=("$out")
    done
fi

# Link the test runner.
clang++ "${COMMON_FLAGS[@]}" "${OBJS[@]}" tests/run_tests.cpp ${LINK_FLAGS[@]+"${LINK_FLAGS[@]}"} -o build/run_tests
echo "Built build/run_tests"

# Link the example.
clang++ "${COMMON_FLAGS[@]}" "${OBJS[@]}" examples/basic_regression.cpp ${LINK_FLAGS[@]+"${LINK_FLAGS[@]}"} -o build/basic_regression
echo "Built build/basic_regression"

# Mauna Loa kernel composition example.
clang++ "${COMMON_FLAGS[@]}" "${OBJS[@]}" examples/mauna_loa.cpp ${LINK_FLAGS[@]+"${LINK_FLAGS[@]}"} -o build/mauna_loa
echo "Built build/mauna_loa"

# Link the benchmarks.
for bench in bench_rbf bench_cholesky bench_gp_e2e bench_matvec bench_sparse bench_paper bench_accuracy bench_datasets; do
    clang++ "${COMMON_FLAGS[@]}" "${OBJS[@]}" "benchmarks/${bench}.cpp" ${LINK_FLAGS[@]+"${LINK_FLAGS[@]}"} -o "build/${bench}"
    echo "Built build/${bench}"
done
