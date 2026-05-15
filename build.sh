#!/usr/bin/env bash
# Build the CPU library, tests, and the basic_regression example.
# Metal sources are included automatically on macOS unless LIGHTGP_NO_METAL is set.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build

UNAME=$(uname -s)
ENABLE_METAL=0
ENABLE_ACCELERATE=0
ENABLE_OPENBLAS=0
ENABLE_CUDA=0
if [[ "${UNAME}" == "Darwin" ]]; then
    [[ -z "${LIGHTGP_NO_METAL:-}" ]] && ENABLE_METAL=1
    [[ -z "${LIGHTGP_NO_ACCELERATE:-}" ]] && ENABLE_ACCELERATE=1
elif [[ "${UNAME}" == "Linux" ]]; then
    # Auto-detect OpenBLAS / LAPACK on Linux (header + runtime libs in standard locations).
    if [[ -z "${LIGHTGP_NO_OPENBLAS:-}" ]] && \
       [[ -e /usr/include/x86_64-linux-gnu/cblas.h ]] && \
       ldconfig -p | grep -q 'libopenblas\.so'; then
        ENABLE_OPENBLAS=1
    fi
    [[ -n "${LIGHTGP_ENABLE_CUDA:-}" ]] && ENABLE_CUDA=1
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
    inference/ski.cpp
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
    tests/test_cuda.cpp
    tests/test_ski.cpp
)

COMMON_FLAGS=(-std=c++17 -O2 -fPIC -Wall -Wextra -Wpedantic -I.)
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
if [[ "${ENABLE_OPENBLAS}" -eq 1 ]]; then
    # cblas.h is provided by the system netlib package; the Fortran spotrf_ symbol is
    # declared inline in blas_accel.cpp so we don't depend on the optional lapacke headers.
    # -lblas / -llapack go through Debian's libblas / liblapack alternatives, which on
    # this box point at OpenBLAS-pthread at runtime.
    COMMON_FLAGS+=(-DLIGHTGP_HAS_OPENBLAS -I/usr/include/x86_64-linux-gnu)
    LINK_FLAGS+=(-lblas -llapack)
fi
if [[ "${ENABLE_CUDA}" -eq 1 ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "LIGHTGP_ENABLE_CUDA=1 but nvcc not found in PATH" >&2
        exit 1
    fi
    # nvcc location → toolkit root → lib64 for runtime libs.
    NVCC_BIN=$(command -v nvcc)
    CUDA_ROOT=$(dirname "$(dirname "${NVCC_BIN}")")
    CUDA_LIBDIR="${CUDA_ROOT}/lib64"
    CUDA_INCDIR="${CUDA_ROOT}/include"
    COMMON_FLAGS+=(-DLIGHTGP_HAS_CUDA -I"${CUDA_INCDIR}")
    LINK_FLAGS+=(-L"${CUDA_LIBDIR}" -lcudart -lcublas -lcusolver)
    # Match the host compiler the CUDA SDK was tested against on this box.
    NVCC_HOST_CXX=${NVCC_HOST_CXX:-g++}
    # ccbin used for host-side parts of .cu translation; pick GCC on Linux because
    # CUDA 12.0's nvcc does not officially support clang-16 as the host compiler.
    NVCC_FLAGS=(-O2 -std=c++17 --compiler-options -fPIC,-Wall
                --compiler-bindir "${NVCC_HOST_CXX}"
                -DLIGHTGP_HAS_CUDA
                -I.)
    if [[ "${ENABLE_OPENBLAS}" -eq 1 ]]; then
        NVCC_FLAGS+=(-DLIGHTGP_HAS_OPENBLAS)
    fi
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

if [[ "${ENABLE_CUDA}" -eq 1 ]]; then
    for src in kernels/cuda/cuda_context.cu kernels/cuda/gemm_cuda.cu kernels/cuda/rbf_cuda.cu kernels/cuda/matern_cuda.cu kernels/cuda/rbf_matvec_cuda.cu solvers/cuda/cholesky_cuda.cu solvers/cuda/cholesky_solve_cuda.cu inference/ski_cuda.cu; do
        out="build/$(echo "$src" | tr '/' '_').o"
        nvcc "${NVCC_FLAGS[@]}" -c "$src" -o "$out"
        OBJS+=("$out")
    done
    # ski_cuda.cu pulls in cuFFT; thread it through link flags.
    LINK_FLAGS+=(-lcufft)
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

# CUDA-specific bench: only built when CUDA is enabled (otherwise its main() exits 1).
if [[ "${ENABLE_CUDA}" -eq 1 ]]; then
    clang++ "${COMMON_FLAGS[@]}" "${OBJS[@]}" "benchmarks/bench_cuda.cpp" ${LINK_FLAGS[@]+"${LINK_FLAGS[@]}"} -o "build/bench_cuda"
    echo "Built build/bench_cuda"
fi

# SKI bench builds on both CPU-only and CUDA configurations.
clang++ "${COMMON_FLAGS[@]}" "${OBJS[@]}" "benchmarks/bench_ski.cpp" ${LINK_FLAGS[@]+"${LINK_FLAGS[@]}"} -o "build/bench_ski"
echo "Built build/bench_ski"
