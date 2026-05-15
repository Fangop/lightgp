#!/usr/bin/env bash
# Build the pybind11 module directly with clang++ (skips scikit-build-core /
# CMake which would require pip install in an isolated build environment).
# Produces python/lightgp/_core.<extension>.so which __init__.py imports.
#
# Auto-detects backend on the host: Accelerate + Metal on macOS, OpenBLAS (+
# optionally CUDA, via LIGHTGP_ENABLE_CUDA=1) on Linux.
set -euo pipefail
cd "$(dirname "$0")/.."

# Make sure the C++ library is built (so we can link in build/*.o object files).
./build.sh > /dev/null

PYTHON=${PYTHON:-.venv/bin/python3}
PYBIND11_INCLUDES=$($PYTHON -m pybind11 --includes)
EXT_SUFFIX=$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# Compile + link the binding module. We deliberately include all C++ object files
# from the regular library build (build/*.o) rather than re-compiling sources.
# Skip tests/, examples/ and bench_*.o (we only want library code). If CUDA wasn't
# requested for this Python build, also drop any leftover .cu.o objects so we don't
# pull in unresolved CUDA symbols.
if [[ -n "${LIGHTGP_ENABLE_CUDA:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    LIB_OBJS=$(ls build/*.o | grep -v -E '(test_|examples_|benchmarks_)' || true)
else
    LIB_OBJS=$(ls build/*.o | grep -v -E '(test_|examples_|benchmarks_|\.cu\.o$)' || true)
fi

UNAME=$(uname -s)
EXTRA_FLAGS=()
EXTRA_LINK=()
if [[ "${UNAME}" == "Darwin" ]]; then
    EXTRA_FLAGS+=(-undefined dynamic_lookup -DLIGHTGP_HAS_METAL -DLIGHTGP_HAS_ACCELERATE)
    EXTRA_LINK+=(-framework Metal -framework Foundation -framework Accelerate)
else
    # Linux: match build.sh's auto-detection so the bindings link against the same backends.
    if [[ -e /usr/include/x86_64-linux-gnu/cblas.h ]] && \
       ldconfig -p | grep -q 'libopenblas\.so'; then
        EXTRA_FLAGS+=(-DLIGHTGP_HAS_OPENBLAS -I/usr/include/x86_64-linux-gnu)
        EXTRA_LINK+=(-lblas -llapack)
    fi
    if [[ -n "${LIGHTGP_ENABLE_CUDA:-}" ]] && command -v nvcc >/dev/null 2>&1; then
        EXTRA_FLAGS+=(-DLIGHTGP_HAS_CUDA)
        CUDA_LIBDIR=$(dirname "$(dirname "$(command -v nvcc)")")/lib64
        EXTRA_LINK+=(-L"${CUDA_LIBDIR}" -lcudart -lcublas -lcusolver -lcufft)
    fi
fi

clang++ -std=c++17 -O2 -fPIC -shared \
    $PYBIND11_INCLUDES \
    -I. \
    "${EXTRA_FLAGS[@]}" \
    python/bindings.cpp $LIB_OBJS \
    "${EXTRA_LINK[@]}" \
    -o "python/lightgp/_core${EXT_SUFFIX}"

echo "Built python/lightgp/_core${EXT_SUFFIX}"
