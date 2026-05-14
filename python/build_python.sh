#!/usr/bin/env bash
# Build the pybind11 module directly with clang++ (skips scikit-build-core /
# CMake which would require pip install in an isolated build environment).
# Produces python/lightgp/_core.<extension>.so which __init__.py imports.
set -euo pipefail
cd "$(dirname "$0")/.."

# Make sure the C++ library is built (so we can link in build/*.o object files).
./build.sh > /dev/null

PYTHON=${PYTHON:-.venv/bin/python3}
PYBIND11_INCLUDES=$($PYTHON -m pybind11 --includes)
EXT_SUFFIX=$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# Compile + link the binding module. We deliberately include all C++ object files
# from the regular library build (build/*.o) rather than re-compiling sources.
# Skip tests/, examples/ and bench_*.o (we only want library code).
LIB_OBJS=$(ls build/*.o | grep -v -E '(test_|examples_|benchmarks_)' || true)

clang++ -std=c++17 -O2 -fPIC -shared \
    -undefined dynamic_lookup \
    $PYBIND11_INCLUDES \
    -I. \
    -DLIGHTGP_HAS_METAL -DLIGHTGP_HAS_ACCELERATE \
    -framework Metal -framework Foundation -framework Accelerate \
    python/bindings.cpp $LIB_OBJS \
    -o "python/lightgp/_core${EXT_SUFFIX}"

echo "Built python/lightgp/_core${EXT_SUFFIX}"
