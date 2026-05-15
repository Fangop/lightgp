cmake_minimum_required(VERSION 3.20)
project(lightgp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIGHTGP_BUILD_TESTS "Build unit tests" ON)
option(LIGHTGP_BUILD_EXAMPLES "Build examples" ON)
option(LIGHTGP_ENABLE_METAL "Enable Metal backend" OFF)
option(LIGHTGP_ENABLE_CUDA "Enable CUDA backend" OFF)
option(LIGHTGP_ENABLE_OPENBLAS "Enable OpenBLAS / LAPACK on Linux" OFF)

set(LIGHTGP_CORE_SOURCES
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
)

if(LIGHTGP_ENABLE_METAL)
    list(APPEND LIGHTGP_CORE_SOURCES
        kernels/metal/metal_context.mm
        kernels/metal/rbf_metal.mm
        kernels/metal/gemm_metal.mm
        kernels/metal/rbf_matvec.mm
        kernels/metal/matern_metal.mm
        solvers/metal/cholesky_metal.mm
        solvers/metal/cholesky_solve_metal.mm
    )
    enable_language(OBJCXX)
endif()

if(LIGHTGP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND LIGHTGP_CORE_SOURCES
        kernels/cuda/cuda_context.cu
        kernels/cuda/gemm_cuda.cu
        kernels/cuda/rbf_cuda.cu
        kernels/cuda/matern_cuda.cu
        kernels/cuda/rbf_matvec_cuda.cu
        solvers/cuda/cholesky_cuda.cu
        solvers/cuda/cholesky_solve_cuda.cu
    )
endif()

add_library(lightgp STATIC ${LIGHTGP_CORE_SOURCES})
target_include_directories(lightgp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(lightgp PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(lightgp PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LIGHTGP_ENABLE_METAL)
    target_compile_definitions(lightgp PUBLIC LIGHTGP_HAS_METAL)
    set_source_files_properties(
        kernels/metal/metal_context.mm
        kernels/metal/rbf_metal.mm
        kernels/metal/gemm_metal.mm
        kernels/metal/rbf_matvec.mm
        kernels/metal/matern_metal.mm
        solvers/metal/cholesky_metal.mm
        solvers/metal/cholesky_solve_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    target_link_libraries(lightgp PUBLIC "-framework Metal" "-framework Foundation")
endif()

if(LIGHTGP_ENABLE_CUDA)
    target_compile_definitions(lightgp PUBLIC LIGHTGP_HAS_CUDA)
    target_link_libraries(lightgp PUBLIC
        CUDA::cudart CUDA::cublas CUDA::cusolver)
    set_target_properties(lightgp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(APPLE AND NOT LIGHTGP_ENABLE_METAL)
    # Empty: Accelerate is opt-in via the bash builder; CMake build defaults to reference C++.
endif()

if((NOT APPLE) AND LIGHTGP_ENABLE_OPENBLAS)
    find_package(BLAS)
    find_package(LAPACK)
    if(BLAS_FOUND AND LAPACK_FOUND)
        target_compile_definitions(lightgp PUBLIC LIGHTGP_HAS_OPENBLAS)
        target_link_libraries(lightgp PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    else()
        message(WARNING "LIGHTGP_ENABLE_OPENBLAS=ON but BLAS/LAPACK not found; falling back to reference path")
    endif()
endif()

if(LIGHTGP_BUILD_TESTS)
    enable_testing()
    add_executable(run_tests
        tests/run_tests.cpp
        tests/test_tensor.cpp
        tests/test_rbf_cpu.cpp
        tests/test_cholesky_cpu.cpp
        tests/test_gp_exact.cpp
        tests/test_rbf_metal.cpp
        tests/test_gp_backend.cpp
        tests/test_cg_cpu.cpp
        tests/test_cholesky_metal.cpp
        tests/test_cholesky_solve_metal.cpp
        tests/test_slq_cpu.cpp
        tests/test_gp_cg.cpp
        tests/test_rbf_matvec.cpp
        tests/test_gp_sparse.cpp
        tests/test_matern.cpp
        tests/test_matern_metal.cpp
        tests/test_kernel_objects.cpp
        tests/test_cuda.cpp
    )
    target_link_libraries(run_tests PRIVATE lightgp)
    add_test(NAME run_tests COMMAND run_tests)
endif()

if(LIGHTGP_BUILD_EXAMPLES)
    add_executable(basic_regression examples/basic_regression.cpp)
    target_link_libraries(basic_regression PRIVATE lightgp)
endif()

option(LIGHTGP_BUILD_BENCHMARKS "Build benchmarks" ON)
if(LIGHTGP_BUILD_BENCHMARKS)
    foreach(bench bench_rbf bench_cholesky bench_gp_e2e bench_matvec bench_sparse bench_paper bench_accuracy bench_datasets)
        add_executable(${bench} benchmarks/${bench}.cpp)
        target_link_libraries(${bench} PRIVATE lightgp)
    endforeach()
    add_custom_target(benchmarks DEPENDS bench_rbf bench_cholesky bench_gp_e2e bench_matvec bench_sparse bench_paper bench_accuracy bench_datasets)
endif()
