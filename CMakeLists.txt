cmake_minimum_required(VERSION 3.20)
project(lightgp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIGHTGP_BUILD_TESTS "Build unit tests" ON)
option(LIGHTGP_BUILD_EXAMPLES "Build examples" ON)
option(LIGHTGP_ENABLE_METAL "Enable Metal backend" OFF)
option(LIGHTGP_ENABLE_CUDA "Enable CUDA backend" OFF)

set(LIGHTGP_CORE_SOURCES
    core/tensor.cpp
    core/dispatch.cpp
    kernels/cpu/rbf_cpu.cpp
    solvers/cpu/cholesky_cpu.cpp
    solvers/cpu/cg_cpu.cpp
    solvers/cpu/slq_cpu.cpp
    inference/gp_exact.cpp
)

if(LIGHTGP_ENABLE_METAL)
    list(APPEND LIGHTGP_CORE_SOURCES
        kernels/metal/metal_context.mm
        kernels/metal/rbf_metal.mm
        kernels/metal/gemm_metal.mm
        kernels/metal/rbf_matvec.mm
        solvers/metal/cholesky_metal.mm
        solvers/metal/cholesky_solve_metal.mm
    )
    enable_language(OBJCXX)
endif()

add_library(lightgp STATIC ${LIGHTGP_CORE_SOURCES})
target_include_directories(lightgp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(lightgp PRIVATE -Wall -Wextra -Wpedantic)

if(LIGHTGP_ENABLE_METAL)
    target_compile_definitions(lightgp PUBLIC LIGHTGP_HAS_METAL)
    set_source_files_properties(
        kernels/metal/metal_context.mm
        kernels/metal/rbf_metal.mm
        kernels/metal/gemm_metal.mm
        kernels/metal/rbf_matvec.mm
        solvers/metal/cholesky_metal.mm
        solvers/metal/cholesky_solve_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    target_link_libraries(lightgp PUBLIC "-framework Metal" "-framework Foundation")
endif()

if(LIGHTGP_ENABLE_CUDA)
    target_compile_definitions(lightgp PUBLIC LIGHTGP_HAS_CUDA)
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
    foreach(bench bench_rbf bench_cholesky bench_gp_e2e bench_matvec)
        add_executable(${bench} benchmarks/${bench}.cpp)
        target_link_libraries(${bench} PRIVATE lightgp)
    endforeach()
    add_custom_target(benchmarks DEPENDS bench_rbf bench_cholesky bench_gp_e2e bench_matvec)
endif()
