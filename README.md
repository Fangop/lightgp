# LightGP

**Lightweight Gaussian Process inference in C++ with Python bindings.**
Apple Metal + Accelerate (AMX) on macOS; NVIDIA CUDA + OpenBLAS on Linux. NumPy-first Python API with no deep-learning framework dependency.

[![CI](https://github.com/Fangop/lightgp/actions/workflows/ci.yml/badge.svg)](https://github.com/Fangop/lightgp/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-fangop.github.io%2Flightgp-2563EB)](https://fangop.github.io/lightgp/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.17898-b31b1b.svg)](https://arxiv.org/abs/2605.17898)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lightgp.svg)](https://pypi.org/project/lightgp/)

---

## Install

```bash
pip install lightgp
```

Prebuilt wheels are published for macOS-arm64 (with Metal + Accelerate)
and manylinux2014 x86_64 (with OpenBLAS). The Linux wheels are CPU-only;
for the CUDA backend, build from source with ``LIGHTGP_ENABLE_CUDA=1``.

From source (any platform — requires a C++17 compiler):

```bash
git clone https://github.com/Fangop/lightgp.git
cd lightgp/python
pip install -e ".[test]"
```

The source build uses ``scikit-build-core``. On macOS-arm64 it
auto-detects Apple Accelerate and Metal; on Linux it auto-detects
OpenBLAS / LAPACK and, when ``LIGHTGP_ENABLE_CUDA=1`` is set, CUDA.

## Quick start

```python
import numpy as np
import lightgp as gp

X = np.linspace(-3, 3, 100, dtype=np.float32).reshape(-1, 1)
y = np.sin(X[:, 0]).astype(np.float32) + 0.1 * np.random.randn(100).astype(np.float32)

model = gp.GPExact(gp.RBF())
model.fit(X, y)
model.optimize(steps=50)
pred = model.predict(X)             # → {'mean': (100,) float32, 'var': (100,) float32}
```

Kernel composition with Python operators:

```python
kernel = gp.Scale(gp.RBF()) + gp.Scale(gp.Periodic(period=1.0))
model = gp.GPExact(kernel, mean=gp.LinearMean(input_dim=1), noise_var=0.01)
model.fit(X, y)
model.optimize(steps=200)
```

Sparse GP for large datasets:

```python
model = gp.GPSparse(noise_var=0.1)
model.fit(X_big, y_big, num_inducing=200)  # scales to N=50000 in ~100 ms
```

## Documentation

Full docs at **https://fangop.github.io/lightgp/** — getting started, six tutorials,
complete API reference, benchmarks gallery, theory pages, and a developer guide.

## Features

- **Four inference paths** — exact Cholesky, matrix-free conjugate gradients,
  sparse Titsias VFE, and SKI/KISS-GP with FFT.
- **Composable kernels** — RBF, Matérn-{½, 3/2, 5/2}, Periodic, Linear, plus
  `+`, `*`, and `Scale` operators that build kernel trees with jointly
  optimisable hyperparameters.
- **Mean functions** — Zero, Constant, Linear.
- **Apple Metal backend** — native Metal Shading Language compute shaders,
  including a fused matrix-free $K\mathbf v$ kernel that keeps CG memory at O(N).
- **NVIDIA CUDA backend** — cuBLAS sgemm, cuSOLVER spotrf, cuFFT-driven SKI, and
  custom kernels for kernel-matrix construction and matrix-free matvecs.
- **Tuned CPU paths** — Apple Accelerate / AMX on macOS, OpenBLAS / LAPACK on
  Linux, auto-detected by the build script.
- **`Backend::Auto`** picks CPU vs Metal vs CUDA based on N, D, and the requested
  solver — users don't have to think about hardware crossover points.
- **Pure-C++17 core** — embeddable in iOS apps, robotics stacks, simulators, and
  game engines without bringing in a deep-learning framework.
- **Python bindings via pybind11** — `scikit-build-core` builds the right
  backend per platform from source (Metal on macOS-arm64, CUDA on Linux
  when `LIGHTGP_ENABLE_CUDA=1`) and exposes the full API with NumPy interop.

## Benchmarks

End-to-end GP fit + predict against GPyTorch on identical hardware
(fp32, D=4, median of 5 runs).

### Apple M4 (10 CPU cores, 8 GPU cores, 16 GB unified memory)

| Config | LightGP CPU | LightGP Metal | GPyTorch CPU | GPyTorch MPS | LightGP best vs GPyTorch best |
|---|--:|--:|--:|--:|--:|
| Exact RBF, N=2048 | **23.6 ms** | 195 ms | 89 ms | (gap*) | **3.8× faster** |
| Exact Matérn-5/2, N=2048 | **42 ms** | 191 ms | 106 ms | (gap*) | **2.5× faster** |
| Sparse RBF, N=10k, M=200 | **18.5 ms** | 42 ms | 42 ms | 69 ms | **2.3× faster** |
| Sparse RBF, N=50k, M=200 | **97.4 ms** | 156 ms | 196 ms | 98 ms | **2.0× faster vs CPU**; on par with MPS |
| Matrix-free $K\mathbf v$, N=20k | n/a | **22 ms** | n/a | (no equiv) | **32× over explicit** |

*GPyTorch MPS falls back to CPU for exact-GP variance because
`aten::_linalg_eigh.eigenvalues` is not yet implemented on PyTorch's MPS
backend — the gap is in PyTorch itself, not in GPyTorch.

### NVIDIA RTX 3060 (12 GB VRAM, CUDA 12.0)

| Config | LightGP CUDA | GPyTorch CUDA | LightGP advantage |
|---|--:|--:|--:|
| Exact RBF, N=512 | **2.0 ms** | 10.3 ms | **5.2×** |
| Exact RBF, N=1024 | **5.3 ms** | 35.4 ms | **6.7×** |
| Exact RBF, N=2048 | **28.0 ms** | 63.0 ms | **2.3×** |
| Exact RBF, N=4096 | 152 ms | **111 ms** | 0.7× (GPyTorch wins) |
| Sparse RBF, N=1000, M=100 (warm) | **0.9 ms** | 13.6 ms | **15.3×** |
| Sparse RBF, N=10k, M=200 (warm) | **13.7 ms** | 23.9 ms | **1.7×** |
| Sparse RBF, N=50k, M=200 (warm) | 75 ms | **55 ms** | 0.7× (GPyTorch wins) |
| Matrix-free $K\mathbf v$, N=20k | **9.8 ms** | (no equiv) | unique to LightGP |
| Matrix-free $K\mathbf v$, N=100k | **204 ms** | (no equiv) | unique to LightGP |
| Cholesky, N=4096 (component) | **37 ms** | n/a (not exposed) | 136× over OpenBLAS |

LightGP wins on 11 of 13 head-to-head Exact and Sparse configurations across
both platforms — the gap comes from a direct C++ → BLAS call path versus the
Python interpreter + PyTorch dispatcher + ATen operator registry that
GPyTorch traverses on every kernel call. Both libraries hit the same
underlying BLAS underneath. GPyTorch keeps the edge at large exact-GP sizes
(N=4096) and large sparse VFE (N=50k) where its persistent device tensors
and compiled autograd amortise the per-call overhead.

The matrix-free $K\mathbf v$ kernel is unique to LightGP on both Apple
Silicon and NVIDIA: PyTorch doesn't yet expose user-defined Metal compute
shaders, and the CUDA fusion would require building a custom op outside
GPyTorch. It enables CG-based GP inference at N=100k+ with O(N) memory
instead of O(N²) for the explicit kernel matrix.

The SKI / KISS-GP path with FFT runs a 500 000-point GP fit + predict in
**under 1 second on the RTX 3060** (and uses Accelerate vDSP for the
equivalent path on Mac). Full numbers, including SKI, GEMM, Cholesky, and
GPyTorch comparisons across more sizes, are in the
[benchmarks gallery](https://fangop.github.io/lightgp/benchmarks/) and the
[accompanying paper](https://arxiv.org/abs/2605.17898).

## C++ usage (embedding without Python)

lightgp is a dependency-free C++17 library — embed in iOS apps, robotics stacks, game engines.

```cpp
#include "lightgp/inference/gp_exact.h"
#include "lightgp/kernels/composite_kernel.h"
#include "lightgp/kernels/rbf_kernel.h"
#include "lightgp/kernels/periodic_kernel.h"
#include "lightgp/core/mean.h"

using namespace lightgp;

auto kernel = scale(std::make_shared<RBFKernel>())
            + scale(std::make_shared<PeriodicKernel>(/*l=*/1.0f, /*period=*/1.0f));
auto mean   = std::make_shared<LinearMean>(/*input_dim=*/1);

GPExact gp(kernel, mean, /*noise_variance=*/0.01f, Backend::Auto);
gp.fit(X_train, y_train);              // X_train, y_train: row-major float32 Tensors
gp.optimize_hyperparameters(/*steps=*/200);

Tensor mean_out, var_out;
gp.predict(X_test, mean_out, var_out);
```

For very large N, switch to matrix-free CG (the N×N kernel is never materialized):

```cpp
GPExact gp_cg(kernel, mean, 0.01f, Backend::Metal, Solver::CG);
gp_cg.fit(X_huge, y_huge);
```

For huge datasets, sparse VFE:

```cpp
GPSparseHyperparams hp;
GPSparse gp_sp(hp);
gp_sp.fit(X_huge, y_huge, /*num_inducing=*/200);   // O(NM² + M³)
```

## Building from source

### macOS (M-series — Metal + Accelerate auto-detected)

```bash
./build.sh
./build/run_tests                       # 853 test cases across the C++ suite
./build/basic_regression
./build/mauna_loa                       # kernel composition demo
./build/bench_paper                     # full benchmark suite, JSON-per-line stdout
```

### Linux (CPU + optional CUDA)

```bash
# CPU only (OpenBLAS / LAPACK auto-detected if installed)
./build.sh

# With CUDA (requires nvcc + CUDA Toolkit)
LIGHTGP_ENABLE_CUDA=1 ./build.sh

./build/run_tests
```

Install OpenBLAS / LAPACK first to get the fast CPU path:

```bash
sudo apt install libopenblas-dev liblapack-dev   # Debian / Ubuntu
```

The CUDA backend wires through ``Backend::CUDA`` and covers cuBLAS GEMM,
cuSOLVER Cholesky, cuFFT (used by ``Solver::SKI``), and custom CUDA kernels
for the RBF / Matérn matrix construction and matrix-free :math:`K\mathbf v`
matvec. ``Backend::Auto`` picks CUDA automatically when the build was
configured with ``LIGHTGP_ENABLE_CUDA=1`` and an NVIDIA device is present.

### Opt-out flags

```bash
LIGHTGP_NO_METAL=1 ./build.sh             # disable Metal even on Darwin
LIGHTGP_NO_ACCELERATE=1 ./build.sh        # use reference C++ instead of Apple BLAS
```

### Python bindings (development build, no CMake required)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pybind11 numpy pytest
./python/build_python.sh                 # produces python/lightgp/_core.<ext>.so
PYTHONPATH=python pytest python/tests -v
```

## Project layout

```
lightgp/
├── core/                Tensor, dispatch, backend / solver enums, Accelerate wrappers
├── kernels/             Kernel hierarchy (RBF, Matérn, Periodic, Linear, Sum/Product/Scale)
│   ├── cpu/             reference CPU + Accelerate paths
│   └── metal/           Metal Shading Language compute shaders
├── solvers/             Cholesky, conjugate gradients, Lanczos log-det
│   ├── cpu/
│   └── metal/
├── inference/           GPExact, GPSparse
├── data/                Bundled benchmark datasets (motorcycle, Mauna Loa, kin40k stand-ins)
├── tests/               853 C++ test cases
├── benchmarks/          10 standalone benches + Python GPyTorch comparison
├── examples/            basic_regression, mauna_loa (kernel composition)
└── python/              pybind11 bindings + pytest suite
```

## Citation

If you use LightGP in your research, please cite:

```bibtex
@misc{fang2026lightgp,
  title         = {LightGP: Lightweight Gaussian Process Inference in C++ on Metal and CUDA},
  author        = {Yu-Hsueh Fang},
  year          = {2026},
  eprint        = {2605.17898},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.17898},
  url           = {https://arxiv.org/abs/2605.17898}
}
```

📄 [Read the paper](https://arxiv.org/abs/2605.17898)

## License

MIT License. Copyright (c) 2026 Yu-Hsueh Fang. See [LICENSE](LICENSE) for the full text.
