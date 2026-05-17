# LightGP

**Lightweight Gaussian Process inference in C++ with Python bindings.**
Apple Metal + Accelerate (AMX) on macOS; CUDA + OpenBLAS on Linux. Pure-C++17 core, no deep-learning framework runtime — designed to complement [GPyTorch](https://gpytorch.ai/) for projects that need GP regression with a small dependency footprint or direct C++ embedding.

[![CI](https://github.com/Fangop/lightgp/actions/workflows/ci.yml/badge.svg)](https://github.com/Fangop/lightgp/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-fangop.github.io%2Flightgp-2563EB)](https://fangop.github.io/lightgp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lightgp.svg)](https://pypi.org/project/lightgp/)

---

## Install

```bash
pip install lightgp           # binary wheels for macOS-arm64 and manylinux-x86_64
```

From source (requires a C++17 compiler):

```bash
git clone https://github.com/Fangop/lightgp.git
cd lightgp/python
pip install -e ".[test]"
```

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

## Where LightGP fits

[GPyTorch](https://gpytorch.ai/) is the canonical Python GP library and offers a
much broader set of models — deep GPs, variational families beyond VFE, spectral
mixture kernels, multi-output likelihoods, and a mature autograd-driven hyperparameter
pipeline built on PyTorch. If you already live in the PyTorch ecosystem, GPyTorch is
almost certainly the right tool.

LightGP targets a narrower slice: GP regression with a minimal binary footprint,
shipping as a single static C++ library and a numpy-only Python wheel. It implements
the four most common inference paths (exact Cholesky, matrix-free CG, sparse VFE,
SKI/KISS-GP) and is designed to be embedded in C++ applications — robotics stacks,
mobile apps, simulators, game engines — where pulling in a deep-learning framework
isn't an option.

The two libraries call the same underlying BLAS on each platform (Apple Accelerate
on macOS, OpenBLAS / cuBLAS on Linux), so the benchmark numbers below mostly reflect
dispatch-layer overhead rather than fundamental algorithmic differences.

## Features

| Aspect | LightGP | GPyTorch |
|---|---|---|
| Inference paths | Exact / CG / Sparse VFE / SKI | Exact / CG / Sparse + variational families / SKI / deep GPs |
| Kernel families | RBF, Matérn-{½,3/2,5/2}, Periodic, Linear (composable) | The above + spectral mixture, polynomial, RFF, and more |
| Hyperparameter optimization | Finite-diff Adam; analytical grads for legacy single-kernel API | Full PyTorch autograd |
| Mean functions | Zero, Constant, Linear | All of the above + custom modules |
| Runtime dependencies | numpy (Python); none (C++) | PyTorch |
| CPU backend | Apple Accelerate / AMX (macOS), OpenBLAS / LAPACK (Linux) | Same Accelerate / MKL / OpenBLAS via PyTorch |
| Apple Metal | Native compute shaders + matrix-free Kv | Partial (PyTorch MPS) |
| NVIDIA CUDA | cuBLAS / cuSOLVER / cuFFT + matrix-free Kv | Yes |
| Numerical precision | float32 | float32 / float64 |
| Likelihoods | Gaussian | Gaussian + non-Gaussian (variational) |
| Embeddable in pure C++ projects | Yes | No |
| `pip install` | Yes | Yes |

## Benchmarks

End-to-end fit + predict on Apple M4 (fp32, median of 3 runs):

| Config | LightGP CPU | LightGP Metal | GPyTorch CPU | GPyTorch MPS |
|---|--:|--:|--:|--:|
| Exact RBF, N=2048, D=4 | 44 ms | 195 ms | 89 ms | (gap*) |
| Exact Matérn-5/2, N=2048, D=4 | 42 ms | 191 ms | 106 ms | (gap*) |
| Sparse RBF, N=10000, M=200 | 25 ms | 42 ms | 42 ms | 69 ms |
| Sparse RBF, N=50000, M=200 | 114 ms | 156 ms | 196 ms | 98 ms |
| Matrix-free $K\mathbf v$, N=20000 | n/a | 22 ms | n/a | (no equiv) |

*GPyTorch MPS falls back to CPU for exact-GP variance because
`aten::_linalg_eigh.eigenvalues` is not yet available on MPS — this is a PyTorch
backend gap, not a GPyTorch design choice.

Both libraries reach the same Accelerate / AMX sgemm and spotrf underneath. The
runtime difference at modest N (exact GP under N≈4 k) comes from dispatch-layer
overhead: LightGP calls BLAS directly from C++ while GPyTorch traverses Python +
PyTorch's dispatcher. At larger sparse-GP sizes GPyTorch's MPS pipeline can
overtake LightGP CPU — see the [linked benchmarks page](https://fangop.github.io/lightgp/benchmarks/)
for the full table including N≥50 k and the Linux RTX 3060 numbers.

The matrix-free $K\mathbf v$ kernel — used by `Solver::CG` to keep memory at O(N)
rather than O(N²) — is implemented as a custom Metal compute shader. GPyTorch's
MPS backend can't currently express this fusion because PyTorch doesn't yet
expose user-defined Metal kernels.

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
./build/run_tests                       # 329,170 assertions across 15 test groups
./build/basic_regression
./build/mauna_loa                       # kernel composition demo
./build/bench_paper                     # full benchmark suite, JSON-per-line stdout
```

### Linux (CPU + optional CUDA)

```bash
# CPU only (OpenBLAS / LAPACKE auto-detected if installed)
LIGHTGP_NO_METAL=1 LIGHTGP_NO_ACCELERATE=1 ./build.sh

# With CUDA (requires nvcc + CUDA Toolkit)
LIGHTGP_ENABLE_CUDA=1 ./build.sh

./build/run_tests
```

Install OpenBLAS / LAPACK first to get the fast CPU path:

```bash
sudo apt install libopenblas-dev liblapacke-dev   # Debian / Ubuntu
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
├── tests/               329,170 C++ assertions
├── benchmarks/          8 standalone benches + Python GPyTorch comparison
├── examples/            basic_regression, mauna_loa (kernel composition)
└── python/              pybind11 bindings + pytest suite
```

## Citation

If you use lightgp in academic work, please cite:

```bibtex
@misc{fang2026lightgp,
  title  = {lightgp: Lightweight Gaussian Process Inference in C++ on Metal and CUDA},
  author = {YuHsueh Fang},
  year   = {2026},
  url    = {https://github.com/Fangop/lightgp}
}
```

## License

MIT — see [LICENSE](LICENSE).
