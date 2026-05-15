# LightGP

**Lightweight Gaussian Process inference in C++ with Python bindings.**
Apple Metal + Accelerate (AMX) on macOS; CUDA backend in progress. No PyTorch. No TensorFlow. Just numpy.

[![CI](https://github.com/Fangop/lightgp/actions/workflows/ci.yml/badge.svg)](https://github.com/Fangop/lightgp/actions/workflows/ci.yml)
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

## Features

| Feature                                | lightgp | GPyTorch          |
|----------------------------------------|--------|-------------------|
| Exact GP (Cholesky)                    | ✅     | ✅                |
| Matrix-free CG-based GP                | ✅     | ✅                |
| Sparse GP (Titsias VFE)                | ✅     | ✅                |
| Kernel composition (`+`, `*`, `Scale`) | ✅     | ✅                |
| RBF, Matérn-{½,3/2,5/2}, Periodic, Linear | ✅  | ✅ (+ more)       |
| Mean functions (Zero, Constant, Linear) | ✅    | ✅                |
| Zero runtime dependencies              | ✅     | ❌ (PyTorch)      |
| Apple Metal backend                    | ✅     | partial (MPS)     |
| Apple Accelerate / AMX                 | ✅     | ✅ (via PyTorch)  |
| CUDA backend                           | 🚧 WIP | ✅                |
| `pip install`                          | ✅     | ✅                |
| Embeddable in pure C++ projects        | ✅     | ❌                |
| Matrix-free $K\mathbf v$ on Metal      | ✅     | ❌                |

## Benchmarks (Apple M4, fp32, median of 3 runs)

End-to-end fit + predict against GPyTorch on the same hardware:

| Config                            | lightgp CPU | lightgp Metal | GPyTorch CPU | GPyTorch MPS | lightgp vs GPyTorch (best) |
|-----------------------------------|-----------|--------------|--------------|--------------|---------------------------|
| Exact RBF, N=2048, D=4            | **44 ms** | 195 ms       | 89 ms        | (gap*)       | **2.0× faster**           |
| Exact Matérn-5/2, N=2048, D=4     | **42 ms** | 191 ms       | 106 ms       | (gap*)       | **2.5× faster**           |
| Sparse RBF, N=10000, M=200        | **25 ms** | 42 ms        | 42 ms        | 69 ms        | **1.7× faster**           |
| Sparse RBF, N=50000, M=200        | **114 ms**| 156 ms       | 196 ms       | **98 ms**    | 1.16× slower (vs MPS)     |
| Matrix-free $K\mathbf v$, N=20000 | n/a       | **22 ms**    | n/a          | (no equiv)   | **60× over explicit**     |

*GPyTorch MPS falls back to CPU for exact-GP variance because `aten::_linalg_eigh.eigenvalues` is not implemented on MPS.

lightgp CPU beats GPyTorch CPU at every measured size — same Accelerate / AMX underneath, less Python dispatch overhead. The matrix-free $K\mathbf v$ path is unique to lightgp on Apple Silicon and enables CG-based GP inference at N=50000+ with O(N) memory (vs O(N²) for the explicit kernel matrix).

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

### Linux (CPU-only for now, CUDA backend coming)

```bash
LIGHTGP_NO_METAL=1 LIGHTGP_NO_ACCELERATE=1 ./build.sh
./build/run_tests
```

Install OpenBLAS / LAPACK first to get the fast CPU path:

```bash
sudo apt install libopenblas-dev liblapacke-dev   # Debian / Ubuntu
```

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
