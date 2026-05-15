#!/usr/bin/env python3
"""Build the six tutorial .ipynb files for the LightGP docs.

Each tutorial is constructed programmatically with nbformat, then executed with
nbconvert against the lightgp-docs kernel so plot outputs are baked into the
saved notebook. Sphinx renders the already-executed notebooks via nbsphinx
(with nbsphinx_execute = 'never').

Run once from the repo root after the Python module is built:

    source .venv/bin/activate
    PYTHONPATH=python python3 docs/build_tutorial_notebooks.py
"""
from __future__ import annotations

import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "docs", "source", "tutorials")
os.makedirs(OUT, exist_ok=True)

# Cell helpers.
def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


# Common preamble for the lightgp/numpy/matplotlib imports.
PREAMBLE = """
import os
import sys

# Make the locally-built lightgp importable. Real users install via 'pip install lightgp'.
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "..", "python")))

import numpy as np
import matplotlib.pyplot as plt
import lightgp as gp

rng = np.random.default_rng(0)
plt.rcParams.update({"figure.figsize": (8, 3.5), "figure.dpi": 90})
"""


# ============================================================================
# Tutorial 01 — Basic regression
# ============================================================================
T01 = [
    md("""
# Basic GP Regression

Your first Gaussian Process model in LightGP. We will:

1. Generate noisy synthetic data
2. Fit a GP with an RBF kernel
3. Visualize predictions with uncertainty bands
4. Improve the fit by optimizing hyperparameters
5. Compare different kernel smoothnesses
"""),
    code(PREAMBLE),
    md("## Generate noisy data\n\nSample 40 points from `y = sin(x) + noise`."),
    code("""
X = np.sort(rng.uniform(-3, 3, 40)).reshape(-1, 1).astype(np.float32)
y = (np.sin(X[:, 0]) + 0.15 * rng.standard_normal(40)).astype(np.float32)
X_test = np.linspace(-4, 4, 250).reshape(-1, 1).astype(np.float32)
print("X:", X.shape, X.dtype, "  y:", y.shape, y.dtype)
"""),
    md("""
## Fit a GP

`GPExact(kernel)` is the simplest entry point. We start with an RBF kernel
at a deliberately bad initial length scale so the un-optimized fit looks rough.
"""),
    code("""
model = gp.GPExact(gp.RBF(length_scale=0.3), noise_var=0.1)
model.fit(X, y)
pred_before = model.predict(X_test)
ll_before = model.log_marginal_likelihood()
print(f"log marginal likelihood before optimize: {ll_before:.3f}")
"""),
    md("## Optimize the hyperparameters\n\nLightGP uses Adam on the log-hyperparameters to maximize the log marginal likelihood."),
    code("""
model.optimize(steps=80)
pred_after = model.predict(X_test)
ll_after = model.log_marginal_likelihood()
print(f"log marginal likelihood after optimize:  {ll_after:.3f}")
print(f"improvement: {ll_after - ll_before:+.3f}")
"""),
    md("## Plot before vs after"),
    code("""
def plot_gp(ax, X, y, X_test, pred, title):
    mu, sd = pred["mean"], np.sqrt(np.maximum(pred["var"], 0))
    ax.fill_between(X_test[:, 0], mu - 2*sd, mu + 2*sd, alpha=0.18, color="#2563EB")
    ax.plot(X_test[:, 0], mu, lw=2, color="#2563EB", label="GP mean")
    ax.plot(X_test[:, 0], np.sin(X_test[:, 0]), "--", color="#6B7280", lw=1, label="true f")
    ax.scatter(X[:, 0], y, c="black", s=18, zorder=5, label="data")
    ax.set_title(title); ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("x"); ax.set_ylabel("y")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
plot_gp(a, X, y, X_test, pred_before, "Before optimize()")
plot_gp(b, X, y, X_test, pred_after, "After optimize()")
plt.tight_layout(); plt.show()
"""),
    md("## Different kernel smoothnesses\n\nThe Matérn-ν family interpolates between rough (ν=½) and smooth (ν=∞ = RBF) fits."),
    code("""
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.scatter(X[:, 0], y, c="black", s=18, zorder=5, label="data")
for nu, color in [(0.5, "#F59E0B"), (1.5, "#10B981"), (2.5, "#2563EB")]:
    m = gp.GPExact(gp.Matern(nu=nu), noise_var=0.1)
    m.fit(X, y); m.optimize(steps=60)
    ax.plot(X_test[:, 0], m.predict(X_test)["mean"], lw=2, color=color, label=f"Matern v={nu}")
ax.legend(fontsize=9, frameon=False); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Smoothness via Matern v")
plt.tight_layout(); plt.show()
"""),
    md("""
## When to use exact GP

`Solver.Cholesky` is **O(N³)**. The rule of thumb on a laptop:

- **N < 5,000**: exact Cholesky is fine and gives the cleanest uncertainty estimates
- **5,000–50,000**: switch to `Solver.CG` (matrix-free on GPU)
- **N > 50,000 and D ≤ 3**: switch to `Solver.SKI` (FFT-accelerated)

The next tutorial shows how to combine kernels for more complex signals.
"""),
]


# ============================================================================
# Tutorial 02 — Kernel composition
# ============================================================================
T02 = [
    md("""
# Kernel Composition

A single kernel can only model one kind of structure. Real signals usually have
several: a trend plus a seasonal cycle, growth times oscillation, and so on.
LightGP exposes kernel composition via familiar Python operators:

- `k1 + k2` — `SumKernel` (additive structure)
- `k1 * k2` — `ProductKernel` (multiplicative structure)
- `gp.Scale(k)` — wraps a kernel with a learnable output scale

We will demonstrate the additive pattern on a Mauna Loa-CO₂-shaped signal.
"""),
    code(PREAMBLE),
    md("## Generate synthetic data\n\nLinear trend + slight curvature + annual seasonality + small noise."),
    code("""
N = 300
t = np.linspace(0, 30, N, dtype=np.float32).reshape(-1, 1)  # 30 years, monthly samples
y = (1.5 * t[:, 0]
     + 0.04 * t[:, 0]**2
     + 3.0 * np.sin(2 * np.pi * t[:, 0])
     + 0.5 * rng.standard_normal(N)).astype(np.float32)
t_test = np.linspace(0, 35, 400).reshape(-1, 1).astype(np.float32)  # extrapolate beyond training
"""),
    md("## Baseline: a single RBF\n\nA wide RBF + LinearMean can capture the trend but not the seasonality."),
    code("""
m_rbf = gp.GPExact(
    gp.RBF(length_scale=2.0, signal_var=10.0),
    mean=gp.LinearMean(input_dim=1),
    noise_var=0.5,
)
m_rbf.fit(t, y)
m_rbf.optimize(steps=80)
p_rbf = m_rbf.predict(t_test)
"""),
    md("## Composed kernel\n\n`Scale(RBF_long) + Scale(Periodic) + Scale(RBF_short)`:"),
    code("""
kernel = (gp.Scale(gp.RBF(length_scale=10.0))           # decade-scale trend wiggle
          + gp.Scale(gp.Periodic(length_scale=1.0, period=1.0))   # annual cycle
          + gp.Scale(gp.RBF(length_scale=0.5)))         # short-term irregular
print("kernel:", kernel.name())
print("num parameters:", kernel.num_params())

m_comp = gp.GPExact(kernel, mean=gp.LinearMean(input_dim=1), noise_var=0.5)
m_comp.fit(t, y)
m_comp.optimize(steps=120)
p_comp = m_comp.predict(t_test)
"""),
    md("## Compare side-by-side"),
    code("""
def panel(ax, p, title):
    mu, sd = p["mean"], np.sqrt(np.maximum(p["var"], 0))
    ax.fill_between(t_test[:, 0], mu - 2*sd, mu + 2*sd, alpha=0.15, color="#2563EB")
    ax.plot(t_test[:, 0], mu, lw=2, color="#2563EB")
    ax.scatter(t[:, 0], y, s=6, color="black", zorder=3)
    ax.axvline(30, color="#6B7280", ls=":", lw=0.8)
    ax.set_title(title); ax.set_xlabel("year")

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
panel(a, p_rbf, "Single RBF")
panel(b, p_comp, "Scale(RBF) + Scale(Periodic) + Scale(RBF)")
a.set_ylabel("y")
plt.tight_layout(); plt.show()
"""),
    md("""
## What to compose, when?

| Pattern | Combiner | Example |
|---------|----------|---------|
| Two independent structures present at the same scale | `+` | Trend + seasonality |
| One signal modulates the amplitude of another | `*` | Growing seasonal wave |
| Two patterns each need their own output scale | `Scale(k1) + Scale(k2)` | Almost always — gives the optimizer per-component freedom |

A common starting recipe:

```python
kernel = gp.Scale(gp.RBF()) + gp.Scale(gp.Periodic(period=p_initial))
```

The next tutorial scales these ideas up with sparse inducing-point GPs.
"""),
]


# ============================================================================
# Tutorial 03 — Sparse GP
# ============================================================================
T03 = [
    md("""
# Sparse GP — Inducing Points

Exact Cholesky is O(N³). For larger datasets, **sparse GPs** approximate the
full kernel matrix using **M ≪ N inducing points**. LightGP's `GPSparse` uses
the variational free-energy (VFE) bound of Titsias (2009).

The trade-off: more inducing points → tighter approximation + more compute.
"""),
    code(PREAMBLE),
    md("## A larger dataset\n\nN = 1500 noisy samples from `sin + cos` mixture."),
    code("""
N = 1500
X = np.sort(rng.uniform(-5, 5, N)).reshape(-1, 1).astype(np.float32)
y = (np.sin(X[:, 0]) + 0.4 * np.cos(2 * X[:, 0])
     + 0.15 * rng.standard_normal(N)).astype(np.float32)
X_test = np.linspace(-5, 5, 400).reshape(-1, 1).astype(np.float32)
print("N =", N)
"""),
    md("## Sweep the number of inducing points M"),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, M in zip(axes, [10, 30, 100]):
    m = gp.GPSparse(length_scale=1.0, signal_var=1.0, noise_var=0.03)
    m.fit(X, y, num_inducing=M)
    p = m.predict(X_test)
    mu, sd = p["mean"], np.sqrt(np.maximum(p["var"], 0))
    ax.scatter(X[:, 0], y, c="lightgray", s=4, zorder=1)
    ax.fill_between(X_test[:, 0], mu - 2*sd, mu + 2*sd, alpha=0.18, color="#2563EB")
    ax.plot(X_test[:, 0], mu, lw=2, color="#2563EB")
    ax.set_title(f"M = {M}")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
plt.tight_layout(); plt.show()
"""),
    md("""
## Picking M

Rules of thumb:

| Regime | M |
|--------|---|
| Quick prototyping | `M ~ sqrt(N)` |
| Production accuracy | `M ~ N / 10`, up to ~200 |
| Memory-bound | Choose the largest M that fits — Sparse storage is O(N + M²) |

For dimensions D > 1, place inducing points throughout the input space.
LightGP initializes by **farthest-point sampling** on the training inputs —
deterministic and a strong default.
"""),
    md("## Timing scaling — fit time at fixed M=100"),
    code("""
import time
sizes = [500, 2000, 8000]
times = []
for n in sizes:
    Xn = np.sort(rng.uniform(-5, 5, n)).reshape(-1, 1).astype(np.float32)
    yn = (np.sin(Xn[:, 0]) + 0.15 * rng.standard_normal(n)).astype(np.float32)
    m = gp.GPSparse(noise_var=0.03)
    # warmup
    m.fit(Xn, yn, num_inducing=100)
    t0 = time.perf_counter()
    m.fit(Xn, yn, num_inducing=100)
    times.append((time.perf_counter() - t0) * 1000.0)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(sizes, times, "o-", color="#2563EB", lw=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N"); ax.set_ylabel("GPSparse.fit (ms)")
ax.set_title("Sparse GP scales near-linearly in N (fixed M=100)")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.show()
"""),
    md("""
## When to use sparse vs CG vs SKI

- **Any D, N up to ~100k**: `GPSparse` with M = 100–500
- **N up to ~50k, kernel matrix fits in GPU memory**: `Solver.CG` (matrix-free on GPU)
- **N ≫ 100k, D ≤ 3**: `Solver.SKI` (next tutorial)

Sparse GP wins on flexibility — it handles high-D inputs and irregular grids,
which SKI cannot. The next tutorial covers the SKI sweet spot.
"""),
]


# ============================================================================
# Tutorial 04 — Large-scale SKI
# ============================================================================
T04 = [
    md("""
# Large-Scale GP with SKI

Structured Kernel Interpolation (KISS-GP) turns matrix-vector products against
the kernel matrix into **FFTs**, dropping the cost from O(N²) to O(N log N) per
matvec. On Apple Silicon LightGP uses Accelerate's vDSP DFT; on NVIDIA it uses
cuFFT.

Restrictions:

- Only practical for **D ≤ 3** (the grid size explodes in higher dimensions)
- Approximation quality depends on grid density
- Best for stationary, well-conditioned kernels (RBF / Matérn on smooth data)
"""),
    code(PREAMBLE),
    md("## Fit 50,000 points\n\n`Solver.SKI` + `Backend.CPU` selects the vDSP path on macOS."),
    code("""
import time

N = 50_000
X = np.linspace(-10, 10, N, dtype=np.float32).reshape(-1, 1)
y = (np.sin(X[:, 0]) + 0.15 * rng.standard_normal(N)).astype(np.float32)

model = gp.GPExact(gp.RBF(), solver=gp.Solver.SKI, backend=gp.Backend.CPU, noise_var=0.05)
t0 = time.perf_counter()
model.fit(X, y)
fit_ms = (time.perf_counter() - t0) * 1000.0
print(f"SKI fit at N={N}: {fit_ms:.1f} ms")
"""),
    md("## Predict and plot a slice"),
    code("""
X_test = np.linspace(-10, 10, 600).reshape(-1, 1).astype(np.float32)
t0 = time.perf_counter()
pred = model.predict(X_test)
pred_ms = (time.perf_counter() - t0) * 1000.0
print(f"SKI predict at M_test={len(X_test)}: {pred_ms:.1f} ms")

mu, sd = pred["mean"], np.sqrt(np.maximum(pred["var"], 0))
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.scatter(X[::500, 0], y[::500], c="lightgray", s=4, label="data (subsampled)")
ax.plot(X_test[:, 0], np.sin(X_test[:, 0]), color="#6B7280", ls="--", lw=1, label="true sin(x)")
ax.fill_between(X_test[:, 0], mu - 2*sd, mu + 2*sd, alpha=0.2, color="#2563EB")
ax.plot(X_test[:, 0], mu, lw=2, color="#2563EB", label="SKI prediction")
ax.legend(fontsize=9, frameon=False); ax.set_title(f"SKI GP at N={N:,}")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.show()
"""),
    md("""
## How SKI works (one-paragraph intuition)

1. Project the data onto a regular grid via a sparse cubic-interpolation matrix W: `K ≈ W K_grid W^T`
2. The grid kernel `K_grid` is Toeplitz (1D) or Kronecker-Toeplitz (2D / 3D)
3. Toeplitz matrix-vector products are FFTs in O(M log M)
4. Conjugate gradients with this fast matvec solves the GP equations without ever forming K

The cost shifts from `O(N²)` per matvec to `O(N + M log M)`.
"""),
    md("""
## Decision flowchart

```text
N < 5,000?                          → Solver.Cholesky (exact, simple)
5k ≤ N < 50k, any D?                → Solver.CG (matrix-free on GPU)
N > 50k, D ≤ 3?                     → Solver.SKI  ← you are here
N > 50k, D > 3?                     → GPSparse (inducing-point)
```

The next tutorial covers how to control the compute backend (CPU / Metal / CUDA).
"""),
]


# ============================================================================
# Tutorial 05 — Backends
# ============================================================================
T05 = [
    md("""
# Choosing a Backend

LightGP picks the fastest available backend automatically with `Backend.Auto`.
This tutorial shows the available choices, what `Auto` does, and how to force
a specific backend when benchmarking or debugging.
"""),
    code(PREAMBLE),
    md("""
## Available backends

| Backend | Library | Available on |
|---------|---------|--------------|
| `Backend.CPU` | reference C++ + Accelerate (macOS) / OpenBLAS (Linux) | always |
| `Backend.Metal` | Apple Metal Shading Language compute shaders | macOS builds |
| `Backend.CUDA` | cuBLAS + cuSOLVER + cuFFT + custom kernels | CUDA-enabled builds |
| `Backend.Auto` | resolves at fit time based on problem shape | always |
"""),
    md("## Forcing a specific backend"),
    code("""
N, D = 1500, 4
X = rng.standard_normal((N, D)).astype(np.float32)
y = (np.sin(X[:, 0]) + 0.1 * rng.standard_normal(N)).astype(np.float32)

for name, backend in [
    ("CPU",   gp.Backend.CPU),
    ("Metal", gp.Backend.Metal),
    ("Auto",  gp.Backend.Auto),
]:
    try:
        m = gp.GPExact(gp.RBF(), backend=backend, noise_var=0.05)
        m.fit(X, y)
        print(f"{name:<6}  fit ok")
    except Exception as e:
        print(f"{name:<6}  not available: {e}")
"""),
    md("## Quick wall-time comparison"),
    code("""
import time
results = {}
for name, backend in [
    ("CPU",   gp.Backend.CPU),
    ("Metal", gp.Backend.Metal),
    ("Auto",  gp.Backend.Auto),
]:
    try:
        m = gp.GPExact(gp.RBF(), backend=backend, noise_var=0.05)
        m.fit(X, y); m.predict(X[:16])   # warmup
        t0 = time.perf_counter()
        m.fit(X, y); m.predict(X[:16])
        results[name] = (time.perf_counter() - t0) * 1000.0
    except Exception:
        pass

fig, ax = plt.subplots(figsize=(6, 3.4))
names = list(results.keys()); vals = list(results.values())
ax.bar(names, vals, color=["#2563EB", "#10B981", "#F59E0B"][: len(names)])
for n, v in zip(names, vals):
    ax.text(n, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("fit + predict (ms)")
ax.set_title(f"GPExact at N={N}, D={D}")
plt.tight_layout(); plt.show()
"""),
    md("""
## How `Backend.Auto` chooses

The dispatch heuristic, derived empirically and baked into
`resolve_auto_backend`:

```text
Solver.SKI                           → CPU (vDSP, on macOS)
Solver.CG, N > 2000                  → Metal (matrix-free matvec)
D ≥ 16 and N ≥ 2000                  → Metal (kernel construction)
otherwise                            → CPU (AMX Cholesky wins)
```

A surprising finding: for moderate-N **dense Cholesky** on Apple Silicon, the
CPU (with its AMX matrix coprocessor) **beats** the integrated Metal GPU.
`Auto` reflects that.

## Cross-platform code

The same Python script runs unchanged on macOS and Linux. With `Backend.Auto`,
LightGP picks the Metal / CUDA / OpenBLAS path that fits the host.
"""),
]


# ============================================================================
# Tutorial 06 — C++ usage (this one is just an RST page, not a notebook).
# Note: we still emit it from this script for consistency, but as an .rst file.
# ============================================================================
T06_RST = """\
Using LightGP from C++
======================

LightGP is a dependency-free C++17 library. The Python bindings are a thin
pybind11 shim — every line of GP math lives in the C++ core. This means you
can embed LightGP in iOS apps, robotics stacks, game engines, or any other
C++ codebase, with no Python runtime requirement.

Building the C++ library
------------------------

.. code-block:: bash

   git clone https://github.com/Fangop/lightgp.git
   cd lightgp
   ./build.sh        # auto-detects Metal + Accelerate on macOS

After ``./build.sh``, every public symbol is available by including the
relevant header and linking the produced ``build/*.o`` object files (or use
the CMake target ``lightgp`` from the install tree).

Basic regression
----------------

.. code-block:: cpp

   #include "core/tensor.h"
   #include "kernels/rbf_kernel.h"
   #include "core/mean.h"
   #include "inference/gp_exact.h"

   using namespace lightgp;

   // Generate 100 points from sin(x).
   const int N = 100;
   Tensor X(N, 1);
   Tensor y(N, 1);
   for (int i = 0; i < N; ++i) {
       const float x = -3.0f + 6.0f * static_cast<float>(i) / (N - 1);
       X(i, 0) = x;
       y(i, 0) = std::sin(x);
   }

   auto kernel = std::make_shared<RBFKernel>(/*length_scale=*/1.0f);
   auto mean   = std::make_shared<ZeroMean>();
   GPExact gp(kernel, mean, /*noise_var=*/0.01f);

   gp.fit(X, y);
   gp.optimize_hyperparameters(/*steps=*/100);

   Tensor mean_pred, var_pred;
   gp.predict(X, mean_pred, var_pred);

Kernel composition
------------------

.. code-block:: cpp

   #include "kernels/composite_kernel.h"
   #include "kernels/periodic_kernel.h"

   auto kernel = gp::scale(std::make_shared<RBFKernel>())
               + gp::scale(std::make_shared<PeriodicKernel>(1.0f, 1.0f, 1.0f));

The free functions ``scale``, ``operator+``, and ``operator*`` are defined in
``kernels/composite_kernel.h`` and accept any ``std::shared_ptr<Kernel>``.

Backend selection
-----------------

.. code-block:: cpp

   #include "core/backend.h"

   // Force the Metal backend even when Auto would pick CPU:
   GPExact gp(kernel, mean, 0.01f, Backend::Metal);

   // Use CG (matrix-free on Metal) for very large N:
   GPExact gp(kernel, mean, 0.01f, Backend::Auto, Solver::CG);

   // SKI on macOS uses the Accelerate vDSP FFT path automatically:
   GPExact gp(kernel, mean, 0.01f, Backend::Auto, Solver::SKI);

iOS embedding (sketch)
----------------------

The C++ core has no runtime dependencies beyond Accelerate and Metal — both
are stock iOS frameworks. To embed:

1. Add the LightGP source tree to your Xcode project (or build it as a static
   library via the bundled ``CMakeLists.txt``).
2. Compile every ``*.cpp`` and ``*.mm`` under ``core/``, ``kernels/``,
   ``solvers/``, and ``inference/``.
3. Link against ``Accelerate.framework`` and ``Metal.framework``.
4. From Swift, expose your wrapper via an Objective-C++ bridging header.

CMake integration
-----------------

For projects that consume LightGP via ``find_package`` / ``add_subdirectory``:

.. code-block:: cmake

   add_subdirectory(${LIGHTGP_DIR})
   target_link_libraries(my_app PRIVATE lightgp)

The CMake target ``lightgp`` exposes the public include directories and links
``Accelerate`` / ``Metal`` automatically when present.
"""


# ============================================================================
# Build + execute each notebook.
# ============================================================================
def build(name: str, cells: list) -> str:
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {
            "name": "lightgp-docs",
            "display_name": "lightgp-docs",
            "language": "python",
        },
        "language_info": {"name": "python"},
    })
    ep = ExecutePreprocessor(timeout=600, kernel_name="lightgp-docs")
    ep.preprocess(nb, {"metadata": {"path": OUT}})
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"  built {name}")
    return path


def write_rst(name: str, content: str):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"  built {name}")


if __name__ == "__main__":
    print("Building tutorial notebooks...")
    build("01_basic_regression.ipynb", T01)
    build("02_kernel_composition.ipynb", T02)
    build("03_sparse_gp.ipynb", T03)
    build("04_large_scale_ski.ipynb", T04)
    build("05_choosing_backends.ipynb", T05)
    write_rst("06_cpp_usage.rst", T06_RST)
    print("\nDone.")
