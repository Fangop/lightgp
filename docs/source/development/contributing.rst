Contributing
============

LightGP welcomes pull requests. The project values:

- **Small, focused changes** — one feature or fix per PR.
- **Tests for new behaviour** — every kernel, solver, or backend addition
  comes with C++ assertions and a Python integration test.
- **Honest benchmarks** — we publish wall-time numbers against the best
  alternative we can find on the same hardware.

Repo at a glance
----------------

.. code-block:: text

   lightgp/
   ├── core/           Tensor, Backend / Solver / KernelType enums, dispatch, Accelerate wrappers
   ├── kernels/        Kernel hierarchy (RBF, Matern, Periodic, Linear, Sum/Product/Scale)
   │   ├── cpu/        reference + Accelerate CPU paths
   │   ├── metal/      Metal Shading Language compute shaders
   │   └── cuda/       CUDA C++ + cuBLAS / cuSOLVER kernels
   ├── solvers/        Cholesky, conjugate gradients, Lanczos log-det, SKI
   ├── inference/      GPExact, GPSparse
   ├── data/           bundled benchmark datasets (motorcycle, mauna_loa stand-ins)
   ├── tests/          ~330k C++ assertions across 15+ test groups
   ├── benchmarks/     C++ benches + Python GPyTorch comparison script
   ├── python/         pybind11 bindings + pytest suite
   └── docs/           this Sphinx documentation

Dev loop
--------

.. code-block:: bash

   # 1. Pick up a branch
   git checkout -b feature/my-change

   # 2. C++ build + tests
   ./build.sh
   ./build/run_tests              # ~330k assertions

   # 3. Python bindings + tests
   source .venv/bin/activate
   ./python/build_python.sh
   PYTHONPATH=python python3 -m pytest python/tests -v

   # 4. Run any affected benchmark
   ./build/bench_paper            # general perf sweep
   ./build/bench_ski              # SKI-specific
   ./build/bench_matvec           # matrix-free matvec

Both the C++ test suite and the Python tests must be green before opening a
PR. GitHub Actions runs both on macOS (Metal + Accelerate) and Linux (CPU
only) — see ``.github/workflows/ci.yml``.

Style
-----

- **C++17**, no exceptions in the hot path. Return ``bool``/``std::optional``
  instead.
- **Header-only where reasonable** — the public ``Kernel`` hierarchy is
  header-only; implementations live in matching ``.cpp`` files when the
  symbol is shared across translation units.
- **Naming**: ``snake_case`` for functions and variables, ``PascalCase``
  for types, ``LIGHTGP_HAS_<X>`` for build-flag macros.
- **Python**: PEP 8, ``ruff``-friendly. Public Python names match the C++
  ones (``RBFKernel`` → ``RBF`` to keep imports concise).

Reporting an issue
------------------

A useful issue report has:

- A minimal reproducer (Python script preferred — under 30 lines)
- The output of ``python3 -c "import lightgp; print(lightgp.__version__)"``
- Which backend was active (``CPU`` / ``Metal`` / ``CUDA`` / ``Auto``)
- The hardware (Apple chip model, or NVIDIA GPU + CUDA version)
