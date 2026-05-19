Installation
============

From PyPI
---------

.. code-block:: bash

   pip install lightgp

This installs a precompiled wheel for the host platform:

- **macOS (Apple Silicon)**: Metal GPU + Accelerate (AMX) CPU backends are
  compiled in and auto-detected at runtime.
- **Linux x86_64**: OpenBLAS / LAPACKE CPU path. CUDA is enabled if an
  NVIDIA-capable wheel is selected by ``pip``.

From source
-----------

For development or to control which backends are enabled:

.. code-block:: bash

   git clone https://github.com/Fangop/lightgp.git
   cd lightgp

   # macOS (Metal + Accelerate auto-detected)
   ./build.sh
   cd python && pip install -e ".[test]"

   # Linux (CPU only)
   LIGHTGP_NO_METAL=1 LIGHTGP_NO_ACCELERATE=1 ./build.sh

   # Linux with CUDA
   LIGHTGP_ENABLE_CUDA=1 ./build.sh

   # Opt-out flags work independently:
   GPLITE_NO_METAL=1 ./build.sh             # disable Metal even on Darwin
   GPLITE_NO_ACCELERATE=1 ./build.sh        # use reference C++ instead of Apple BLAS

Verify
------

.. code-block:: pycon

   >>> import lightgp as gp
   >>> gp.__version__
   '0.1.2'
   >>> # Check that the SKI solver is wired in:
   >>> hasattr(gp.Solver, "SKI")
   True

Requirements
------------

- Python ≥ 3.9
- NumPy ≥ 1.20
- A C++17 compiler (handled by ``pip install`` for source builds)
- *Optional*: NVIDIA CUDA Toolkit ≥ 11.0 for the CUDA backend

Build flags
-----------

The environment variables below toggle the build script behaviour. They are
read by ``./build.sh`` and ``./python/build_python.sh``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Effect
   * - ``LIGHTGP_NO_METAL=1``
     - Skip the Metal compute backend even on macOS.
   * - ``LIGHTGP_NO_ACCELERATE=1``
     - Use the reference C++ matmul / Cholesky instead of Apple Accelerate.
       Useful for cross-checking and for non-Apple platforms.
   * - ``LIGHTGP_ENABLE_CUDA=1``
     - Compile in the CUDA backend (requires ``nvcc``).
