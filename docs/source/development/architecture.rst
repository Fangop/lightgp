Architecture
============

LightGP is organized in five layers. Each one only knows about the layers
beneath it; the inference layer at the top is portable across all backends.

::

   ┌───────────────────────────────────────────────────────────┐
   │  Inference                GPExact, GPSparse               │
   │  (inference/*)            optimize(), fit(), predict()    │
   ├───────────────────────────────────────────────────────────┤
   │  Solvers                  Cholesky, CG, SLQ, SKI          │
   │  (solvers/cpu, /metal,    matrix-free matvec callback     │
   │   /cuda)                                                  │
   ├───────────────────────────────────────────────────────────┤
   │  Kernels + Mean           Kernel ABC + concrete kernels   │
   │  (kernels/*, core/mean.h) + Sum/Product/Scale composition │
   ├───────────────────────────────────────────────────────────┤
   │  Dispatch                 routes by Backend + KernelType  │
   │  (core/dispatch.cpp)      to CPU / Metal / CUDA paths     │
   ├───────────────────────────────────────────────────────────┤
   │  Backends                 Accelerate (CBLAS / LAPACK /    │
   │  (core/blas_accel,        AMX / vDSP), Metal shaders,     │
   │   kernels/metal,          cuBLAS / cuSOLVER / cuFFT       │
   │   kernels/cuda)                                           │
   └───────────────────────────────────────────────────────────┘

Tensor
------

The Tensor type (``core/tensor.h``) is the basic data container — a 2D
row-major fp32 matrix with a small set of operations:

- ``matmul`` — routes to Accelerate ``cblas_sgemm`` (macOS) or OpenBLAS
  on Linux, fallback reference implementation otherwise
- ``add_jitter`` — diagonal addition used by Cholesky retries
- ``transpose``, ``add``, elementwise ops
- ``randn``, ``zeros`` factory methods

All higher-level code allocates and passes ``Tensor`` values. The class
intentionally has no shape-polymorphic broadcasting machinery — every kernel
or solver that needs reshaping does it explicitly.

Kernel hierarchy
----------------

The abstract ``Kernel`` base in ``kernels/kernel_base.h`` declares:

.. code-block:: cpp

   virtual Tensor compute(const Tensor& X1, const Tensor& X2, Backend) const = 0;
   virtual Tensor compute_diag(const Tensor& X) const = 0;
   virtual int    num_params() const = 0;
   virtual std::vector<float> get_log_params() const = 0;
   virtual void   set_log_params(const std::vector<float>& p) = 0;
   virtual std::string name() const = 0;

Concrete kernels (``RBFKernel``, ``MaternKernel``, ``PeriodicKernel``,
``LinearKernel``) implement ``compute`` by calling into the dispatch layer.
Composite kernels (``SumKernel``, ``ProductKernel``, ``ScaleKernel``) wrap
``std::shared_ptr<Kernel>`` children and forward to them.

The parameter ordering for a composite is **stable and left-to-right**:
``(k1 + k2).get_log_params()`` returns ``k1.get_log_params() ++
k2.get_log_params()``. ``Scale(base)`` prepends its single ``log_scale``
parameter in front of the base's parameters.

Dispatch
--------

``core/dispatch.cpp`` is the single place where ``Backend`` enums are
translated into actual function calls:

.. code-block:: cpp

   Tensor dispatch_kernel(const Tensor& X1, const Tensor& X2,
                          float length_scale, float signal_variance,
                          KernelType type, Backend backend);

   bool dispatch_cholesky_with_jitter(const Tensor& K, Tensor& L,
                                      float& jitter_used, Backend backend);

   Tensor dispatch_cholesky_solve(const Tensor& L, const Tensor& b,
                                  Backend backend);

   Tensor dispatch_forward_solve(const Tensor& L, const Tensor& b,
                                 Backend backend);

Each function inspects the ``Backend`` argument, optionally consults a
shape-driven heuristic for ``Backend::Auto``, and forwards to the
platform-specific implementation. Backends that aren't compiled in fall back
to CPU.

Solvers
-------

- **Cholesky** (``solvers/cpu/cholesky_cpu``, ``solvers/metal/cholesky_metal``,
  ``solvers/cuda/cholesky_cuda``) — exact :math:`O(N^3)` factorization.
- **Conjugate Gradients** (``solvers/cpu/cg_cpu`` plus a matrix-free
  variant on Metal and CUDA) — iterative :math:`O(N^2 k)`.
- **Stochastic Lanczos Quadrature** (``solvers/cpu/slq_cpu``) — log
  determinant estimate for CG mode.
- **SKI** (``inference/ski.cpp`` + ``ski_accel.cpp`` for vDSP /
  ``ski_cuda.cu`` for cuFFT) — FFT-accelerated matvec on a grid.

Inference
---------

``GPExact`` and ``GPSparse`` are the user-facing classes. ``GPExact`` carries
either the legacy ``GPHyperparams`` (kept for binary compatibility) or the
new ``std::shared_ptr<Kernel>`` + ``std::shared_ptr<MeanFunction>``
constructor. ``fit()`` and ``predict()`` branch on which path was taken;
``optimize()`` uses analytical gradients in the legacy path and central
finite differences in the new path so it supports arbitrary kernel
compositions.

Python bindings
---------------

``python/bindings.cpp`` (pybind11) is a thin shim that:

1. Exposes the ``Kernel`` hierarchy with ``std::shared_ptr`` ownership and
   ``__add__``/``__mul__`` Python operators.
2. Converts ``numpy.ndarray`` <-> ``Tensor`` with ``forcecast`` to fp32 in
   ``numpy_to_tensor``.
3. Wraps ``GPExact::predict`` so it returns a ``dict`` with ``'mean'`` and
   ``'var'`` keys instead of mutating two output parameters.

The Python package name is ``lightgp``; the import is ``import lightgp as gp``.
