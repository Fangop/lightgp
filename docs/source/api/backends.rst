Backends
========

.. py:class:: lightgp.Backend

   Compute backend selector. Each model takes a ``backend=`` constructor
   argument; the default is :attr:`Backend.Auto`.

   .. py:attribute:: CPU

      Reference C++ paths, or Apple Accelerate (macOS) / OpenBLAS (Linux)
      when compiled in. Cholesky always goes through CBLAS/LAPACK when
      available — on Apple Silicon this means the AMX matrix coprocessor,
      which is faster than the integrated GPU for moderate-N dense
      Cholesky.

   .. py:attribute:: Metal

      Apple Metal compute shaders for RBF, Matérn (function-constant
      specialized), GEMM (naive / tiled / simdgroup-matrix), triangular
      solve, blocked Cholesky, and the matrix-free RBF kernel-vector
      product. Only available on Apple Silicon builds.

   .. py:attribute:: CUDA

      cuBLAS GEMM, cuSOLVER Cholesky, cuFFT (for SKI), plus custom CUDA
      kernels for RBF / Matérn matrix construction and matrix-free
      matvec. Only available in CUDA-enabled builds.

   .. py:attribute:: Auto

      Resolves at fit time based on the problem shape. See the dispatch
      heuristic in :func:`lightgp.resolve_auto_backend` (and the rules
      below).

Auto dispatch rules
-------------------

On a CUDA-enabled build, ``Auto`` routes everything to ``CUDA`` once
:math:`N \ge 1024` (Cholesky) or unconditionally (CG / SKI).

On Apple Silicon (no CUDA), the rules are:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Condition
     - Resolution
     - Why
   * - :attr:`Solver.SKI`
     - ``CPU`` (vDSP)
     - The Toeplitz FFT path lives only in Accelerate vDSP.
   * - :attr:`Solver.CG` and :math:`N > 2000`
     - ``Metal``
     - Matrix-free RBF matvec on Metal wins by ~60× vs explicit.
   * - :math:`D \ge 16` and :math:`N \ge 2000`
     - ``Metal``
     - Tiled / float4 kernel construction beats CPU once the kernel
       matrix is the dominant cost.
   * - otherwise
     - ``CPU``
     - Dense Cholesky on AMX is the fastest path for low-D, moderate-N.

These rules are empirical — see the
:doc:`benchmarks <../benchmarks/index>` for the underlying measurements.
