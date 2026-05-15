Solvers
=======

.. py:class:: lightgp.Solver

   Inference method for :class:`GPExact`. Pick one based on the size and
   structure of your problem.

   .. py:attribute:: Cholesky

      Direct exact GP via the Cholesky factorization of
      :math:`K + \sigma_n^2 I`. Cost :math:`O(N^3)` time,
      :math:`O(N^2)` memory.

      Best for :math:`N \lesssim 5{,}000` and any :math:`D`.

   .. py:attribute:: CG

      Conjugate-gradient solve of :math:`(K + \sigma_n^2 I)\alpha = y`.
      Cost :math:`O(N^2 k)` time per fit, where :math:`k` is the iteration
      count (typically 30–100). With the Metal or CUDA backend the
      matrix-vector products go through the matrix-free RBF kernel — the
      :math:`N \times N` kernel is never materialized, so peak memory
      stays :math:`O(N)`.

      Best for :math:`5{,}000 \lesssim N \lesssim 50{,}000` and any
      :math:`D`. Variance estimates use a stochastic Hutchinson probe
      estimator; the log marginal likelihood uses stochastic Lanczos
      quadrature.

   .. py:attribute:: SKI

      Structured Kernel Interpolation (KISS-GP, Wilson & Nickisch 2015).
      Approximates :math:`K \approx W K_{\text{grid}} W^\top` where
      :math:`W` is a sparse cubic-interpolation matrix from data to a
      regular grid, and :math:`K_{\text{grid}}` is (multi-D)
      Toeplitz-structured. Matrix-vector products are then
      :math:`O(M \log M)` via FFT (cuFFT on CUDA, vDSP on macOS).

      Best for :math:`N \gtrsim 100{,}000` and :math:`D \le 3`. Beyond
      :math:`D = 3` the grid size :math:`M^D` explodes.

Picking a solver
----------------

A rough decision tree:

.. code-block:: text

   Is N < 5000?
       → Solver.Cholesky (exact, simple)
   Is D ≤ 3 and N > 50000?
       → Solver.SKI (FFT trick, very fast)
   Is D > 3 and N > 5000?
       → Solver.CG (matrix-free) or use GPSparse (inducing-point)
