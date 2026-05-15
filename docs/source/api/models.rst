Models
======

GPExact
-------

.. py:class:: lightgp.GPExact(kernel, mean=None, noise_var=0.1, backend=Backend.Auto, solver=Solver.Cholesky)

   Exact Gaussian Process regression.

   :param kernel: A :class:`lightgp.Kernel` (possibly composed).
   :param mean: A :class:`lightgp.Mean`. Defaults to :class:`ZeroMean` when
       omitted.
   :param noise_var: Observation noise variance :math:`\sigma_n^2`.
   :param backend: Compute backend. :attr:`Backend.Auto` selects the fastest
       available for the problem shape (see :doc:`backends`).
   :param solver: Inference method (see :doc:`solvers`).

   Solver complexity:

   .. list-table::
      :header-rows: 1
      :widths: 25 25 50

      * - Solver
        - Time per fit
        - When to use
      * - ``Cholesky``
        - :math:`O(N^3)`
        - :math:`N < 5{,}000`. Exact.
      * - ``CG``
        - :math:`O(N^2 k)`
        - :math:`5{,}000 \le N < 50{,}000`. Matrix-free on GPU
          (no :math:`N \times N` kernel matrix in memory).
      * - ``SKI``
        - :math:`O(N \log N \cdot k)`
        - :math:`N > 50{,}000` and :math:`D \le 3`. FFT-based.

   .. py:method:: fit(X, y) -> bool

      Fit the GP to training data.

      :param X: Inputs, shape ``(N, D)``, dtype ``float32``.
      :param y: Targets, shape ``(N,)``, dtype ``float32``.
      :returns: ``True`` if the (jittered) Cholesky or CG solve succeeded.

   .. py:method:: predict(X_test) -> dict

      Posterior mean + variance at test points.

      :param X_test: Test inputs, shape ``(M, D)``, ``float32``.
      :returns: ``dict`` with keys ``"mean"`` (M,) and ``"var"`` (M,).

   .. py:method:: optimize(steps=100, lr=0.05, verbose=False) -> bool

      Adam optimizer on the flat log-hyperparameter vector
      ``[kernel.params, log_noise, mean.params]``. With the new
      ``Kernel`` API the gradient is computed by central finite differences
      to support arbitrary compositions.

   .. py:method:: log_marginal_likelihood() -> float

      The (negative-Hutchinson-estimated when in CG mode) log marginal
      likelihood under the current hyperparameters.

   .. py:method:: fitted() -> bool

      ``True`` once a successful ``fit()`` has been called.

GPSparse
--------

.. py:class:: lightgp.GPSparse(length_scale=1.0, signal_var=1.0, noise_var=0.1, backend=Backend.Auto)

   Sparse GP regression using Titsias' variational free energy (VFE) bound,
   2009. Scales to :math:`N \gtrsim 10^5` with :math:`M` inducing points.

   :param length_scale: RBF length scale (legacy API; the
       ``Kernel``-object refactor for GPSparse is on the roadmap).
   :param signal_var: RBF signal variance.
   :param noise_var: Observation noise variance.
   :param backend: Compute backend.

   .. py:method:: fit(X, y, num_inducing=100) -> bool

      Fit with ``M`` inducing points (initialized by farthest-point sampling).

   .. py:method:: predict(X_test) -> dict

      Same return shape as :py:meth:`GPExact.predict`.

   .. py:method:: log_marginal_likelihood() -> float

      VFE bound at the current state.

   .. py:method:: fitted() -> bool
