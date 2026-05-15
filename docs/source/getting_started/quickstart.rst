Quick Start
===========

Your first GP regression in six lines:

.. code-block:: python

   import numpy as np
   import lightgp as gp

   X = np.sort(np.random.uniform(-3, 3, 50)).reshape(-1, 1).astype(np.float32)
   y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(50)).astype(np.float32)

   model = gp.GPExact(gp.RBF(), noise_var=0.01)
   model.fit(X, y)
   model.optimize(steps=100)

   X_test = np.linspace(-4, 4, 200).reshape(-1, 1).astype(np.float32)
   pred = model.predict(X_test)
   print(pred["mean"].shape, pred["var"].shape)   # (200,) (200,)

The predict method returns a dictionary with two keys:

- ``"mean"`` — posterior mean predictions, shape ``(N_test,)``
- ``"var"`` — posterior marginal variance, shape ``(N_test,)``

Plotting the result
-------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   mu = pred["mean"]
   std = np.sqrt(pred["var"])
   plt.fill_between(X_test[:, 0], mu - 2 * std, mu + 2 * std, alpha=0.2)
   plt.plot(X_test[:, 0], mu, lw=2, label="GP mean")
   plt.scatter(X[:, 0], y, c="black", s=20, zorder=5, label="data")
   plt.legend()
   plt.show()

What's happening under the hood
--------------------------------

1. ``gp.RBF()`` constructs an RBF (squared-exponential) kernel with default
   length scale ``ℓ = 1`` and signal variance ``σ² = 1``.
2. ``model.fit(X, y)`` Cholesky-factorizes ``K + σ_n² I`` and computes
   ``α = K_y⁻¹ y``.
3. ``model.optimize(100)`` runs 100 steps of Adam over the log-hyperparameters
   ``[log ℓ, log σ², log σ_n²]`` to maximize the log marginal likelihood.
4. ``model.predict(X_test)`` returns ``μ* = K_*ᵀ α`` for the mean and
   ``σ*² = k(x*,x*) − K_*ᵀ K_y⁻¹ K_*`` for the marginal variance.

LightGP automatically picks the fastest backend for the problem shape:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Backend
     - Library
     - When ``Backend.Auto`` picks it
   * - ``Backend.CPU``
     - Accelerate (macOS) / OpenBLAS (Linux)
     - Dense Cholesky at any size
   * - ``Backend.Metal``
     - Apple Metal compute
     - ``D ≥ 16``, ``N ≥ 2000``, or CG at ``N > 2000``
   * - ``Backend.CUDA``
     - cuBLAS / cuSOLVER / cuFFT
     - Always preferred when compiled in

You can force a backend explicitly with the ``backend=`` constructor argument
— see :doc:`../api/backends`.
