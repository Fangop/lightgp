GP Regression Basics
====================

A Gaussian Process defines a distribution over functions:

.. math::

   f \sim \mathcal{GP}\!\bigl(m(x),\; k(x, x')\bigr)

It is fully characterized by a **mean function** :math:`m(x)` and a
**kernel** (covariance function) :math:`k(x, x')`. For any finite set of
inputs :math:`X = (x_1, \dots, x_N)`, the function values
:math:`f(X) = (f(x_1), \dots, f(x_N))` are jointly Gaussian:

.. math::

   f(X) \sim \mathcal{N}\!\bigl(m(X),\; K_{XX}\bigr),
   \qquad (K_{XX})_{ij} = k(x_i, x_j).

Regression posterior
--------------------

Given noisy observations :math:`y = f(X) + \varepsilon` with
:math:`\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)`, the predictive
distribution at a test point :math:`x_*` is also Gaussian:

.. math::

   p(f_* \mid X, y, x_*) = \mathcal{N}(\mu_*, \sigma_*^2),

with

.. math::

   \mu_*       &= m(x_*) + K_{*X}\, (K_{XX} + \sigma_n^2 I)^{-1} (y - m(X)), \\
   \sigma_*^2  &= k(x_*, x_*) - K_{*X}\, (K_{XX} + \sigma_n^2 I)^{-1} K_{X*}.

In LightGP, the matrix :math:`K_y := K_{XX} + \sigma_n^2 I` is factorized
once at ``fit()`` time (Cholesky, CG, or SKI depending on the solver); the
factor is then reused for every test point at ``predict()`` time.

Log marginal likelihood
-----------------------

To learn hyperparameters :math:`\theta` (length scale, signal variance,
noise variance), maximize the log marginal likelihood:

.. math::

   \log p(y \mid X, \theta) =
       -\tfrac{1}{2}\, y^\top K_y^{-1} y
       - \tfrac{1}{2}\, \log\lvert K_y \rvert
       - \tfrac{N}{2}\, \log(2\pi).

The three terms have intuitive interpretations:

* :math:`-\tfrac{1}{2}\, y^\top K_y^{-1} y` — **data fit**. Penalizes
  prediction errors weighted by the inverse covariance.
* :math:`-\tfrac{1}{2}\, \log\lvert K_y \rvert` — **complexity penalty**.
  Discourages overfitting by penalizing models that allow too many
  functions.
* :math:`-\tfrac{N}{2}\, \log(2\pi)` — constant, drops in optimization.

LightGP optimizes :math:`\log p` via Adam over the log-hyperparameters
:math:`[\log \ell, \log \sigma^2, \log \sigma_n^2]`. The legacy API uses
analytical gradients; the new ``Kernel``-object API uses central
finite differences so it supports arbitrary kernel compositions.

Why O(N³)?
----------

The cost of an exact GP fit is dominated by the Cholesky factorization of
:math:`K_y`, which is :math:`O(N^3)`. Predict variance also needs a
forward solve, :math:`O(N^2 M)` for :math:`M` test points.

In practice, exact GP fits comfortably for :math:`N \lesssim 5{,}000` on a
laptop. The next three pages describe how LightGP scales past that:

* :doc:`sparse_gp` — approximate :math:`K_y` with :math:`M \ll N` inducing
  points
* :doc:`ski` — when :math:`D \le 3`, replace :math:`K_y` with a
  Toeplitz-structured grid kernel and use FFTs
* :doc:`matrix_free` — solve the linear system iteratively without ever
  forming :math:`K_y` explicitly
