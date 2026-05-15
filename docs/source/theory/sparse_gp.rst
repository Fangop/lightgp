Sparse GP (Titsias VFE)
=======================

A sparse GP replaces the full :math:`N \times N` kernel matrix
:math:`K_{ff} = K_{XX}` with an approximation involving :math:`M \ll N`
**inducing points** :math:`Z = (z_1, \dots, z_M)`.

Titsias (2009) derives the **variational free energy** (VFE) bound by
introducing auxiliary inducing-function values :math:`u = f(Z)` and
maximizing a variational lower bound on :math:`\log p(y)`.

Kernel approximation
--------------------

Define the cross-covariances

.. math::

   K_{uu} = K(Z, Z) \in \mathbb{R}^{M \times M}, \qquad
   K_{uf} = K(Z, X) \in \mathbb{R}^{M \times N}.

The full :math:`K_{ff}` is approximated by

.. math::

   Q_{ff} := K_{fu}\, K_{uu}^{-1}\, K_{uf}.

In effect, the data covariance is projected through the :math:`M`-dim
inducing-point subspace.

VFE bound
---------

With per-data diagonal residual

.. math::

   \Lambda = \operatorname{diag}\!\bigl(K_{ff} - Q_{ff}\bigr) + \sigma_n^2 I,

and the :math:`M \times M` "Sigma" matrix

.. math::

   \Sigma = K_{uu} + K_{uf}\, \Lambda^{-1}\, K_{fu},

the variational lower bound is

.. math::

   \mathcal{L}(\theta, Z) =
       -\tfrac{1}{2}\, y^\top (Q_{ff} + \Lambda)^{-1} y
       - \tfrac{1}{2}\, \log\lvert Q_{ff} + \Lambda \rvert
       - \tfrac{N}{2}\, \log(2\pi)
       - \tfrac{1}{2\sigma_n^2}\, \operatorname{tr}\!\bigl(K_{ff} - Q_{ff}\bigr).

The last term is the **regularization** that distinguishes VFE from the
older FITC approximation — it penalizes inducing-point placements that
leave residual variance on the training data.

Cost
----

Storage is :math:`O(N + M^2)`. Training cost per fit is
:math:`O(NM^2 + M^3)` — linear in :math:`N`. Predict mean is
:math:`O(M)` per test point; predict variance is :math:`O(M^2)`.

LightGP's ``GPSparse`` initializes inducing locations by **farthest-point
sampling** on the training inputs. This is deterministic, requires no
random seed, and gives strong defaults on real data:

.. code-block:: python

   import lightgp as gp

   model = gp.GPSparse(noise_var=0.1)
   model.fit(X, y, num_inducing=100)
   pred = model.predict(X_test)

When to use a sparse GP
-----------------------

The decision tree from the user docs:

* :math:`N \le 5{,}000`: exact Cholesky is usually fine
* :math:`5{,}000 \lesssim N \lesssim 50{,}000`, **D ≤ 3**: try
  :doc:`ski`
* :math:`N \gtrsim 5{,}000`, **D > 3**: sparse VFE with
  :math:`M \approx \sqrt{N}` is a strong starting point

For real datasets, a useful sanity check is to plot inducing-point
locations against the data:

.. code-block:: python

   import matplotlib.pyplot as plt
   plt.scatter(X[:, 0], y, alpha=0.2, s=4)
   plt.scatter(model.Z[:, 0], np.zeros(len(model.Z)), color="red",
               marker="x", s=50)

Inducing points should be roughly uniform over the input support — if
they cluster, your length scale is mis-calibrated and you should
``model.optimize()`` first.
