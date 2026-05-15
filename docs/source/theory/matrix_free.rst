Matrix-Free Conjugate Gradients
===============================

The GP posterior mean and variance both require solving
:math:`(K + \sigma_n^2 I)\, \alpha = y`. The textbook approach forms the
:math:`N \times N` matrix :math:`K_y` and Cholesky-factorizes it, costing
:math:`O(N^3)` time and :math:`O(N^2)` memory.

Conjugate gradients is an alternative that **never forms** :math:`K_y`
explicitly — it only needs a function that returns :math:`K_y\, v` for
arbitrary :math:`v`.

The CG algorithm
----------------

Starting from :math:`x_0 = 0` and :math:`r_0 = y`, repeat for
:math:`k = 0, 1, \dots`:

.. math::

   \alpha_k    &= \frac{r_k^\top r_k}{p_k^\top (K_y p_k)} \\
   x_{k+1}     &= x_k + \alpha_k p_k \\
   r_{k+1}     &= r_k - \alpha_k (K_y p_k) \\
   \beta_k     &= \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k} \\
   p_{k+1}     &= r_{k+1} + \beta_k p_k

Each iteration performs **one** matrix-vector product :math:`K_y p_k` plus
a handful of vector dots and axpys. After :math:`k` iterations the
residual :math:`\lVert r_k \rVert` typically drops by a factor of
:math:`\mathcal{O}(\kappa(K_y)^{-1/2})` per step, where :math:`\kappa`
is the condition number.

In LightGP, ``Solver.CG`` runs until :math:`\lVert r_k \rVert / \lVert y \rVert
< 10^{-4}` or 1000 iterations, whichever comes first. Convergence is
typically 30–100 iterations for well-conditioned RBF kernels.

The matrix-free :math:`K_y v` shader
------------------------------------

The trick that makes this fast on Metal is implementing :math:`K_y\, v`
**without materializing** :math:`K_y`. The kernel matrix-vector product

.. math::

   (K_y v)_i = \sum_j k(x_i, x_j)\, v_j + \sigma_n^2\, v_i

can be computed in a single GPU shader pass: each output element :math:`i`
loops over all training points :math:`j`, evaluates the kernel function,
multiplies by :math:`v_j`, and accumulates — all without storing the
intermediate :math:`N \times N` matrix.

The result: memory stays :math:`O(N)` regardless of how large the kernel
matrix would be. At :math:`N = 20{,}000` the explicit kernel matrix is
1.6 GB; the matrix-free path needs ~80 KB for :math:`v` and :math:`w`,
plus the input :math:`X` itself.

LightGP's matrix-free shader lives at
``kernels/metal/rbf_matvec.metal`` and ``kernels/cuda/rbf_matvec.cu``. The
host-side CG loop uses these as the matvec callback:

.. code-block:: cpp

   auto K_matvec = [&](const Tensor& v) {
       return rbf_matvec_metal(X, v, length_scale, signal_var, noise_var);
   };
   cg_solve(K_matvec, y, alpha);

Hutchinson variance estimation
------------------------------

The predictive variance for a test point :math:`x_*` is

.. math::

   \sigma_*^2 = k(x_*, x_*) - K_{*X}\, K_y^{-1} K_{X*}.

The middle term needs another solve. In matrix-free CG mode, LightGP uses
**Hutchinson probes**: sample :math:`p` Rademacher vectors :math:`z_j \in
\{-1, +1\}^N`, solve :math:`K_y\, u_j = z_j` once per probe, and estimate

.. math::

   K_y^{-1}\, K_{*X}^\top \;\approx\;
       \frac{1}{p} \sum_{j=1}^p z_j \odot u_j.

This gives a low-rank approximation of :math:`K_y^{-1}` that we then
contract against :math:`K_{*X}`. The variance estimate is biased low but
the bias shrinks as :math:`O(1/p)`.

Log-marginal-likelihood via SLQ
-------------------------------

The :math:`\log\lvert K_y \rvert` term in the log marginal likelihood can
also be computed without ever factorizing :math:`K_y`, using **stochastic
Lanczos quadrature** (Ubaru et al., 2017): run :math:`k` steps of
Lanczos with random initial vectors, extract the tridiagonal matrix
:math:`T_k`, and approximate

.. math::

   \log\lvert K_y \rvert \approx N \cdot \mathbb{E}_z\!\left[
       z^\top \log(K_y) z
   \right]
   \approx \frac{N}{p}\sum_{j=1}^p z_j^\top\, \operatorname{e}_1^\top
       Q_k^\top \log(T_k)\, Q_k\, \operatorname{e}_1.

The :math:`\log(T_k)` is cheap because :math:`T_k` is small
(:math:`k \times k`, typically :math:`k = 30`). LightGP's
``solvers/cpu/slq_cpu.cpp`` implements this and is shared between CG and
SKI modes.

When CG (and matrix-free) wins
------------------------------

* The kernel matrix would exceed RAM (:math:`N \gtrsim 20{,}000`, GB scale)
* You have a GPU with much higher memory bandwidth than CPU
* Variance estimates can tolerate Hutchinson noise

For dense :math:`N < 5{,}000` problems, exact Cholesky is faster and
gives noiseless variance — use :doc:`gp_basics` directly.
