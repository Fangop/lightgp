SKI (KISS-GP)
=============

Structured Kernel Interpolation (Wilson & Nickisch, 2015) takes the sparse
GP idea further: place inducing points on a **regular grid** so the
inducing-inducing kernel matrix :math:`K_{uu}` becomes
**Toeplitz-structured** (in 1D) or **Kronecker-Toeplitz** (in 2D / 3D).
Matrix-vector products against Toeplitz matrices can then be done with
**FFTs** in :math:`O(M \log M)`.

The approximation
-----------------

Let :math:`Z` be a regular grid of size :math:`M` covering the input
range. Define :math:`W \in \mathbb{R}^{N \times M}`, a **sparse cubic
interpolation matrix** that interpolates each training point onto its
4 (or 16, in 2D) nearest grid neighbors. Then:

.. math::

   K_{ff} \approx W\, K_{uu}\, W^\top.

Each row of :math:`W` has 4 nonzero entries (in 1D), so :math:`W` is
sparse with :math:`4N` nonzeros total. The product :math:`W \cdot v`
costs :math:`O(N)`.

Fast matrix-vector products
---------------------------

The key trick: a Toeplitz matrix :math:`T` embeds into a circulant matrix
:math:`C` of double the size, and **circulant matrix-vector products are
diagonalized by the FFT**. Concretely:

.. math::

   T v = \operatorname{first\;half\;of}\bigl(\,
        \mathcal{F}^{-1}\!\bigl(\mathcal{F}(c) \odot \mathcal{F}(\tilde v)\bigr)
        \,\bigr)

where :math:`c` is the first column of :math:`C` and :math:`\tilde v` is
:math:`v` zero-padded.

Putting it together, the SKI matrix-vector product for a test vector
:math:`v` is:

.. math::

   K_{ff}\,v \;\approx\; W\,\bigl(K_{uu}\,\bigl(W^\top v\bigr)\bigr)

with the middle :math:`K_{uu}\,(W^\top v)` evaluated by FFT. Each matvec
costs :math:`O(N + M \log M)`.

Inference
---------

With a fast :math:`K_y \, v`, the inference is a standard conjugate-gradient
solve. LightGP uses :math:`O(30)`–:math:`O(50)` CG iterations to converge
to a target residual; total fit cost is :math:`O(k \cdot (N + M \log M))`.

Variance estimation in SKI mode uses Hutchinson probes through the same
CG-with-FFT-matvec pipeline.

Where the FFT happens
---------------------

LightGP dispatches the inner FFT by backend:

* ``Backend.CUDA`` → cuFFT
* ``Backend.CPU`` on macOS → Apple Accelerate ``vDSP_DFT_Execute``
* ``Backend.CPU`` elsewhere → reference dense Toeplitz (no FFT yet)

``Backend.Auto`` correctly resolves to CUDA when available and to CPU
otherwise. On Apple Silicon, the vDSP backend is fast enough that SKI
reaches the same regime that needs CUDA elsewhere.

When SKI works (and when it doesn't)
------------------------------------

**SKI is the right tool when**:

* :math:`D \le 3` (the grid has :math:`M^D` cells)
* The data is stationary or near-stationary
* You want O(M log M) matvecs at very large N

**SKI is not the right tool when**:

* :math:`D > 3`: grid size :math:`M^D` makes the FFT too expensive
* Inputs are irregularly distributed in regions of very different density:
  the grid wastes coverage
* You're using a non-stationary kernel

For high-dimensional problems, prefer :doc:`sparse_gp` with farthest-point
or k-means inducing initialization.
