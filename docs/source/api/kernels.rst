Kernels
=======

All kernels inherit from :class:`lightgp.Kernel` and support composition via
the ``+``, ``*``, and :class:`Scale <lightgp.Scale>` operators.

.. code-block:: python

   kernel = gp.Scale(gp.RBF()) + gp.Scale(gp.Periodic())

Base class
----------

.. py:class:: lightgp.Kernel

   Abstract base for all kernels. Subclasses implement :py:meth:`compute`,
   :py:meth:`compute_diag`, parameter access, and a printable :py:meth:`name`.

   .. py:method:: name() -> str

      Human-readable kernel name (used in repr and composite-kernel display).

   .. py:method:: num_params() -> int

      Number of learnable log-hyperparameters.

   .. py:method:: get_params() -> list[float]

      Flat list of log-hyperparameters. Order is fixed per-kernel and matches
      :py:meth:`set_params`.

   .. py:method:: set_params(params: list[float]) -> None

      Overwrite log-hyperparameters from a flat list.

   .. py:method:: __add__(other: Kernel) -> Kernel

      Return ``SumKernel(self, other)``. The result's parameter list is the
      concatenation of both.

   .. py:method:: __mul__(other: Kernel) -> Kernel

      Return ``ProductKernel(self, other)`` (elementwise multiply of
      kernel matrices).

Concrete kernels
----------------

.. py:class:: lightgp.RBF(length_scale: float = 1.0, signal_var: float = 1.0)

   Radial Basis Function (squared exponential) kernel.

   .. math::

      k(x, x') = \sigma^2 \exp\!\left(-\tfrac{\|x - x'\|^2}{2\ell^2}\right)

   :param length_scale: Length scale :math:`\ell`. Larger values produce
       smoother fits.
   :param signal_var: Signal variance :math:`\sigma^2`. Controls the prior
       output amplitude.

.. py:class:: lightgp.Matern(nu: float = 2.5, length_scale: float = 1.0, signal_var: float = 1.0)

   Matérn kernel with smoothness parameter :math:`\nu`.

   .. math::

      k_{5/2}(r) = \sigma^2 \left(1 + \tfrac{\sqrt{5}\,r}{\ell}
                      + \tfrac{5 r^2}{3 \ell^2}\right)\,
                  \exp\!\left(-\tfrac{\sqrt{5}\,r}{\ell}\right)

   :param nu: Smoothness; one of ``0.5`` (rough), ``1.5``
       (once-differentiable), or ``2.5`` (twice-differentiable).
   :param length_scale: As for :class:`RBF`.
   :param signal_var: As for :class:`RBF`.

.. py:class:: lightgp.Periodic(length_scale: float = 1.0, period: float = 1.0, signal_var: float = 1.0)

   Periodic kernel for cyclical signals.

   .. math::

      k(x, x') = \sigma^2 \exp\!\left(
          -\tfrac{2 \sin^2(\pi |x-x'| / p)}{\ell^2}\right)

   :param period: Period ``p`` of the pattern.
   :param length_scale: Smoothness within one period.

.. py:class:: lightgp.Linear(signal_var: float = 1.0, input_dim: int = 1, offset: float = 0.0)

   Linear (dot-product) kernel. Equivalent to Bayesian linear regression
   when used alone.

   .. math::

      k(x, x') = \sigma^2 (x - c)^\top (x' - c)

   :param offset: Vector offset ``c`` (broadcast across dimensions).

Composition
-----------

.. py:class:: lightgp.Scale(base: Kernel, scale: float = 1.0)

   Wraps a kernel with a learnable scalar output scale ``exp(log_scale)``.
   Adds one log-hyperparameter to the parameter list (prepended in front of
   the base kernel's parameters).

   .. math::

      k_{\text{scaled}}(x, x') = \exp(\text{log\_scale}) \cdot k_{\text{base}}(x, x')

.. py:class:: lightgp.SumKernel(a: Kernel, b: Kernel)

   :math:`k(x, x') = k_a(x, x') + k_b(x, x')`. Created by ``a + b``.

.. py:class:: lightgp.ProductKernel(a: Kernel, b: Kernel)

   :math:`k(x, x') = k_a(x, x') \cdot k_b(x, x')` (elementwise on the
   kernel matrix). Created by ``a * b``.

Example
-------

A composed kernel for the Mauna Loa CO₂ benchmark (multi-decade rising trend
+ annual seasonal cycle + medium-term irregularities):

.. code-block:: python

   trend = gp.Scale(gp.RBF(length_scale=10.0))
   seasonal = gp.Scale(gp.Periodic(period=1.0))
   medium = gp.Scale(gp.RBF(length_scale=0.5))
   kernel = trend + seasonal + medium
   print(kernel.name())
   # ((Scale(RBF) + Scale(Periodic)) + Scale(RBF))
   print(kernel.num_params())   # 9
