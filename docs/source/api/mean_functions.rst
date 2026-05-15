Mean functions
==============

Mean functions are subtracted from ``y`` before fitting and added back at
predict time. Defaults to :class:`ZeroMean`.

.. py:class:: lightgp.Mean

   Abstract base for mean functions.

   .. py:method:: name() -> str
   .. py:method:: num_params() -> int

.. py:class:: lightgp.ZeroMean

   Returns the zero vector. Has no parameters. The default.

.. py:class:: lightgp.ConstantMean(c: float = 0.0)

   :math:`m(x) = c`. One scalar parameter.

   .. code-block:: python

      # Useful when targets have a known nonzero offset.
      model = gp.GPExact(gp.RBF(), mean=gp.ConstantMean())

.. py:class:: lightgp.LinearMean(input_dim: int)

   :math:`m(x) = w^\top x + b`. ``input_dim + 1`` parameters.

   .. code-block:: python

      # Useful for data with a linear trend (e.g. Mauna Loa CO2).
      model = gp.GPExact(gp.RBF(), mean=gp.LinearMean(input_dim=1))
