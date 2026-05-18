# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""lightgp — Lightweight Gaussian Process inference.

C++/Metal/Accelerate under the hood. No PyTorch. No TensorFlow. Just numpy.

Example:
    >>> import numpy as np
    >>> import lightgp as gp
    >>> X = np.random.randn(50, 1).astype(np.float32)
    >>> y = np.sin(X[:, 0]).astype(np.float32)
    >>> model = gp.GPExact(gp.RBF())
    >>> model.fit(X, y)
    >>> model.optimize(steps=50)
    >>> pred = model.predict(X)
    >>> pred['mean'].shape, pred['var'].shape
    ((50,), (50,))

Kernel composition:
    >>> kernel = gp.Scale(gp.RBF()) + gp.Scale(gp.Periodic(period=1.0))
    >>> model = gp.GPExact(kernel, mean=gp.LinearMean(input_dim=1))
"""

from lightgp._core import (
    Backend,
    Solver,
    Kernel,
    RBF,
    Matern,
    Periodic,
    Linear,
    Scale,
    SumKernel,
    ProductKernel,
    Mean,
    ZeroMean,
    ConstantMean,
    LinearMean,
    GPExact,
    GPSparse,
    GPSparseHyperparams,
)

__version__ = "0.1.1"

__all__ = [
    "Backend",
    "Solver",
    "Kernel",
    "RBF",
    "Matern",
    "Periodic",
    "Linear",
    "Scale",
    "SumKernel",
    "ProductKernel",
    "Mean",
    "ZeroMean",
    "ConstantMean",
    "LinearMean",
    "GPExact",
    "GPSparse",
    "GPSparseHyperparams",
]
