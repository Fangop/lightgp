# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""Sparse GP smoke tests via the Python bindings.

The Python GPSparse currently uses the legacy GPHyperparams API
(constructor takes hp params directly, not a Kernel object — the
Kernel-object refactor for GPSparse is deferred to next-moves).
"""
import numpy as np
import pytest

import lightgp as gp


def test_sparse_basic(sin_data):
    X, y, X_test = sin_data
    model = gp.GPSparse(length_scale=1.0, signal_var=1.0, noise_var=0.05)
    model.fit(X, y, num_inducing=20)
    pred = model.predict(X_test)
    assert pred["mean"].shape == (50,)
    assert np.all(pred["var"] >= 0)


def test_sparse_at_scale():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5000, 4)).astype(np.float32)
    y = np.sin(X[:, 0]).astype(np.float32)
    X_test = rng.standard_normal((100, 4)).astype(np.float32)
    model = gp.GPSparse(noise_var=0.1)
    model.fit(X, y, num_inducing=100)
    pred = model.predict(X_test)
    assert pred["mean"].shape == (100,)
    assert pred["var"].shape == (100,)


def test_sparse_log_ml_finite(sin_data):
    X, y, _ = sin_data
    model = gp.GPSparse()
    model.fit(X, y, num_inducing=20)
    ll = model.log_marginal_likelihood()
    assert np.isfinite(ll)
