"""End-to-end tests for GPExact via the Python bindings."""
import numpy as np
import pytest

import lightgp as gp


def test_fit_predict_shape(sin_data):
    X, y, X_test = sin_data
    model = gp.GPExact(gp.RBF())
    model.fit(X, y)
    pred = model.predict(X_test)
    assert pred["mean"].shape == (50,)
    assert pred["var"].shape == (50,)
    assert np.all(pred["var"] >= 0)


def test_predict_dtype_float32(sin_data):
    X, y, X_test = sin_data
    model = gp.GPExact(gp.RBF())
    model.fit(X, y)
    pred = model.predict(X_test)
    assert pred["mean"].dtype == np.float32
    assert pred["var"].dtype == np.float32


def test_fitted_flag(sin_data):
    X, y, _ = sin_data
    model = gp.GPExact(gp.RBF())
    assert not model.fitted()
    model.fit(X, y)
    assert model.fitted()


def test_optimize_improves_ll(sin_data):
    X, y, _ = sin_data
    model = gp.GPExact(gp.RBF(), noise_var=0.1)
    model.fit(X, y)
    ll_before = model.log_marginal_likelihood()
    model.optimize(steps=10)
    ll_after = model.log_marginal_likelihood()
    assert ll_after >= ll_before - 0.5  # finite-diff Adam, allow tiny dip


def test_sin_rmse(sin_data):
    X, y, X_test = sin_data
    model = gp.GPExact(gp.RBF(), noise_var=0.01)
    model.fit(X, y)
    model.optimize(steps=20)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred["mean"] - np.sin(X_test[:, 0])) ** 2)))
    assert rmse < 0.2


def test_composed_kernel(seasonal_data):
    X, y = seasonal_data
    kernel = gp.Scale(gp.RBF()) + gp.Scale(gp.Periodic(period=1.0))
    model = gp.GPExact(kernel, mean=gp.LinearMean(input_dim=1))
    model.fit(X, y)
    X_test = np.linspace(0, 8, 50, dtype=np.float32).reshape(-1, 1)
    pred = model.predict(X_test)
    assert pred["mean"].shape == (50,)
    assert np.all(pred["var"] >= 0)


def test_constant_mean(sin_data):
    X, y, X_test = sin_data
    model = gp.GPExact(gp.RBF(), mean=gp.ConstantMean())
    model.fit(X, y)
    pred = model.predict(X_test)
    assert pred["mean"].shape == (50,)


def test_matern_variants_fit(sin_data):
    X, y, X_test = sin_data
    for nu in (0.5, 1.5, 2.5):
        model = gp.GPExact(gp.Matern(nu=nu))
        model.fit(X, y)
        pred = model.predict(X_test)
        assert pred["mean"].shape == (50,)


def test_cg_solver(sin_data):
    X, y, X_test = sin_data
    model = gp.GPExact(gp.RBF(), solver=gp.Solver.CG)
    model.fit(X, y)
    pred = model.predict(X_test)
    assert pred["mean"].shape == (50,)


def test_predict_without_fit_raises():
    model = gp.GPExact(gp.RBF())
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 1), dtype=np.float32))
