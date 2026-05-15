import numpy as np

import lightgp as gp


def test_solver_ski_exposed():
    assert hasattr(gp.Solver, "SKI")


def _pick_backend():
    """Prefer CUDA when the binding was built with CUDA. CPU is a slow fallback for SKI."""
    return getattr(gp.Backend, "CUDA", gp.Backend.CPU)


def test_ski_sin_1d():
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 6.0, 1000, dtype=np.float32).reshape(-1, 1)
    y = np.sin(X[:, 0]).astype(np.float32) + 0.02 * rng.standard_normal(1000).astype(np.float32)

    model = gp.GPExact(gp.RBF(), solver=gp.Solver.SKI, backend=_pick_backend(), noise_var=0.01)
    model.fit(X, y)
    pred = model.predict(X[:50])
    mean = pred["mean"]
    assert mean.shape == (50,)
    rmse = float(np.sqrt(np.mean((mean - np.sin(X[:50, 0])) ** 2)))
    assert rmse < 0.3, f"RMSE too large: {rmse:.3f}"


def test_ski_large_n():
    # 50k points; with SKI on CUDA this should fit and predict in well under a few seconds.
    # Test runs even on the CPU build, but SKI CPU at N=50k is O(M^2) per matvec and very slow,
    # so we cap N when the CUDA backend isn't available.
    backend = _pick_backend()
    N = 50000 if backend == gp.Backend.CUDA else 5000
    X = np.linspace(0.0, 10.0, N, dtype=np.float32).reshape(-1, 1)
    y = np.sin(X[:, 0]).astype(np.float32)
    model = gp.GPExact(gp.RBF(), solver=gp.Solver.SKI, backend=backend, noise_var=0.01)
    model.fit(X, y)
    pred = model.predict(X[:100])
    assert pred["mean"].shape == (100,)
    # log marginal likelihood is finite even though it's an SLQ estimate.
    assert np.isfinite(model.log_marginal_likelihood())
