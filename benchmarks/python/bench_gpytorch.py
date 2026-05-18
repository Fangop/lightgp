# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""GPyTorch comparison benchmark for the lightgp paper.

Matches the (N, D, M) grids used by the C++ benches and emits JSON to stdout.

Run:
    pip install torch gpytorch
    python3 benchmarks/python/bench_gpytorch.py > gpytorch_results.json

Each emitted record has the form:
    {"method": str, "device": str, "N": int, "D": int, "M": int or null,
     "fit_ms": float, "predict_ms": float, "total_ms": float, "runs": int,
     "torch_version": str, "gpytorch_version": str, "notes": str}

`device` ∈ {"cpu", "mps"}. `M` is null for non-sparse methods.
Median over 5 runs, warmup discarded.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable

import torch
import gpytorch


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _median_ms(f: Callable[[], None], runs: int = 5) -> tuple[float, float]:
    """Run f once for warmup, then `runs` more times. Return (median_ms, total_ms)."""
    f()  # warmup
    samples = []
    t_total_start = _now_ms()
    for _ in range(runs):
        t0 = _now_ms()
        f()
        samples.append(_now_ms() - t0)
    t_total = _now_ms() - t_total_start
    return statistics.median(samples), t_total


def _make_synthetic(N: int, D: int, device: torch.device, seed: int = 0):
    """y(x) = sin(sum(x)) + small noise. Matches the C++ bench_gp_e2e signal."""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(N, D, generator=g, device=device)
    y = torch.sin(X.sum(dim=-1)) + 0.01 * torch.randn(N, generator=g, device=device)
    return X, y


# -----------------------------------------------------------------------------
# Exact GP — Cholesky
# -----------------------------------------------------------------------------
class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel: str = "rbf"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel == "matern52":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            raise ValueError(kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def bench_exact(N: int, D: int, device: torch.device, kernel: str = "rbf", runs: int = 5):
    X, y = _make_synthetic(N, D, device)
    Xt, _ = _make_synthetic(max(1, N // 4), D, device, seed=999)

    def fit_predict():
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = _ExactGP(X, y, likelihood, kernel=kernel).to(device)
        # Single likelihood-fit step (we don't bench optimization here — match the C++ bench_gp_e2e setup).
        model.train(); likelihood.train()
        out = model(X)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        _ = -mll(out, y)
        # Predict
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(Xt))
            _ = pred.mean
            _ = pred.variance
        # MPS is lazy — force a sync.
        if device.type == "mps":
            torch.mps.synchronize()

    median_ms, _ = _median_ms(fit_predict, runs=runs)
    return median_ms


# -----------------------------------------------------------------------------
# Sparse GP — SGPR (Titsias VFE)
# -----------------------------------------------------------------------------
class _SparseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base, inducing_points=inducing, likelihood=likelihood
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def bench_sparse(N: int, D: int, M: int, device: torch.device, runs: int = 3):
    X, y = _make_synthetic(N, D, device)
    Xt, _ = _make_synthetic(64, D, device, seed=999)

    def fit_predict():
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        # Inducing points by random subsample (GPyTorch standard).
        idx = torch.randperm(N, device=device)[:M]
        inducing = X[idx].clone()
        model = _SparseGP(X, y, likelihood, inducing).to(device)
        model.train(); likelihood.train()
        out = model(X)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        _ = -mll(out, y)
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(Xt))
            _ = pred.mean
            _ = pred.variance
        if device.type == "mps":
            torch.mps.synchronize()

    median_ms, _ = _median_ms(fit_predict, runs=runs)
    return median_ms


# -----------------------------------------------------------------------------
# Bench grids — keep in sync with bench_paper.cpp
# -----------------------------------------------------------------------------
EXACT_GRID = [(N, D) for D in (4, 16, 64) for N in (256, 512, 1024, 2048)]
SPARSE_GRID = [
    (1000, 50), (1000, 100),
    (5000, 100), (5000, 200),
    (10000, 100), (10000, 200),
    (50000, 200),
]


def run_all(devices: list[str], runs: int, max_exact_N: int):
    torch_ver = torch.__version__
    gp_ver = gpytorch.__version__
    out = []
    for dev_name in devices:
        device = torch.device(dev_name)
        for N, D in EXACT_GRID:
            if N > max_exact_N:
                continue
            for kernel in ("rbf", "matern52"):
                try:
                    ms = bench_exact(N, D, device, kernel=kernel, runs=runs)
                except Exception as e:
                    print(f"# skip exact {kernel} N={N} D={D} on {dev_name}: {e}", file=sys.stderr)
                    continue
                out.append({
                    "method": f"exact_{kernel}", "device": dev_name,
                    "N": N, "D": D, "M": None,
                    "total_ms": ms, "runs": runs,
                    "torch_version": torch_ver, "gpytorch_version": gp_ver,
                    "notes": "fit + predict, no hyperparameter optimization",
                })
                print(json.dumps(out[-1]), flush=True)

        for N, M in SPARSE_GRID:
            try:
                ms = bench_sparse(N, 4, M, device, runs=max(2, runs // 2))
            except Exception as e:
                print(f"# skip sparse N={N} M={M} on {dev_name}: {e}", file=sys.stderr)
                continue
            out.append({
                "method": "sparse_rbf", "device": dev_name,
                "N": N, "D": 4, "M": M,
                "total_ms": ms, "runs": max(2, runs // 2),
                "torch_version": torch_ver, "gpytorch_version": gp_ver,
                "notes": "SGPR with random-subsample inducing init",
            })
            print(json.dumps(out[-1]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"],
                    help='subset of {"cpu", "mps"}')
    ap.add_argument("--threads", type=int, default=1, help="torch CPU thread count")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-exact-N", type=int, default=2048)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    devices = [d for d in args.devices
               if d != "mps" or torch.backends.mps.is_available()]
    if "mps" in args.devices and "mps" not in devices:
        print("# MPS requested but not available; skipping.", file=sys.stderr)

    run_all(devices, runs=args.runs, max_exact_N=args.max_exact_N)


if __name__ == "__main__":
    main()
