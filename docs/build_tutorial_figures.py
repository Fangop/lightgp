#!/usr/bin/env python3
"""Generate every PNG embedded in the LightGP tutorials.

Run once from the repo root after a successful Python build:

    source .venv/bin/activate
    PYTHONPATH=python python3 docs/build_tutorial_figures.py

Outputs land in docs/source/_static/figures/. Committed to git so the docs
build does not need matplotlib or the lightgp module.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the locally built lightgp importable.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
import lightgp as gp

OUT = os.path.join(REPO, "docs", "source", "_static", "figures")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
BLUE = "#2563EB"
GREY = "#6B7280"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.relpath(path, REPO)}")


def gp_band(ax, X, mu, var, color=BLUE, label="GP mean"):
    sd = np.sqrt(np.maximum(var, 0.0))
    ax.fill_between(X[:, 0], mu - 2 * sd, mu + 2 * sd, alpha=0.18, color=color)
    ax.plot(X[:, 0], mu, lw=2, color=color, label=label)


# ============================================================
# Tutorial 01: Basic regression
# ============================================================

def fig_basic_regression():
    print("[01] basic regression")
    X = np.sort(RNG.uniform(-3, 3, 40)).reshape(-1, 1).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.15 * RNG.standard_normal(40)).astype(np.float32)
    Xt = np.linspace(-4, 4, 250).reshape(-1, 1).astype(np.float32)

    # Before optimization
    m1 = gp.GPExact(gp.RBF(length_scale=0.3), noise_var=0.1)
    m1.fit(X, y)
    p1 = m1.predict(Xt)

    # After optimization
    m2 = gp.GPExact(gp.RBF(), noise_var=0.1)
    m2.fit(X, y)
    m2.optimize(steps=80)
    p2 = m2.predict(Xt)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, p, title in [(a, p1, "Before optimize()"), (b, p2, "After optimize()")]:
        ax.scatter(X[:, 0], y, c="black", s=18, zorder=3)
        gp_band(ax, Xt, p["mean"], p["var"])
        ax.plot(Xt[:, 0], np.sin(Xt[:, 0]), color=GREY, ls="--", lw=1, label="true f")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    a.set_ylabel("y")
    save(fig, "01_optimize_before_after.png")

    # Kernel comparison
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(X[:, 0], y, c="black", s=18, zorder=3, label="data")
    for nu, color in [(0.5, "#F59E0B"), (1.5, "#10B981"), (2.5, BLUE)]:
        m = gp.GPExact(gp.Matern(nu=nu), noise_var=0.1)
        m.fit(X, y)
        m.optimize(steps=60)
        mu = m.predict(Xt)["mean"]
        ax.plot(Xt[:, 0], mu, lw=2, color=color, label=f"Matérn-ν={nu}")
    ax.set_title("Smoothness via Matérn ν")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    save(fig, "01_matern_smoothness.png")


# ============================================================
# Tutorial 02: Kernel composition (Mauna Loa-like)
# ============================================================

def fig_kernel_composition():
    print("[02] kernel composition")
    # Synthetic "Mauna Loa": linear trend + annual seasonal + small noise.
    N = 300
    t = np.linspace(0, 30, N, dtype=np.float32).reshape(-1, 1)  # 30 years
    y = (1.5 * t[:, 0] + 0.04 * t[:, 0] ** 2
         + 3.0 * np.sin(2 * np.pi * t[:, 0])
         + 0.5 * RNG.standard_normal(N)).astype(np.float32)
    Xt = np.linspace(0, 35, 400).reshape(-1, 1).astype(np.float32)  # extrapolate

    # Single RBF baseline.
    m_rbf = gp.GPExact(gp.RBF(length_scale=2.0, signal_var=10.0),
                       mean=gp.LinearMean(input_dim=1), noise_var=0.5)
    m_rbf.fit(t, y)
    m_rbf.optimize(steps=80)
    p_rbf = m_rbf.predict(Xt)

    # Composed kernel.
    kernel = (gp.Scale(gp.RBF(length_scale=10.0))
              + gp.Scale(gp.Periodic(length_scale=1.0, period=1.0))
              + gp.Scale(gp.RBF(length_scale=0.5)))
    m_comp = gp.GPExact(kernel, mean=gp.LinearMean(input_dim=1), noise_var=0.5)
    m_comp.fit(t, y)
    m_comp.optimize(steps=120)
    p_comp = m_comp.predict(Xt)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, p, title in [(a, p_rbf, "Single RBF"),
                          (b, p_comp, "Scale(RBF) + Scale(Periodic) + Scale(RBF)")]:
        ax.scatter(t[:, 0], y, c="black", s=8, zorder=3, label="training data")
        gp_band(ax, Xt, p["mean"], p["var"])
        ax.axvline(30, color=GREY, lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("time (years)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    a.set_ylabel("y")
    save(fig, "02_composition_vs_single.png")


# ============================================================
# Tutorial 03: Sparse GP
# ============================================================

def fig_sparse_gp():
    print("[03] sparse GP")
    N = 1500
    X = np.sort(RNG.uniform(-5, 5, N)).reshape(-1, 1).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.4 * np.cos(2 * X[:, 0])
         + 0.15 * RNG.standard_normal(N)).astype(np.float32)
    Xt = np.linspace(-5, 5, 400).reshape(-1, 1).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, M in zip(axes, [10, 30, 100]):
        m = gp.GPSparse(length_scale=1.0, signal_var=1.0, noise_var=0.03)
        m.fit(X, y, num_inducing=M)
        p = m.predict(Xt)
        ax.scatter(X[:, 0], y, c="lightgray", s=4, zorder=1)
        gp_band(ax, Xt, p["mean"], p["var"])
        ax.set_title(f"M = {M} inducing points")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    save(fig, "03_sparse_inducing_sweep.png")


# ============================================================
# Tutorial 04: SKI scaling
# ============================================================

def fig_ski_scaling():
    print("[04] SKI scaling")
    Ns = [2_000, 10_000, 50_000]
    chol_ms = []
    ski_ms = []
    for N in Ns:
        X = np.linspace(-5, 5, N, dtype=np.float32).reshape(-1, 1)
        y = (np.sin(X[:, 0])
             + 0.1 * RNG.standard_normal(N)).astype(np.float32)

        if N <= 4000:
            m = gp.GPExact(gp.RBF(), noise_var=0.05, solver=gp.Solver.Cholesky)
            t0 = time.perf_counter()
            m.fit(X, y)
            chol_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            chol_ms.append(None)

        m_ski = gp.GPExact(gp.RBF(), noise_var=0.05,
                           solver=gp.Solver.SKI, backend=gp.Backend.CPU)
        t0 = time.perf_counter()
        m_ski.fit(X, y)
        ski_ms.append((time.perf_counter() - t0) * 1000.0)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot([n for n, t in zip(Ns, chol_ms) if t is not None],
            [t for t in chol_ms if t is not None],
            "o-", color=BLUE, lw=2, label="Solver.Cholesky")
    ax.plot(Ns, ski_ms, "s-", color="#10B981", lw=2, label="Solver.SKI (vDSP)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (training points)"); ax.set_ylabel("fit time (ms, log)")
    ax.set_title("Fit time: Cholesky vs SKI")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    save(fig, "04_ski_scaling.png")


# ============================================================
# Tutorial 05: Backend comparison (mini-benchmark)
# ============================================================

def fig_backends():
    print("[05] backends")
    # Quick wall-time comparison at moderate N for several backends, if available.
    N = 1500
    X = RNG.standard_normal((N, 4)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * RNG.standard_normal(N)).astype(np.float32)

    results = {}
    for name, backend in [("CPU", gp.Backend.CPU),
                          ("Metal", gp.Backend.Metal),
                          ("Auto", gp.Backend.Auto)]:
        try:
            m = gp.GPExact(gp.RBF(), backend=backend, noise_var=0.05)
            # warmup
            m.fit(X, y); m.predict(X[:16])
            t0 = time.perf_counter()
            m.fit(X, y); m.predict(X[:16])
            results[name] = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"  skipped {name}: {e}")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(results.keys()); times = list(results.values())
    ax.bar(names, times, color=[BLUE, "#10B981", "#F59E0B"][: len(names)])
    ax.set_ylabel("fit + predict (ms)")
    ax.set_title(f"Backends at N={N}, D=4 (Apple M4)")
    for x, t in zip(names, times):
        ax.text(x, t, f"{t:.1f}", ha="center", va="bottom", fontsize=9)
    save(fig, "05_backends_bar.png")


if __name__ == "__main__":
    fig_basic_regression()
    fig_kernel_composition()
    fig_sparse_gp()
    fig_ski_scaling()
    fig_backends()
    print("\nAll figures generated.")
