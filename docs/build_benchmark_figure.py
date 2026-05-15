#!/usr/bin/env python3
"""Generate the benchmarks gallery scaling figure.

Wall-time per fit vs N, three inference methods on synthetic 1-D sin data.
Mirrors the numbers documented in README.md (Apple M4, fp32) and exercises
the actual lightgp library so the plot is the genuine library output, not
a mockup. Real users can rerun this on their own hardware.

Output: docs/source/_static/figures/scaling.png
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
import lightgp as gp

OUT = os.path.join(REPO, "docs", "source", "_static", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#10B981"
ORANGE = "#F59E0B"
GREY = "#6B7280"


def bench_fit(model_factory, X, y, repeats=3):
    """Run fit `repeats` times and return median wall time in ms."""
    times = []
    for _ in range(repeats):
        m = model_factory()
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main():
    print("benchmarks scaling plot")

    Ns = [500, 2000, 5000, 10_000, 20_000, 50_000]
    chol_ms = []      # Solver.Cholesky, exact, O(N^3); only run up to 4k
    cg_ms = []        # Solver.CG, matrix-free CG; run up to 20k
    ski_ms = []       # Solver.SKI, FFT-accelerated; all sizes
    sparse_ms = []    # GPSparse VFE with M=100; all sizes

    for N in Ns:
        X = np.linspace(-5.0, 5.0, N, dtype=np.float32).reshape(-1, 1)
        y = (np.sin(X[:, 0]) + 0.1 * np.random.default_rng(0).standard_normal(N)).astype(
            np.float32
        )
        print(f"  N={N}:")

        if N <= 4000:
            t = bench_fit(
                lambda: gp.GPExact(gp.RBF(), noise_var=0.05, solver=gp.Solver.Cholesky), X, y
            )
            chol_ms.append(t); print(f"    Cholesky: {t:.1f} ms")
        else:
            chol_ms.append(None)

        if N <= 20_000:
            t = bench_fit(
                lambda: gp.GPExact(
                    gp.RBF(), noise_var=0.05, solver=gp.Solver.CG, backend=gp.Backend.CPU
                ),
                X, y, repeats=2,
            )
            cg_ms.append(t); print(f"    CG: {t:.1f} ms")
        else:
            cg_ms.append(None)

        t = bench_fit(
            lambda: gp.GPExact(
                gp.RBF(), noise_var=0.05, solver=gp.Solver.SKI, backend=gp.Backend.CPU
            ),
            X, y, repeats=2,
        )
        ski_ms.append(t); print(f"    SKI: {t:.1f} ms")

        # Sparse: fit() takes num_inducing positionally; wrap fit in factory.
        def make_sparse():
            return gp.GPSparse(noise_var=0.05)
        repeats = 2
        times = []
        for _ in range(repeats):
            m = make_sparse()
            t0 = time.perf_counter()
            m.fit(X, y, num_inducing=100)
            times.append((time.perf_counter() - t0) * 1000.0)
        t = float(np.median(times)); sparse_ms.append(t); print(f"    Sparse M=100: {t:.1f} ms")

    # Plot.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    def line(times, color, marker, label):
        xs = [n for n, t in zip(Ns, times) if t is not None]
        ys = [t for t in times if t is not None]
        if xs:
            ax.plot(xs, ys, marker + "-", color=color, lw=2, ms=7, label=label)

    line(chol_ms, BLUE, "o", "Solver.Cholesky (exact)")
    line(cg_ms, GREEN, "s", "Solver.CG (matrix-free)")
    line(ski_ms, ORANGE, "^", "Solver.SKI (FFT)")
    line(sparse_ms, "#7C3AED", "D", "GPSparse (M=100)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training points N")
    ax.set_ylabel("fit time (ms, log scale)")
    ax.set_title("LightGP inference scaling on Apple M4 (CPU, fp32)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(OUT, "scaling.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
