# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

#!/usr/bin/env python3
"""Build the hero landing-page figure for the LightGP docs.

Produces docs/source/_static/figures/hero_regression.png — a wide-format 1-D
GP regression illustration showing training points, posterior mean, 95%
credible band, the true generating function, and uncertainty growth on
extrapolation.

Run once after a successful Python build:

    source .venv/bin/activate
    PYTHONPATH=python python3 docs/build_landing_figure.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
import lightgp as gp

OUT_DIR = os.path.join(REPO, "docs", "source", "_static", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2563EB"
GREY = "#6B7280"


def build_hero():
    rng = np.random.default_rng(7)

    # Training data: 25 points concentrated in [-3, 3], with a small gap
    # around x=1 so the uncertainty band visibly widens through it.
    X_train = np.concatenate(
        [rng.uniform(-3.0, 0.5, 14), rng.uniform(1.6, 3.0, 11)]
    )
    X_train = np.sort(X_train).reshape(-1, 1).astype(np.float32)
    y_train = (np.sin(X_train[:, 0]) + 0.18 * rng.standard_normal(len(X_train))).astype(
        np.float32
    )

    # Test grid: extend well past training range on both sides.
    X_test = np.linspace(-5.5, 5.5, 400).reshape(-1, 1).astype(np.float32)

    model = gp.GPExact(gp.RBF(), noise_var=0.05)
    model.fit(X_train, y_train)
    model.optimize(steps=80)
    pred = model.predict(X_test)
    mu = pred["mean"]
    sd = np.sqrt(np.maximum(pred["var"], 0.0))

    fig, ax = plt.subplots(figsize=(11, 3.4))

    # Uncertainty band first so other elements draw on top.
    ax.fill_between(
        X_test[:, 0],
        mu - 2.0 * sd,
        mu + 2.0 * sd,
        color=BLUE,
        alpha=0.16,
        label="95% credible interval",
    )

    # True generating function as a faint dashed line for reference.
    ax.plot(
        X_test[:, 0],
        np.sin(X_test[:, 0]),
        color=GREY,
        linestyle="--",
        linewidth=1.0,
        label="true f(x) = sin(x)",
    )

    # GP posterior mean.
    ax.plot(X_test[:, 0], mu, color=BLUE, linewidth=2.2, label="GP posterior mean")

    # Training points last so they sit cleanly on top.
    ax.scatter(
        X_train[:, 0],
        y_train,
        color="black",
        s=22,
        zorder=5,
        label="training observations",
    )

    # Subtle vertical guides at the training-data extents.
    ax.axvline(X_train[0, 0], color=GREY, linewidth=0.5, alpha=0.4)
    ax.axvline(X_train[-1, 0], color=GREY, linewidth=0.5, alpha=0.4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-2.4, 2.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(OUT_DIR, "hero_regression.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    build_hero()
