# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

import numpy as np
import pytest


@pytest.fixture
def sin_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(-3, 3, 100)).reshape(-1, 1).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.standard_normal(100)).astype(np.float32)
    X_test = np.linspace(-3, 3, 50, dtype=np.float32).reshape(-1, 1)
    return X, y, X_test


@pytest.fixture
def seasonal_data():
    rng = np.random.default_rng(42)
    X = np.linspace(0, 6, 200, dtype=np.float32).reshape(-1, 1)
    y = (0.5 * X[:, 0]
         + np.sin(2 * np.pi * X[:, 0])
         + 0.1 * rng.standard_normal(200)).astype(np.float32)
    return X, y


@pytest.fixture
def multidim_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.5 * X[:, 1]
         + 0.1 * rng.standard_normal(200)).astype(np.float32)
    X_test = rng.standard_normal((50, 4)).astype(np.float32)
    return X, y, X_test
