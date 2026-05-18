# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""Smoke tests for the Kernel hierarchy through the Python bindings."""
import numpy as np
import pytest

import lightgp as gp


def test_rbf_basic():
    k = gp.RBF()
    assert k.num_params() == 2
    assert "RBF" in k.name()


def test_matern_variants():
    for nu in (0.5, 1.5, 2.5):
        k = gp.Matern(nu=nu)
        assert k.num_params() == 2
        assert "Matern" in k.name()


def test_periodic_three_params():
    k = gp.Periodic(period=1.0)
    assert k.num_params() == 3
    assert "Periodic" in k.name()


def test_linear_one_param():
    k = gp.Linear()
    assert k.num_params() == 1


def test_composition_add():
    k = gp.RBF() + gp.Periodic()
    assert "+" in k.name()
    assert k.num_params() == gp.RBF().num_params() + gp.Periodic().num_params()


def test_composition_mul():
    k = gp.RBF() * gp.Linear()
    assert "*" in k.name()
    assert k.num_params() == gp.RBF().num_params() + gp.Linear().num_params()


def test_scale_adds_one_param():
    base = gp.RBF()
    sc = gp.Scale(base)
    assert sc.num_params() == base.num_params() + 1


def test_deep_composition():
    k = (gp.Scale(gp.RBF()) + gp.Scale(gp.Periodic())) * gp.Scale(gp.Linear())
    expected = (2 + 1) + (3 + 1) + (1 + 1)
    assert k.num_params() == expected


def test_params_roundtrip():
    k = gp.Scale(gp.RBF()) + gp.Periodic()
    params = list(k.get_params())
    k.set_params([p + 0.1 for p in params])
    new_params = list(k.get_params())
    assert len(params) == len(new_params)
    for old, new in zip(params, new_params):
        assert abs(new - old - 0.1) < 1e-5


def test_repr_includes_name():
    r = repr(gp.RBF())
    assert "lightgp" in r and "RBF" in r


def test_zero_mean_no_params():
    m = gp.ZeroMean()
    assert m.num_params() == 0


def test_constant_mean_one_param():
    m = gp.ConstantMean()
    assert m.num_params() == 1


def test_linear_mean_d_plus_one_params():
    m = gp.LinearMean(input_dim=3)
    assert m.num_params() == 4
