"""Characterization ("smoke") tests for the linear model family.

These pin the numeric outputs of the linear model implementations so that
refactors and clean-code changes can be verified to leave all statistical
outputs bit-for-bit (or float-noise) identical.

LinearRandomEffectModel values are pinned against the pure-Python lme4-style
implementation (linear_random_effect_model.py), validated against R lme4 to
1e-8 precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pprof_py import LinearFixedEffectModel
from pprof_py.exceptions import NotFittedError
from pprof_py.models.linear import LinearRandomEffectModel


def _toy_data(n_groups=6, seed=12345):
    """~6 providers, sizes 15-30, 2 covariates, continuous outcome."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(15, 31, size=n_groups)
    rows = []
    for g in range(n_groups):
        n = sizes[g]
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 2.0 + 0.5 * x1 - 0.3 * x2 + (g - 2.5) * 0.6 + rng.normal(scale=1.0, size=n)
        for i in range(n):
            rows.append((f"P{g}", x1[i], x2[i], float(y[i])))
    df = pd.DataFrame(rows, columns=["provider", "x1", "x2", "y"])
    return df


@pytest.fixture(scope="module")
def data():
    return _toy_data()


class TestLinearFixedEffect:
    @pytest.fixture(scope="class")
    def model(self, data):
        m = LinearFixedEffectModel()
        m.fit(data, x_vars=["x1", "x2"], y_var="y", group_var="provider")
        return m

    def test_beta(self, model):
        beta = model.coefficients_["beta"].flatten()
        assert beta == pytest.approx([0.4505974450005268, -0.35750969780081876], rel=1e-8)

    def test_gamma(self, model):
        gamma = model.coefficients_["gamma"].flatten()
        assert gamma == pytest.approx(
            [0.1765214570226251, 1.0201649378894606, 1.6404827567688767,
             2.268409102606767, 2.5582675408119773, 3.4996149211782925],
            rel=1e-6,
        )

    def test_aic_bic_sigma(self, model):
        assert model.aic_ == pytest.approx(389.31054154678225, rel=1e-8)
        assert model.bic_ == pytest.approx(415.52443551840673, rel=1e-8)
        assert model.sigma_ == pytest.approx(0.976760361271433, rel=1e-8)

    def test_summary(self, model):
        summary = model.summary()
        assert list(summary["estimate"]) == pytest.approx(
            [0.4505974450005268, -0.35750969780081876], rel=1e-8
        )
        assert list(summary["p_value"]) == pytest.approx(
            [3.464937e-07, 0.0001089621], rel=1e-4
        )

    def test_test(self, model):
        result = model.test()
        assert list(result["p_value"]) == pytest.approx(
            [0.0, 9.17e-05, 0.1025434, 0.1566176, 0.0114913, 0.0], rel=1e-4, abs=1e-8,
        )

    def test_ci_gamma(self, model):
        result = model.calculate_confidence_intervals(option="gamma")
        gamma_ci = result["gamma_ci"]
        assert list(gamma_ci["lower"]) == pytest.approx(
            [-0.2039456770031374, 0.5625485775162901, 1.2627021288115579,
             1.8324301982037654, 2.0923565334053307, 3.122367966838708],
            rel=1e-5,
        )
        assert list(gamma_ci["upper"]) == pytest.approx(
            [0.5569885910483876, 1.4777812982626313, 2.0182633847261955,
             2.704388007009769, 3.024178548218624, 3.876861875517877],
            rel=1e-5,
        )


class TestLinearRandomEffect:
    @pytest.fixture(scope="class")
    def model(self, data):
        m = LinearRandomEffectModel(verbose=False)
        m.fit(data, x_vars=["x1", "x2"], y_var="y", group_var="provider")
        return m

    def test_beta(self, model):
        beta = list(model.coefficients_["beta"])
        assert beta == pytest.approx(
            [1.8608102729997062, 0.45255745280373444, -0.3643456797321602], rel=1e-6
        )

    def test_standardized_measures(self, model):
        sm = model.calculate_standardized_measures()
        indirect = sm["indirect"]
        assert list(indirect["observed"]) == pytest.approx(
            [5.01566100764183, 17.426588755388252, 38.602518044792646,
             42.42011894816041, 49.12259201876251, 99.0938663396824],
            rel=1e-6,
        )

    def test_test(self, model):
        result = model.test()
        assert list(result["p_value"]) == pytest.approx(
            [0.0, 0.0003505, 0.2525637, 0.0651199, 0.0031193, 0.0], rel=1e-4, abs=1e-8,
        )

    def test_ci_alpha(self, model):
        result = model.calculate_confidence_intervals(option="alpha")
        alpha_ci = result["alpha_ci"]
        assert list(alpha_ci["alpha"]) == pytest.approx(
            [-1.6386765622990334, -0.8072643232921222, -0.21229971367525113,
             0.39588333592491026, 0.6674600459620006, 1.5948972173794325],
            rel=1e-6,
        )
        assert list(alpha_ci["alpha_lower"]) == pytest.approx(
            [-2.009100213519645, -1.2498611845798564, -0.575977338087396,
             -0.0247968798094938, 0.22486318467426641, 1.2312195929672876],
            rel=1e-5,
        )
        assert list(alpha_ci["alpha_upper"]) == pytest.approx(
            [-1.2682529110784215, -0.364667462004388, 0.15137791073689383,
             0.8165635516593144, 1.1100569072497348, 1.9585748417915774],
            rel=1e-5,
        )


class TestNotFittedError:
    def test_fixed_effect_predict_raises(self):
        model = LinearFixedEffectModel()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_random_effect_summary_raises(self):
        model = LinearRandomEffectModel()
        with pytest.raises(NotFittedError):
            model.summary()
