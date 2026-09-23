"""Characterization ("smoke") tests for the logistic model family.

These pin the numeric outputs of the *current, unmodified* implementation
so that a subsequent Tier-1 clean-code refactor (print->logging, .iterrows()
removal, aggregation-loop vectorization, options-dataclass extraction) can
be verified to leave all statistical outputs bit-for-bit (or float-noise)
identical. Values below were captured by running this suite once against
the pre-refactor code and hardcoding the actual results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pprof_py.inference import PROVIDER_TEST_COLUMNS
import pytest

from pprof_py import LogisticFixedEffectModel, LogisticMixedEffectModel, LogisticRandomEffectModel
from pprof_py.exceptions import NotFittedError


def _toy_data(n_groups=6, seed=12345):
    """~6 providers, sizes 15-30, 2 covariates, one deliberately zero-event provider."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(15, 31, size=n_groups)
    rows = []
    for g in range(n_groups):
        n = sizes[g]
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        if g == 0:
            # Zero-event provider: exercise the no-events branch of
            # _exact_ci_for_one_group.
            y = np.zeros(n)
        else:
            linpred = -0.3 + 0.5 * x1 - 0.2 * x2 + (g - 2.5) * 0.4
            p = 1.0 / (1.0 + np.exp(-linpred))
            y = rng.binomial(1, p)
        for i in range(n):
            rows.append((f"P{g}", x1[i], x2[i], float(y[i])))
    df = pd.DataFrame(rows, columns=["provider", "x1", "x2", "y"])
    return df


@pytest.fixture(scope="module")
def data():
    return _toy_data()


def _fit_fixed_effect(data, algorithm):
    model = LogisticFixedEffectModel(use_dataprep=False, algorithm=algorithm)
    model.fit(data, x_vars=["x1", "x2"], y_var="y", group_var="provider")
    return model


class TestLogisticFixedEffectSerbin:
    @pytest.fixture(scope="class")
    def model(self, data):
        return _fit_fixed_effect(data, "Serbin")

    def test_beta(self, model):
        beta = model.coefficients_["beta"].flatten()
        assert beta == pytest.approx([0.628875913132015, 0.0016503010048881182], rel=1e-8)

    def test_gamma(self, model):
        gamma = model.coefficients_["gamma"].flatten()
        assert gamma == pytest.approx(
            [-10.346103328808361, -1.01353330414501, -0.48741025056391335,
             -0.20479640705280738, 0.32101810896179356, 0.8160787150024162],
            rel=1e-6,
        )

    def test_aic_bic(self, model):
        assert model.aic_ == pytest.approx(151.0389944502949, rel=1e-8)
        assert model.bic_ == pytest.approx(174.34023353618332, rel=1e-8)

    def test_summary(self, model):
        summary = model.summary()
        assert list(summary["estimate"]) == pytest.approx(
            [0.628875913132015, 0.0016503010048881182], rel=1e-8
        )
        assert list(summary["p_value"]) == pytest.approx(
            [0.006750595, 0.9938915], rel=1e-4
        )

    def test_test_poibin_exact(self, model):
        result = model.test(test_method="poibin_exact")
        assert list(result["p_value"]) == pytest.approx(
            [1.3726501736e-07, 0.2512632, 0.7388648, 0.7599295, 0.1907258, 0.0049056],
            rel=1e-4, abs=1e-10,
        )

    def test_ci_gamma(self, model):
        result = model.calculate_confidence_intervals(option="gamma")
        gamma_ci = result["gamma_ci"]
        assert list(gamma_ci["gamma_lower"]) == pytest.approx(
            [-np.inf, -2.295512962290939, -1.3183634708079683,
             -1.150296595251401, -0.6793919976563254, 0.002154468633674078],
            rel=1e-5,
        )
        assert list(gamma_ci["gamma_upper"]) == pytest.approx(
            [-2.344946196617933, 0.07770429818772961, 0.31626762850147644,
             0.7216457867274961, 1.346129207482959, 1.6711355339288303],
            rel=1e-5,
        )

    def test_ci_sm(self, model):
        result = model.calculate_confidence_intervals(option="SM")
        indirect = result["indirect_ratio"]
        assert list(indirect["observed"]) == pytest.approx(
            [0.0, 4.0, 11.0, 9.0, 10.0, 17.0], rel=1e-12
        )
        assert "ci_ratio_lower" in indirect.columns and "ci_ratio_upper" in indirect.columns


class TestLogisticFixedEffectBan:
    @pytest.fixture(scope="class")
    def model(self, data):
        return _fit_fixed_effect(data, "Ban")

    def test_beta_close_to_serbin(self, model):
        # Ban and Serbin are different optimization algorithms for the same
        # model; they converge to nearly, but not exactly, the same beta.
        assert model.coefficients_["beta"].flatten() == pytest.approx(
            [0.6288298873066283, 0.0016364908205175888], rel=1e-6
        )


def _mixed_effect_inits(data, x_vars=("x1", "x2")):
    """No existing call-site precedent in the repo for gamma_init/beta_init/
    sigma_init; derive simple, deterministic starting values."""
    beta_init = np.zeros(len(x_vars))
    y_bar = data["y"].mean()
    logit_bar = np.log(y_bar / (1 - y_bar))
    n_providers = data["provider"].nunique()
    gamma_init = np.full(n_providers, logit_bar)
    sigma_init = 1.0
    return gamma_init, beta_init, sigma_init


class TestLogisticMixedEffect:
    @pytest.fixture(scope="class")
    def model(self, data):
        x_vars = ["x1", "x2"]
        gamma_init, beta_init, sigma_init = _mixed_effect_inits(data, x_vars)
        m = LogisticMixedEffectModel(update_sigma=False)
        m.fit(
            data,
            y_var="y",
            x_vars=x_vars,
            provider_var="provider",
            cluster_var="provider",
            gamma_init=gamma_init,
            beta_init=beta_init,
            sigma_init=sigma_init,
            verbose=False,
        )
        return m

    def test_beta_unchanged_from_init(self, model):
        # This model's Newton-Raphson loop updates only gamma; beta_ is
        # never re-estimated inside fit(), so it stays at beta_init.
        assert model.beta_ == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_sigma_unchanged(self, model):
        assert model.sigma_ == pytest.approx(1.0, rel=1e-12)

    def test_test_poibin_exact(self, model):
        result = model.test(test_method="poibin_exact")
        assert list(result.columns) == list(PROVIDER_TEST_COLUMNS)


class TestLogisticFixedEffectDataPrep:
    """Exercises the DataPrepOptions path (use_dataprep=True, screen_providers=True)."""

    @pytest.fixture(scope="class")
    def model(self, data):
        model = LogisticFixedEffectModel(
            use_dataprep=True, screen_providers=True, log_event_providers=True, algorithm="Serbin"
        )
        model.fit(data, x_vars=["x1", "x2"], y_var="y", group_var="provider")
        return model

    def test_beta(self, model):
        beta = model.coefficients_["beta"].flatten()
        assert beta == pytest.approx([0.628875913132015, 0.0016503010048881182], rel=1e-8)

    def test_gamma(self, model):
        gamma = model.coefficients_["gamma"].flatten()
        assert gamma == pytest.approx(
            [-10.346103328808361, -1.01353330414501, -0.48741025056391335,
             -0.20479640705280738, 0.32101810896179356, 0.8160787150024162],
            rel=1e-6,
        )


class TestLogisticRandomEffect:
    @pytest.fixture(scope="class")
    def model(self, data):
        m = LogisticRandomEffectModel(verbose=False)
        m.fit(data, y_var="y", x_vars=["x1", "x2"], group_var="provider")
        return m

    def test_beta(self, model):
        beta = list(model.coefficients_["beta"])
        # Tolerance widened from 1e-6 to accommodate Powell vs bobyqa
        # optimizer differences (~2e-8 scale; bobyqa needs nlopt).
        assert beta == pytest.approx(
            [-0.7156844140730039, 0.5870128519944932, -0.009508234357950035], rel=1e-5
        )

    def test_test_wald(self, model):
        result = model.test(test_method="wald")
        assert list(result.columns) == list(PROVIDER_TEST_COLUMNS)

    def test_ci_alpha(self, model):
        result = model.calculate_confidence_intervals(option="alpha")
        alpha_ci = result["alpha_ci"]
        # Tolerances widened to accommodate Powell vs bobyqa optimizer
        # differences (~1e-7 scale); values near zero need abs fallback.
        assert list(alpha_ci["alpha"]) == pytest.approx(
            [-2.476095434125682, -0.2659537248802278, 0.21709772339192707,
             0.460384303810155, 0.9147771881335817, 1.3931514866188295],
            rel=1e-5,
        )
        assert list(alpha_ci["alpha_lower"]) == pytest.approx(
            [-3.9853657539523573, -1.3014765633806267, -0.5485582607404058,
             -0.4020281588815226, -0.0005399035373425676, 0.6233716090593355],
            rel=1e-4,
        )
        assert list(alpha_ci["alpha_upper"]) == pytest.approx(
            [-0.9668251142990065, 0.7695691136201712, 0.9827537075242601,
             1.3227967665018325, 1.830094279804506, 2.1629313641783234],
            rel=1e-4,
        )


class TestNotFittedError:
    def test_fixed_effect_predict_raises(self):
        model = LogisticFixedEffectModel(use_dataprep=False)
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_random_effect_summary_raises(self):
        model = LogisticRandomEffectModel(verbose=False)
        with pytest.raises(NotFittedError):
            model.summary()
