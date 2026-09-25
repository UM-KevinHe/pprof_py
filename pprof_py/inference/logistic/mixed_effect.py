"""Covariate-effect inference for ``LogisticMixedEffectModel``.  Mixed
into the model class so that ``models/logistic/mixed_effect.py`` stays
focused on configuration, fitting, and prediction.

Stage 3 holds beta fixed at its Stage 1 estimate, so covariate inference
belongs to Stage 1: ``summary()`` returns the Stage 1 model's Wald table, as
R's ``summary.glmm.covar`` does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MixedEffectInferenceMixin:
    """Covariate (beta) inference for `LogisticMixedEffectModel`, taken from Stage 1."""

    def summary(self, stage1_model=None, covariates=None, level: float = 0.95, null: float = 0.0,
                alternative: str = "two_sided") -> pd.DataFrame:
        """Wald inference for the covariate effects, from the Stage 1 model.

        Beta is estimated in Stage 1 (``LogisticFixedEffectModel``) and held
        fixed here, so its standard errors are Stage 1's, whose Wald variance
        accounts for the estimated provider effects. (Earlier versions computed
        an information matrix from the Stage 3 fit at fixed beta and gamma,
        which understated the standard errors.)

        Parameters
        ----------
        stage1_model : LogisticFixedEffectModel, optional
            The fitted Stage 1 model whose beta was passed as ``beta_init``.
            Defaults to the one given to ``fit(stage1_model=...)``.
        covariates, level, null, alternative
            Passed to the Stage 1 model's ``summary(test_method="wald")``.

        Returns
        -------
        pandas.DataFrame
            The Stage 1 Wald table.

        Raises
        ------
        ValueError
            If no Stage 1 model is available, or its beta differs from the beta
            this model was fit with.
        """
        self._check_is_fitted()
        stage1 = stage1_model if stage1_model is not None else getattr(self, "stage1_model_", None)
        if stage1 is None:
            raise ValueError("summary() requires the Stage 1 model for covariate inference. "
                             "Pass it via stage1_model= at fit() or summary() time.")
        coefs = getattr(stage1, "coefficients_", None)
        if not isinstance(coefs, dict) or "beta" not in coefs:
            raise ValueError("stage1_model must be a fitted LogisticFixedEffectModel (it has no coefficients_['beta']).")
        beta1 = np.asarray(coefs["beta"], dtype=np.float64).ravel()
        if beta1.shape != np.shape(self.beta_) or not np.allclose(beta1, self.beta_, rtol=1e-10, atol=1e-12):
            raise ValueError("stage1_model's beta differs from the beta this model was fit with (beta_init); "
                             "pass the Stage 1 model whose beta was used.")
        return stage1.summary(covariates=covariates, level=level, null=null, alternative=alternative,
                              test_method="wald")
