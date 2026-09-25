"""Variance estimation and covariate (beta) hypothesis tests for
``LogisticFixedEffectModel``.  Mixed into the model class so that
``models/logistic/fixed_effect.py`` stays focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np
from ....utils.numerical import covariance_from_information, solve_information
import pandas as pd
from scipy.stats import norm, chi2

from ....utils.numerical import sigmoid


class _LogisticFEInferenceHost(Protocol):
    """Attribute contract that `LogisticFixedEffectInferenceMixin` expects from its
    host class (`LogisticFixedEffectModel`).

    This Protocol exists purely for documentation and static-analysis
    purposes; it is never instantiated at runtime.  It declares exactly
    which ``self.<attr>`` names the mixin's methods read, so that the
    dependencies are visible at the top of this file rather than
    discoverable only by reading every method body.
    """
    # --- Fitted attributes (set by fit()) ---
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    provider_ids_: Optional[np.ndarray]
    provider_indices_: Optional[np.ndarray]
    outcome_: Optional[np.ndarray]
    obs_ids_: Optional[np.ndarray]
    N_: Optional[np.ndarray]
    covariate_names_: list
    robust_variances_: Optional[Dict[str, Any]]
    # --- Design matrix (set during fit()) ---
    X: Optional[np.ndarray]
    # --- Configuration ---
    algorithm: Any  # SerbinAlgorithm | BanAlgorithm
    algorithm_type: str

    def _check_is_fitted(self) -> None: ...


class _CovariateInferenceMethods:
    """Variance estimation and covariate-level (beta) statistical
    inference for `LogisticFixedEffectModel`."""

    def _estimate_variances(self) -> dict:
        """Estimate variances of beta and gamma coefficients for inferential statistics.

        Returns
        -------
        dict
            Variances of beta (covariance matrix) and gamma (diagonal variances).

        Raises
        ------
        ValueError
            If the model has not been fitted (i.e., required attributes are None).
        """
        # Check if the model has been fitted
        if self.fitted_ is None or self.X is None or self.provider_indices_ is None:
            raise ValueError("Model must be fitted before estimating variances.")

        # Use precomputed predicted probabilities
        p = self.fitted_
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Ensure numerical stability

        q = self.N_ * p * (1 - p)

        # Number of groups
        n_groups = len(self.provider_ids_)

        # Information for gamma: inverse of sum(q) per group
        info_gamma_inv = 1 / np.bincount(self.provider_indices_, weights=q, minlength=n_groups)

        # Cross-information: sum(X * q) per group for each covariate
        info_beta_gamma = np.array([
            np.bincount(self.provider_indices_, weights=q * self.X[:, i], minlength=n_groups)
            for i in range(self.X.shape[1])
        ])  # Shape: (n_covariates, n_groups)

        # Information for beta: X.T @ (q * X)
        info_beta = self.X.T @ (q[:, None] * self.X)

        # Variance for beta
        mat_tmp1 = (info_gamma_inv * info_beta_gamma).T  # Shape: (n_covariates, n_groups)
        schur_complement = info_beta - info_beta_gamma @ mat_tmp1
        info_beta_inv = covariance_from_information(
            schur_complement, warn=True, what="Schur complement of the information matrix")
        var_beta = info_beta_inv

        # Variance for gamma
        quad_term = np.sum((mat_tmp1 @ info_beta_inv) * mat_tmp1, axis=1)  # Shape: (n_groups,)
        var_gamma = info_gamma_inv + quad_term

        return {"beta": var_beta, "gamma": var_gamma}

    def _compute_robust_variances(self) -> dict:
        """Compute cluster-robust (sandwich) variances for gamma and beta coefficients.

        Implements a Huber-White sandwich estimator clustered at the observation
        ID level (e.g., patient). This accounts for within-cluster correlation
        when observations are repeated (e.g., same patient across multiple years).

        For gamma (provider effects), the sandwich formula is:
            robust_var(gamma_j) = (1/I_j)^2 * A0_j
        where:
            I_j = sum(p_i * (1 - p_i)) for observations i in group j
            A0_j = sum_k[ (sum_{i in cluster k}(Y_i - p_i))^2 ] within group j

        For beta (covariate coefficients), the full joint sandwich is:
            V_beta = S^{-1} @ M_eff @ S^{-1}
        where:
            S = Schur complement of the information matrix (same as model-based)
            M_eff = B @ diag(A0 * D^{-2}) @ B^T
                  - B @ diag(D^{-1}) @ A1_t
                  - A1_t^T @ diag(D^{-1}) @ B^T
                  + A2
            B = cross-information matrix (p x m)
            D = diag(I_j) = gamma information (m x m diagonal)
            A0 = gamma meat vector (m,)
            A1_t = cross-term meat matrix (m x p)
            A2 = beta meat matrix (p x p)

        This matches R's robust_wald_covar() from summary_fe_covar.R.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'gamma': array of robust variances for each group's fixed effect
            - 'beta': robust covariance matrix for covariate coefficients (p x p)

        Raises
        ------
        ValueError
            If the model has not been fitted or observation IDs are not available.
        """
        if self.fitted_ is None or self.obs_ids_ is None:
            raise ValueError(
                "Model must be fitted with obs_id_var provided to compute robust variances."
            )

        p = self.fitted_
        p = np.clip(p, 1e-10, 1 - 1e-10)
        residuals = self.outcome_ - self.N_ * p
        q = self.N_ * p * (1 - p)

        n_groups = len(self.provider_ids_)
        n_covariates = self.X.shape[1]

        # Pre-compute U_b = X * residual (observation-level beta scores)
        U_b = self.X * residuals[:, np.newaxis]  # (n, p)

        # Meat components
        robust_var_gamma = np.zeros(n_groups)
        A0 = np.zeros(n_groups)           # gamma meat: sum of (cluster res sum)^2 per provider
        A1_t = np.zeros((n_groups, n_covariates))  # cross-term: (m, p)
        A2 = np.zeros((n_covariates, n_covariates))  # beta meat: (p, p)

        for j in range(n_groups):
            mask_j = (self.provider_indices_ == j)

            # Bread: 1 / I_j where I_j = sum(q) within group j
            info_gamma_j = np.sum(q[mask_j])
            if info_gamma_j < 1e-14:
                robust_var_gamma[j] = np.nan
                A0[j] = 0.0
                continue
            bread_j = 1.0 / info_gamma_j

            # Cluster aggregation: group by observation ID within this provider
            obs_ids_j = self.obs_ids_[mask_j]
            residuals_j = residuals[mask_j]
            U_b_j = U_b[mask_j]  # (n_j, p)

            unique_ids, inverse_idx = np.unique(obs_ids_j, return_inverse=True)
            n_clusters = len(unique_ids)

            # Cluster sums of residuals
            cluster_res_sums = np.bincount(inverse_idx, weights=residuals_j,
                                           minlength=n_clusters)

            # Cluster sums of U_b (beta scores) for each covariate
            cluster_ub_sums = np.zeros((n_clusters, n_covariates))
            for col in range(n_covariates):
                cluster_ub_sums[:, col] = np.bincount(
                    inverse_idx, weights=U_b_j[:, col], minlength=n_clusters
                )

            # A0[j]: gamma meat for provider j
            A0[j] = np.sum(cluster_res_sums ** 2)

            # A2: beta meat (accumulates across all providers)
            # sum of outer products of cluster beta-score sums
            A2 += cluster_ub_sums.T @ cluster_ub_sums

            # A1_t[j]: cross-term for provider j
            # sum of (cluster_res_sum_k * cluster_ub_sum_k) over clusters k in provider j
            A1_t[j, :] = cluster_res_sums @ cluster_ub_sums

            # Gamma robust variance (same as before)
            robust_var_gamma[j] = bread_j ** 2 * A0[j]

        # --- Beta robust variance via full sandwich ---
        # Recompute information matrix components needed for the sandwich
        # D_inv (m,): inverse of gamma information per provider
        info_gamma_inv = 1.0 / np.bincount(self.provider_indices_, weights=q,
                                            minlength=n_groups)

        # B (p, m): cross-information matrix
        info_beta_gamma = np.array([
            np.bincount(self.provider_indices_, weights=q * self.X[:, i],
                        minlength=n_groups)
            for i in range(n_covariates)
        ])  # shape: (p, m)

        # S_inv (p, p): model-based variance for beta (Schur complement inverse)
        # This is already stored in self.variances_["beta"]
        S_inv = self.variances_["beta"]  # (p, p)

        # Compute effective meat for beta:
        # M_eff = B @ diag(A0 * D_inv^2) @ B^T
        #       - B @ diag(D_inv) @ A1_t
        #       - A1_t^T @ diag(D_inv) @ B^T
        #       + A2
        D_inv = info_gamma_inv  # (m,)

        # Term 1: B @ diag(A0 * D_inv^2) @ B^T
        weight_1 = A0 * D_inv ** 2  # (m,)
        term1 = (info_beta_gamma * weight_1[np.newaxis, :]) @ info_beta_gamma.T  # (p, p)

        # Term 2: B @ diag(D_inv) @ A1_t → (p, m) @ (m, p) = (p, p)
        BD_inv = info_beta_gamma * D_inv[np.newaxis, :]  # (p, m)
        term2 = BD_inv @ A1_t  # (p, p)

        # Effective meat
        M_eff = term1 - term2 - term2.T + A2  # (p, p)

        # Robust covariance for beta: V = S^{-1} @ M_eff @ S^{-1}
        V_beta_robust = S_inv @ M_eff @ S_inv  # (p, p)

        return {"gamma": robust_var_gamma, "beta": V_beta_robust}

    def _compute_wald_beta(self, index: int, null: float = 0, alternative: str = "two_sided", alpha: float = 0.05, variance_type: str = "model"):
        """Perform a Wald test for a specific covariate coefficient.

        Parameters
        ----------
        index : int
            Index of the covariate to test.
        null : float, default=0
            Null hypothesis value for the coefficient.
        alternative : str, default="two_sided"
            Alternative hypothesis: "two_sided", "less", or "greater".
        alpha : float, default=0.05
            Significance level for confidence intervals.
        variance_type : str, default="model"
            Variance estimator to use: 'model' (information-based) or
            'robust' (cluster-robust sandwich).

        Returns
        -------
        dict
            Results including test statistic, p-value, and confidence interval.
        """
        self._check_is_fitted()
        if variance_type == "robust":
            if self.robust_variances_ is None or "beta" not in self.robust_variances_:
                raise ValueError(
                    "Robust variances for beta not available. "
                    "Fit the model with obs_id_var to enable cluster-robust variance estimation."
                )
            var_matrix = self.robust_variances_["beta"]
        else:
            var_matrix = self.variances_["beta"]

        beta = self.coefficients_['beta'][index]
        se_beta = np.sqrt(var_matrix[index, index])
        stat = (beta - null) / se_beta

        if alternative == "two_sided":
            p_value = 2 * (1 - norm.cdf(np.abs(stat)))
            crit_value = norm.ppf(1 - alpha / 2)
            ci_lower = beta - crit_value * se_beta
            ci_upper = beta + crit_value * se_beta
        elif alternative == "less":
            p_value = norm.cdf(stat)
            crit_value = norm.ppf(1 - alpha)
            ci_lower = -np.inf
            ci_upper = beta + crit_value * se_beta
        elif alternative == "greater":
            p_value = 1 - norm.cdf(stat)
            crit_value = norm.ppf(1 - alpha)
            ci_lower = beta - crit_value * se_beta
            ci_upper = np.inf
        else:
            raise ValueError("Alternative must be 'two_sided', 'less', or 'greater'")

        return {
            "statistic": stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "alternative": alternative
        }

    def _compute_lr_beta(self, index: int):
        """Perform a Likelihood Ratio test for a specific covariate.

        Parameters
        ----------
        index : int
            Index of the covariate to test.

        Returns
        -------
        dict
            Results including test statistic and p-value.
        """
        self._check_is_fitted()

        # Full model log-likelihood
        gamma_obs_full = self.coefficients_['gamma'][self.provider_indices_]
        loglik_full = self.algorithm._loglikelihood(gamma_obs_full, self.coefficients_['beta'])

        # Fit reduced model excluding the covariate at index
        reduced_X = np.delete(self.X, index, axis=1)
        reduced_model = self.__class__(algorithm=self.algorithm_type)
        reduced_model.fit(
            reduced_X, self.outcome_, self.provider_indices_,
            max_iter=self.algorithm.max_iter,
            tol=self.algorithm.tol,
            bound=self.algorithm.bound,
            backtrack=self.algorithm.backtrack
        )

        # Reduced model log-likelihood
        gamma_obs_reduced = reduced_model.coefficients_['gamma'][reduced_model.provider_indices_]
        loglik_reduced = reduced_model.algorithm._loglikelihood(gamma_obs_reduced, reduced_model.coefficients_['beta'])

        # Test statistic and p-value
        test_stat = 2 * (loglik_full - loglik_reduced)
        p_value = chi2.sf(test_stat, df=1)

        return {
            "statistic": test_stat,
            "p_value": p_value,
            "df": 1
        }

    def _compute_score_beta(self, index: int):
        """Perform a Score test for a specific covariate.

        Parameters
        ----------
        index : int
            Index of the covariate to test.

        Returns
        -------
        dict
            Results including test statistic and p-value.
        """
        self._check_is_fitted()

        # Fit reduced model excluding the covariate at index
        reduced_X = np.delete(self.X, index, axis=1)
        reduced_model = self.__class__(algorithm=self.algorithm_type)
        reduced_model.fit(
            reduced_X, self.outcome_, self.provider_indices_,
            max_iter=self.algorithm.max_iter,
            tol=self.algorithm.tol,
            bound=self.algorithm.bound,
            backtrack=self.algorithm.backtrack
        )

        # Compute probabilities and weights under the reduced model
        gamma_obs = reduced_model.coefficients_['gamma'][reduced_model.provider_indices_]
        p = sigmoid(gamma_obs + reduced_X @ reduced_model.coefficients_['beta'])
        q = p * (1 - p)

        # Score for the excluded covariate
        score_excluded = np.sum((self.outcome_ - p) * self.X[:, index])

        # Information components
        info_excluded_excluded = np.sum(q * self.X[:, index] ** 2)  # scalar
        info_excluded_beta = (q * self.X[:, index]).T @ reduced_X   # (1, p_reduced)
        info_beta = reduced_X.T @ (q[:, None] * reduced_X)          # (p_reduced, p_reduced)

        # Inverse of the information matrix for beta
        schur_inv = covariance_from_information(info_beta, warn=True, what="Reduced-model information")  # (p_reduced, p_reduced)

        # Variance of the score
        info_full = info_excluded_excluded - info_excluded_beta @ schur_inv @ info_excluded_beta.T

        # Test statistic and p-value
        test_stat = score_excluded ** 2 / info_full
        p_value = chi2.sf(test_stat, df=1)

        return {
            "statistic": test_stat,
            "p_value": p_value,
            "df": 1
        }

    def summary(self, covariates: list = None, level: float = 0.95, null: float = 0, alternative: str = "two_sided", test_method: str = "wald", variance_type: str = "model"):
        """Provides summary statistics for the covariate estimates in a fitted fixed effects model.

        Parameters
        ----------
        covariates : list of str or int, optional
            Covariate names or indices to include in the summary. If None, 
            all are included.
        level : float, default=0.95
            Confidence level for intervals.
        null : float, default=0
            Null hypothesis value for the coefficient. Used directly for 
            the Wald test. For LR and Score tests, only null=0 is implemented.
        alternative : str, default="two_sided"
            Hypothesis type ("two_sided", "greater", or "less") for the Wald test.
            LR and Score tests are implicitly two-sided for coefficient = 0.
        test_method : str, default="wald"
            Testing approach: "wald", "lr", or "score".
        variance_type : str, default="model"
            Type of variance estimator for standard errors and confidence intervals.
            - 'model': information-matrix-based SE (assumes independence).
            - 'robust': cluster-robust sandwich SE (requires obs_id_var at fit).
            Only applicable when test_method='wald'. LR and Score tests use their
            own variance estimation and ignore this parameter.

        Returns
        -------
        pd.DataFrame
            Summary statistics with columns:
            - estimate : coefficient estimate
            - std_error : standard error (from the selected variance estimator)
            - stat : test statistic (from wald_test, lr_test, or score_test)
            - p_value : p-value from wald_test, lr_test, or score_test
            - ci_lower, ci_upper : confidence interval bounds (Wald-based)

        Raises
        ------
        ValueError
            If the model is not fitted or if 'test' is invalid or if 
            'covariates' are mis-specified.
        NotImplementedError
            If 'null' != 0 for LR/Score (only null=0 is implemented).
        """
        # ----------------------------------------------------------------------
        # 1. Ensure the model is fitted
        self._check_is_fitted()
        if self.variances_ is None:
            raise ValueError("Variances are not available. Ensure the model is fully fitted.")

        # ----------------------------------------------------------------------
        # 2. Check test validity
        if test_method not in ["wald", "lr", "score"]:
            raise ValueError("Argument 'test' must be one of 'wald', 'lr', or 'score'.")

        # For LR and Score, we only handle null=0
        if test_method in ["lr", "score"] and null != 0:
            raise NotImplementedError(
                f"{test_method.upper()} test currently only implemented for null=0."
            )

        # Validate variance_type
        if variance_type not in ["model", "robust"]:
            raise ValueError("Argument 'variance_type' must be 'model' or 'robust'.")
        if variance_type == "robust":
            if self.robust_variances_ is None or "beta" not in self.robust_variances_:
                raise ValueError(
                    "Robust variances for beta not available. "
                    "Fit the model with obs_id_var to enable cluster-robust variance estimation."
                )

        # Select the appropriate variance matrix for beta
        var_beta_matrix = (
            self.robust_variances_["beta"] if variance_type == "robust"
            else self.variances_["beta"]
        )

        # ----------------------------------------------------------------------
        # 3. Identify which covariate indices to summarize
        beta = self.coefficients_["beta"].flatten()
        n_cov = len(beta)

        if covariates is not None:
            # Covariates can be int indices or string names
            if all(isinstance(c, str) for c in covariates):
                indices = [self.covariate_names_.index(cov) for cov in covariates]
            elif all(isinstance(c, int) for c in covariates):
                indices = covariates
            else:
                raise ValueError("Argument 'covariates' must be a list of names or indices.")
        else:
            indices = range(n_cov)

        # Build the display names (if not provided, generate x0, x1, etc.)
        cov_names = (
            [self.covariate_names_[i] for i in indices]
            if self.covariate_names_
            else [f"x{i}" for i in indices]
        )

        # ----------------------------------------------------------------------
        # 4. Prepare containers for summary results
        estimates = []
        std_errors = []
        stats = []
        p_values = []
        ci_lowers = []
        ci_uppers = []

        # Wald-based intervals use alpha = 1 - level
        alpha = 1 - level

        # We'll compute the standard error from the variance matrix
        # but for "wald" we rely on self._compute_wald_beta() for the official
        # statistic, p-value, and intervals. For "lr"/"score" we 
        # use that method for stat/p-value, but still give a wald-based
        # interval for convenience.
        for idx, name in zip(indices, cov_names):
            estimate = beta[idx]
            # Standard error from the selected variance matrix
            se = np.sqrt(var_beta_matrix[idx, idx])

            if test_method == "wald":
                # NOTE: wald_test expects alpha, not level
                wald_res = self._compute_wald_beta(
                    index=idx, null=null, alternative=alternative, alpha=alpha,
                    variance_type=variance_type
                )
                stat_val = wald_res["statistic"]
                p_val = wald_res["p_value"]
                ci_lower = wald_res["ci_lower"]
                ci_upper = wald_res["ci_upper"]

            elif test_method == "lr":
                # Use the lr_test method for test stat & p-value
                lr_res = self._compute_lr_beta(idx)
                stat_val = lr_res["statistic"]
                p_val = lr_res["p_value"]
                # Construct Wald-based confidence interval for reference
                # two-sided intervals only
                zcrit = norm.ppf(1 - alpha / 2)
                ci_lower = estimate - zcrit * se
                ci_upper = estimate + zcrit * se

            else:  # score
                score_res = self._compute_score_beta(idx)
                stat_val = score_res["statistic"]
                p_val = score_res["p_value"]
                # Construct Wald-based confidence interval
                zcrit = norm.ppf(1 - alpha / 2)
                ci_lower = estimate - zcrit * se
                ci_upper = estimate + zcrit * se

            # Accumulate results for the summary dataframe
            estimates.append(estimate)
            std_errors.append(se)
            stats.append(stat_val)
            p_values.append(p_val)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        # ----------------------------------------------------------------------
        # 5. Construct the final DataFrame
        summary_df = pd.DataFrame(
            {
                "estimate": estimates,
                "std_error": std_errors,
                "stat": stats,
                "p_value": p_values,
                "ci_lower": ci_lowers,
                "ci_upper": ci_uppers,
            },
            index=cov_names,
        )

        return summary_df
