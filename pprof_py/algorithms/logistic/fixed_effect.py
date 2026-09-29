"""Newton-based optimization for logistic fixed-effect provider models.

Two algorithms are implemented:

* ``SerbinAlgorithm`` — block-update Newton steps (gamma and beta solved
  jointly via a Schur-complement factorization), with optional
  Armijo backtracking.
* ``BanAlgorithm`` — alternating one-block-at-a-time updates (gamma
  then beta), with optional backtracking on each block.

Both share a common ``BaseAlgorithm`` that owns data validation,
precomputed provider structure (sparse indicator matrix), and the
log-likelihood evaluation.
"""
import logging

import numpy as np
from ...utils.numerical import solve_information
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from scipy.special import expit as sigmoid
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

logger = logging.getLogger(__name__)


class Algorithm(Enum):
    """Enumeration for optimization algorithms."""
    SERBIN = "Serbin"
    BAN = "Ban"


@dataclass
class AlgorithmOptions:
    """Solver knobs shared by `SerbinAlgorithm` and `BanAlgorithm`.

    Parameters
    ----------
    backtrack : bool, default=True
        Whether to use backtracking line search during optimization.
    max_iter : int, default=10000
        The maximum number of iterations for the optimization algorithm.
    bound : float, default=10.0
        The bound for clipping group effects to prevent extreme values.
    tol : float, default=1e-8
        The tolerance for convergence, based on the change in parameters.
    """
    backtrack: bool = True
    max_iter: int = 10000
    bound: float = 10.0
    tol: float = 1e-8


class BaseAlgorithm(ABC):
    """Base class for optimization algorithms in logistic fixed effect models.

    This abstract base class defines the common interface and shared functionality
    for optimization algorithms used in logistic regression with fixed effects.
    It handles the initialization of model parameters, data, and optimization settings.
    Subclasses must implement the `fit` method to perform the actual optimization.

    Parameters
    ----------
    data : np.ndarray
        The input data array, containing the response variable, group identifiers,
        and covariates.
    y_index : int
        The column index of the response variable in the data array.
    X : np.ndarray
        The design matrix for the covariates.
    prov_index : int
        The column index of the group (provider) identifiers in the data array.
    n_prov : np.ndarray
        An array containing the number of observations for each group.
    gamma_prov : np.ndarray
        Initial values for the group-specific fixed effects.
    beta : np.ndarray
        Initial values for the covariate coefficients.
    options : AlgorithmOptions
        Solver knobs (backtrack, max_iter, bound, tol).
    """
    def __init__(
        self,
        data: np.ndarray,
        y_index: int,
        X: np.ndarray,
        prov_index: int,
        n_prov: np.ndarray,
        gamma_prov: np.ndarray,
        beta: np.ndarray,
        options: AlgorithmOptions,
        N: np.ndarray = None,
    ):
        """Initialize the BaseAlgorithm with data and parameters.

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        y_index : int
            The column index of the response variable.
        X : np.ndarray
            The design matrix for covariates.
        prov_index : int
            The column index of the provider identifiers.
        n_prov : np.ndarray
            Number of observations per provider.
        gamma_prov : np.ndarray
            Initial group-specific fixed effects.
        beta : np.ndarray
            Initial covariate coefficients.
        options : AlgorithmOptions
            Solver knobs (backtrack, max_iter, bound, tol).
        N : np.ndarray, optional
            Number of trials per observation (binomial N). If None,
            defaults to 1 for all observations (Bernoulli model).
        """
        self.data = data
        self.y_index = y_index
        self.X = X
        self.prov_index = prov_index
        self.n_prov = n_prov
        self.gamma_prov = gamma_prov
        self.beta = beta
        self.backtrack = options.backtrack
        self.max_iter = options.max_iter
        self.bound = options.bound
        self.tol = options.tol

        # Convert response column to integers and validate
        if self.data[:, self.y_index].dtype == object:
            self.data[:, self.y_index] = self.data[:, self.y_index].astype(int)
        y_col = self.data[:, self.y_index].astype(float)
        if np.isnan(y_col).any() or np.isinf(y_col).any():
            raise ValueError("Response variable contains NaN or infinite values")
        self.y = y_col.astype(int)

        # Binomial N: number of trials per observation (default 1 = Bernoulli)
        self.N = N if N is not None else np.ones(len(self.y))

        # Precompute provider structure (invariant across iterations)
        self._prov_unique, self._prov_indices = np.unique(
            self.data[:, self.prov_index], return_inverse=True
        )
        self._n_providers = len(self._prov_unique)
        # Sparse indicator matrix for vectorized per-provider aggregation
        n_obs = len(self.y)
        self._prov_indicator = sparse.csc_matrix(
            (np.ones(n_obs), (np.arange(n_obs), self._prov_indices)),
            shape=(n_obs, self._n_providers)
        )

    def _loglikelihood(self, gamma_obs: np.ndarray, beta: np.ndarray) -> float:
        """Compute the log-likelihood under logistic model with fixed effects.

        Parameters
        ----------
        gamma_obs : np.ndarray
            Provider-specific linear predictors repeated per observation.
        beta : np.ndarray
            Regression coefficients.

        Returns
        -------
        float
            Total log-likelihood.
        """
        linear = gamma_obs + self.X @ beta
        # Numerically stable log(1+exp(x)): avoids overflow when linear > 709
        # np.logaddexp(0, x) = log(exp(0) + exp(x)) = log(1 + exp(x)), stable for all x
        return np.sum(self.y * linear - self.N * np.logaddexp(0, linear))

    @staticmethod
    def _below_noise(predicted: float, loglik: float) -> bool:
        """Whether an Armijo target is below the log-likelihood's rounding level.

        C3: near the optimum a Newton step's predicted gain (``s * v * lambda``)
        falls below the rounding error of the log-likelihood sum, so the
        Armijo test compares rounding noise and can fail at every step size;
        backtracking then shrank the step to exactly 0 (about 1,400
        evaluations with ``t = 0.6``) and the zero step read as convergence.
        A step whose predicted gain is at that level is accepted as it is.
        """
        return 0.0 < predicted <= 1e-12 * (1.0 + abs(loglik))


    @abstractmethod
    def _backtrack(self) -> None:
        """Gradient ascent with backtracking line search.
        """
        pass

    @abstractmethod
    def _no_backtrack(self) -> None:
        """Gradient ascent without line search.
        """
        pass

    def fit(self) -> dict:
        """Fit the model until convergence or maximum iterations.

        Returns
        -------
        dict
            Dictionary with final estimates:
            - 'gamma': provider effects (np.ndarray)
            - 'beta': regression coefficients (np.ndarray)
        """
        self.iter = 0
        self.beta_crit = np.inf
        self.gamma_crit = np.inf
        if self.backtrack:
            self._backtrack()
        else:
            self._no_backtrack()
        return {'gamma': self.gamma_prov, 'beta': self.beta}


class SerbinAlgorithm(BaseAlgorithm):
    """Serbin's algorithm for logistic fixed-effect estimation.

    Extends BaseAlgorithm with block-update Newton steps and optional backtracking.
    """
    def _compute_scores_and_info(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> tuple:
        """Compute score vectors and information matrix components.

        Parameters
        ----------
        p : np.ndarray
            Predicted probabilities per observation.
        q : np.ndarray
            Variance terms p*(1-p) per observation.

        Returns
        -------
        tuple
            (score_gamma, score_beta, info_gamma_inv,
             info_beta_gamma, info_beta)
        """
        # Zero-guard: prevent division by zero when p≈0 or p≈1
        # (matches R's Fixed_effect.cpp lines 128-129)
        q = np.maximum(q, 1e-20)

        indices = self._prov_indices
        n_provs = self._n_providers
        residuals = self.y - self.N * p

        score_gamma = np.bincount(indices, weights=residuals, minlength=n_provs)
        score_beta = self.X.T @ residuals
        info_gamma_inv = 1 / np.bincount(indices, weights=q, minlength=n_provs)

        # Vectorized info_beta_gamma using sparse indicator matrix
        # Replaces Python loop: for i in range(p): bincount(weights=q*X[:,i])
        qX = q[:, None] * self.X  # (n, p) — reused for info_beta below
        info_beta_gamma = (self._prov_indicator.T @ qX).T  # (p, m)

        info_beta = self.X.T @ qX  # (p, p)
        return score_gamma, score_beta, info_gamma_inv, info_beta_gamma, info_beta

    def _compute_deltas(
        self,
        score_gamma: np.ndarray,
        score_beta: np.ndarray,
        info_gamma_inv: np.ndarray,
        mat_tmp1: np.ndarray,
        mat_tmp2: np.ndarray,
        schur_score_beta: np.ndarray,
    ) -> tuple:
        """Compute Newton deltas using the same linear algebra as the Rcpp SerBIN code.

        Parameters
        ----------
        score_gamma : np.ndarray
            Score vector for provider effects, shape (m,).
        score_beta : np.ndarray
            Score vector for regression coefficients, shape (p,).
        info_gamma_inv : np.ndarray
            Inverse provider-information diagonal, shape (m,).
        mat_tmp1 : np.ndarray
            J1 matrix = info_beta_gamma * diag(info_gamma_inv), shape (p, m).
        mat_tmp2 : np.ndarray
            J2 matrix = solve(S, J1), shape (p, m).
        schur_score_beta : np.ndarray
            solve(S, score_beta), shape (p,).

        Returns
        -------
        tuple
            (d_gamma_prov, d_beta)
        """
        d_gamma_prov = info_gamma_inv * score_gamma + mat_tmp2.T @ (mat_tmp1 @ score_gamma - score_beta)
        d_beta = schur_score_beta - mat_tmp2 @ score_gamma
        return d_gamma_prov, d_beta

    def _backtrack(self) -> None:
        """Backtracking line search
        This is a method for choosing the step size in the gradient ascent algorithm.
        It starts with a full step and reduces the step size until the increase in the log-likelihood is sufficient.
        """
        s = 0.01
        t = 0.6
        self._last_step = (1.0, 0.0)
        while self.iter <= self.max_iter and self.beta_crit >= self.tol:
            self.iter += 1
            gamma_obs = self.gamma_prov[self._prov_indices]

            p = sigmoid(gamma_obs + self.X @ self.beta)
            q = self.N * p * (1 - p)

            score_gamma, score_beta, info_gamma_inv, info_beta_gamma, info_beta = self._compute_scores_and_info(p, q)

            # Match the Rcpp SerBIN implementation exactly and avoid forming S^{-1} explicitly.
            mat_tmp1 = info_beta_gamma * info_gamma_inv
            schur_matrix = info_beta - mat_tmp1 @ info_beta_gamma.T
            cho_L, lower = cho_factor(schur_matrix)
            mat_tmp2 = cho_solve((cho_L, lower), mat_tmp1)
            schur_score_beta = cho_solve((cho_L, lower), score_beta)

            d_gamma_prov, d_beta = self._compute_deltas(
                score_gamma, score_beta, info_gamma_inv, mat_tmp1, mat_tmp2, schur_score_beta
            )

            # C27: the joint Newton direction is used as it is, as in R's
            # logis_BIN_fe_prov.  Clipping the gamma block alone (formerly to
            # +-2*bound) changes the direction, can make it a descent direction
            # when the step needs gamma to offset xbar'd_beta (covariates far
            # from 0), and makes the iterates depend on the covariates' origin.

            v = 1
            loglkd = self._loglikelihood(self.gamma_prov[self._prov_indices], self.beta)
            d_loglkd = self._loglikelihood((self.gamma_prov + v * d_gamma_prov)[self._prov_indices], self.beta + v * d_beta) - loglkd
            lambda_ = np.concatenate([score_gamma, score_beta]) @ np.concatenate([d_gamma_prov, d_beta])

            while d_loglkd < s * v * lambda_ and not self._below_noise(s * v * lambda_, loglkd):
                v *= t
                d_loglkd = self._loglikelihood((self.gamma_prov + v * d_gamma_prov)[self._prov_indices], self.beta + v * d_beta) - loglkd

            self.gamma_prov += v * d_gamma_prov

            self.gamma_prov = np.clip(self.gamma_prov, np.median(self.gamma_prov) - self.bound, np.median(self.gamma_prov) + self.bound)
            beta_new = self.beta + v * d_beta

            self.beta_crit = np.linalg.norm(self.beta - beta_new,  ord=np.inf)

            self.beta = beta_new
            logger.debug(f"Inf norm of running diff in est reg parm is {self.beta_crit:.3e};")
            self._last_step = (v, float(np.max(np.abs(d_beta), initial=0.0)))

        self._report_stop()

    def _report_stop(self) -> None:
        """Warn when the loop ended without the Newton step becoming small.

        The stopping rule is R's (``stop = "beta"``): the accepted beta step,
        ``v * |d_beta|``, below ``tol``.  It reads a line search that shrank the
        step as convergence; C27 was such a case (a step shrunk to 8e-16 while
        the Newton beta step was 0.46).  The rule and the estimates are kept;
        only a warning is added, for that case and for the iteration limit.
        """
        if not self.beta_crit < self.tol:  # also NaN
            logger.warning(
                "Serbin did not converge in %d iterations (beta_crit=%.3e, tol=%.3e)",
                self.max_iter, self.beta_crit, self.tol,
            )
            return
        v, newton_beta_step = self._last_step
        if v < 1.0 and newton_beta_step >= self.tol:
            logger.warning(
                "Serbin stopped because the line search shortened the last step to "
                "v=%.3e (accepted beta step %.3e, Newton beta step %.3e, tol=%.3e); "
                "the estimates may not be at the maximum",
                v, self.beta_crit, newton_beta_step, self.tol,
            )

    def _no_backtrack(self) -> None:
        """Single Newton step without line search.
        """
        while self.iter < self.max_iter and self.beta_crit > self.tol:
            self.iter += 1
            gamma_obs = self.gamma_prov[self._prov_indices]
            p = sigmoid(gamma_obs + self.X @ self.beta)
            q = self.N * p * (1 - p)

            sc_g, sc_b, ig_inv, ibg, ib = self._compute_scores_and_info(p, q)
            mat1 = ibg * ig_inv
            schur_matrix = ib - mat1 @ ibg.T
            cho_L, lower = cho_factor(schur_matrix)
            mat2 = cho_solve((cho_L, lower), mat1)
            schur_score_beta = cho_solve((cho_L, lower), sc_b)
            d_gamma, d_beta = self._compute_deltas(sc_g, sc_b, ig_inv, mat1, mat2, schur_score_beta)
            # The full Newton direction (C27; see _backtrack).
            self.gamma_prov += d_gamma
            med = np.median(self.gamma_prov)
            self.gamma_prov = np.clip(self.gamma_prov, med - self.bound, med + self.bound)
            beta_cand = self.beta + d_beta
            self.beta_crit = np.linalg.norm(self.beta - beta_cand, np.inf)
            self.beta = beta_cand
        if not self.beta_crit <= self.tol:  # also NaN
            logger.warning(
                "Serbin (no backtracking) did not converge in %d iterations "
                "(beta_crit=%.3e, tol=%.3e)", self.max_iter, self.beta_crit, self.tol,
            )


class BanAlgorithm(BaseAlgorithm):
    """Ban's alternating updates for logistic fixed-effect estimation.
    """

    def fit(self) -> dict:
        """Fit by alternating updates in centred covariates (C30).

        Alternating updates of gamma and beta slow down when the two blocks
        are strongly coupled, and a common offset in the covariates couples
        them through the provider effects, which carry the intercept: on the
        AOH goldens Ban took 19 iterations centred, 705 with the covariates
        shifted by (+5, -3) and did not converge in 10,000 with (+50, -30).
        The updates therefore run with ``X - xbar`` (``xbar`` the
        trials-weighted mean covariate row) from the corresponding start, and
        the provider effects are returned in the original coordinates,
        ``gamma = gamma_c - xbar' beta``.  The median-relative clamp and the
        stopping rule are unchanged by the shift.
        """
        X = self.X
        if X.shape[1] == 0:
            return super().fit()
        xbar = self.N @ X / np.sum(self.N)
        self.X = X - xbar
        self.gamma_prov = self.gamma_prov + xbar @ self.beta
        try:
            super().fit()
        finally:
            self.X = X
            self.gamma_prov = self.gamma_prov - xbar @ self.beta
        return {'gamma': self.gamma_prov, 'beta': self.beta}
    def _update_gamma(self) -> tuple:
        """Update provider effects given fixed beta.

        Returns
        -------
        tuple
            (gamma_obs, p, q, score_gamma, delta_gamma)
        """
        gamma_obs = self.gamma_prov[self._prov_indices]

        eta = gamma_obs + self.X @ self.beta
        p = sigmoid(eta)
        q = self.N * p * (1.0 - p)
        q = np.maximum(q, 1e-20)

        residuals = self.y - self.N * p

        score_gamma = np.bincount(
            self._prov_indices,
            weights=residuals,
            minlength=self._n_providers,
        )

        info_gamma = np.bincount(
            self._prov_indices,
            weights=q,
            minlength=self._n_providers,
        )

        delta_gamma = score_gamma / info_gamma

        return gamma_obs, p, q, score_gamma, delta_gamma

    def _update_beta(self, p: np.ndarray, q: np.ndarray) -> tuple:
        """Update regression coefficients given fixed gamma.

        Returns
        -------
        tuple
            (score_beta, delta_beta)
        """
        residuals = self.y - self.N * p

        score_beta = self.X.T @ residuals
        info_beta = self.X.T @ (q[:, None] * self.X)

        delta_beta = solve_information(info_beta, score_beta, warn=True, what="Newton beta information")

        return score_beta, delta_beta

    def _backtrack(self) -> None:
        """BAN alternating Newton updates with backtracking.
        """
        s, t = 0.01, 0.6
        while self.iter < self.max_iter and max(self.gamma_crit, self.beta_crit) > self.tol:
            self.iter += 1

            # ---------------------------------------------------------
            # 1. Update gamma with beta fixed
            # ---------------------------------------------------------
            gamma_old = self.gamma_prov.copy()

            (
                gamma_obs,
                p,
                q,
                score_gamma,
                delta_gamma,
            ) = self._update_gamma()

            v = 1.0
            ll_old = self._loglikelihood(gamma_obs, self.beta)

            lambda_gamma = score_gamma @ delta_gamma

            while True:
                gamma_candidate = self.gamma_prov + v * delta_gamma

                ll_new = self._loglikelihood(
                    gamma_candidate[self._prov_indices],
                    self.beta,
                )

                if (ll_new - ll_old >= s * v * lambda_gamma
                        or self._below_noise(s * v * lambda_gamma, ll_old)):
                    break

                v *= t

            self.gamma_prov += v * delta_gamma

            med = np.median(self.gamma_prov)
            self.gamma_prov = np.clip(
                self.gamma_prov,
                med - self.bound,
                med + self.bound,
            )

            self.gamma_crit = np.linalg.norm(
                self.gamma_prov - gamma_old,
                ord=np.inf,
            )

            # ---------------------------------------------------------
            # 2. Recompute p and q after gamma update
            # ---------------------------------------------------------
            gamma_obs = self.gamma_prov[self._prov_indices]

            eta = gamma_obs + self.X @ self.beta
            p = sigmoid(eta)
            q = self.N * p * (1.0 - p)
            q = np.maximum(q, 1e-20)

            # ---------------------------------------------------------
            # 3. Update beta with gamma fixed
            # ---------------------------------------------------------
            score_beta, delta_beta = self._update_beta(p, q)

            v = 1.0
            ll_old = self._loglikelihood(
                gamma_obs,
                self.beta,
            )

            lambda_beta = score_beta @ delta_beta

            while True:
                beta_candidate = self.beta + v * delta_beta

                ll_new = self._loglikelihood(
                    gamma_obs,
                    beta_candidate,
                )

                if (ll_new - ll_old >= s * v * lambda_beta
                        or self._below_noise(s * v * lambda_beta, ll_old)):
                    break

                v *= t

            beta_candidate = self.beta + v * delta_beta

            self.beta_crit = np.linalg.norm(
                self.beta - beta_candidate,
                ord=np.inf,
            )

            self.beta = beta_candidate

        if self.iter >= self.max_iter and max(self.gamma_crit, self.beta_crit) > self.tol:
            logger.warning(
                "BAN backtracking did not converge in %d iterations "
                "(gamma_crit=%.3e, beta_crit=%.3e, tol=%.3e)",
                self.max_iter, self.gamma_crit, self.beta_crit, self.tol,
            )

    def _no_backtrack(self) -> None:
        """BAN alternating Newton updates without line search.
        """
        while self.iter < self.max_iter and max(self.gamma_crit, self.beta_crit) > self.tol:
            self.iter += 1

            # ---------------------------------------------------------
            # 1. Update gamma
            # ---------------------------------------------------------
            gamma_old = self.gamma_prov.copy()

            _, _, _, _, delta_gamma = self._update_gamma()

            self.gamma_prov += delta_gamma

            med = np.median(self.gamma_prov)
            self.gamma_prov = np.clip(
                self.gamma_prov,
                med - self.bound,
                med + self.bound,
            )

            self.gamma_crit = np.linalg.norm(
                self.gamma_prov - gamma_old,
                ord=np.inf,
            )

            # ---------------------------------------------------------
            # 2. Recompute p and q after gamma update
            # ---------------------------------------------------------
            gamma_obs = self.gamma_prov[self._prov_indices]

            eta = gamma_obs + self.X @ self.beta
            p = sigmoid(eta)
            q = self.N * p * (1.0 - p)
            q = np.maximum(q, 1e-20)

            # ---------------------------------------------------------
            # 3. Update beta
            # ---------------------------------------------------------
            _, delta_beta = self._update_beta(p, q)

            beta_candidate = self.beta + delta_beta

            self.beta_crit = np.linalg.norm(
                self.beta - beta_candidate,
                ord=np.inf,
            )

            self.beta = beta_candidate

        if self.iter >= self.max_iter and max(self.gamma_crit, self.beta_crit) > self.tol:
            logger.warning(
                "BAN did not converge in %d iterations "
                "(gamma_crit=%.3e, beta_crit=%.3e, tol=%.3e)",
                self.max_iter, self.gamma_crit, self.beta_crit, self.tol,
            )
