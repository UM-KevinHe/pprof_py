import logging

import numpy as np
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

            # Cap gamma direction to prevent one extreme provider from hijacking
            # the backtracking step size. Any d_gamma beyond 2*bound is wasted
            # since gamma will be clipped to [median-bound, median+bound] anyway.
            max_gamma_step = 2.0 * self.bound
            d_gamma_prov = np.clip(d_gamma_prov, -max_gamma_step, max_gamma_step)

            v = 1
            loglkd = self._loglikelihood(self.gamma_prov[self._prov_indices], self.beta)
            d_loglkd = self._loglikelihood((self.gamma_prov + v * d_gamma_prov)[self._prov_indices], self.beta + v * d_beta) - loglkd
            lambda_ = np.concatenate([score_gamma, score_beta]) @ np.concatenate([d_gamma_prov, d_beta])

            while d_loglkd < s * v * lambda_:
                v *= t
                d_loglkd = self._loglikelihood((self.gamma_prov + v * d_gamma_prov)[self._prov_indices], self.beta + v * d_beta) - loglkd

            self.gamma_prov += v * d_gamma_prov

            self.gamma_prov = np.clip(self.gamma_prov, np.median(self.gamma_prov) - self.bound, np.median(self.gamma_prov) + self.bound)
            beta_new = self.beta + v * d_beta

            self.beta_crit = np.linalg.norm(self.beta - beta_new,  ord=np.inf)

            self.beta = beta_new
            logger.debug(f"Inf norm of running diff in est reg parm is {self.beta_crit:.3e};")

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
            # Cap gamma direction (same rationale as _backtrack)
            max_gamma_step = 2.0 * self.bound
            d_gamma = np.clip(d_gamma, -max_gamma_step, max_gamma_step)
            self.gamma_prov += d_gamma
            med = np.median(self.gamma_prov)
            self.gamma_prov = np.clip(self.gamma_prov, med - self.bound, med + self.bound)
            beta_cand = self.beta + d_beta
            self.beta_crit = np.linalg.norm(self.beta - beta_cand, np.inf)
            self.beta = beta_cand


class BanAlgorithm(BaseAlgorithm):
    """Ban's alternating updates for logistic fixed-effect estimation.
    """
    def _update_gamma(self) -> tuple:
        """Update provider effects given fixed beta.

        Returns
        -------
        tuple
            (gamma_obs, p, q, score_gamma, delta_gamma)
        """
        gamma_obs = self.gamma_prov[self._prov_indices]
        linear = gamma_obs + self.X @ self.beta
        p = sigmoid(linear)
        q = self.N * p * (1 - p)
        # Zero-guard (matches R's Fixed_effect.cpp lines 128-129)
        q = np.maximum(q, 1e-20)
        indices = self._prov_indices
        n_provs = self._n_providers
        score_gamma = np.bincount(indices, weights=(self.y - self.N * p), minlength=n_provs)
        delta_gamma = score_gamma / np.bincount(indices, weights=q, minlength=n_provs)
        return gamma_obs, p, q, score_gamma, delta_gamma

    def _update_beta(self, p: np.ndarray, q: np.ndarray) -> tuple:
        """Update regression coefficients given fixed gamma.

        Returns
        -------
        tuple
            (score_beta, delta_beta)
        """
        score_beta = self.X.T @ (self.y - self.N * p)
        info_beta = self.X.T @ (q[:, None] * self.X)
        delta_beta = np.linalg.solve(info_beta, score_beta)
        return score_beta, delta_beta

    def _backtrack(self) -> None:
        """Alternating backtracking updates for gamma and beta.
        """
        s, t = 0.01, 0.8
        while self.iter < self.max_iter and self.beta_crit > self.tol:
            self.iter += 1
            # Gamma update
            gamma_obs, p, q, sc_g, d_g = self._update_gamma()
            v = 1.0
            ll_old = self._loglikelihood(gamma_obs, self.beta)
            while True:
                ll_new = self._loglikelihood(
                    (self.gamma_prov + v * d_g)[self._prov_indices],
                    self.beta
                )
                if ll_new - ll_old >= s * v * (sc_g @ d_g):
                    break
                v *= t
            self.gamma_prov += v * d_g
            med = np.median(self.gamma_prov)
            self.gamma_prov = np.clip(self.gamma_prov, med - self.bound, med + self.bound)

            # Beta update
            _, d_b = self._update_beta(p, q)
            v = 1.0
            ll_old = self._loglikelihood(self.gamma_prov[self._prov_indices], self.beta)
            while True:
                beta_cand = self.beta + v * d_b
                ll_new = self._loglikelihood(self.gamma_prov[self._prov_indices], beta_cand)
                if ll_new - ll_old >= s * v * (d_b @ d_b):
                    break
                v *= t
            beta_cand = self.beta + v * d_b
            self.beta_crit = np.linalg.norm(self.beta - beta_cand, np.inf)
            self.beta = beta_cand

    def _no_backtrack(self) -> None:
        """Alternating simple updates without line search.
        """
        while self.iter < self.max_iter and self.beta_crit > self.tol:
            self.iter += 1
            # Gamma update
            _, p, q, _, d_g = self._update_gamma()
            self.gamma_prov += d_g
            med = np.median(self.gamma_prov)
            self.gamma_prov = np.clip(self.gamma_prov, med - self.bound, med + self.bound)

            # Beta update
            _, d_b = self._update_beta(p, q)
            beta_cand = self.beta + d_b
            self.beta_crit = np.linalg.norm(self.beta - beta_cand, np.inf)
            self.beta = beta_cand
