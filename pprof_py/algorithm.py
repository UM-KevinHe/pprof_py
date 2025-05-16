import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from scipy.special import expit as sigmoid


class Algorithm(Enum):
    """Enumeration for optimization algorithms."""
    SERBIN = "Serbin"
    BAN = "Ban"


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
    backtrack : bool
        Whether to use backtracking line search during optimization.
    max_iter : int
        The maximum number of iterations for the optimization algorithm.
    bound : float
        The bound for clipping group effects to prevent extreme values.
    tol : float
        The tolerance for convergence, based on the change in parameters.
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
        backtrack: bool,
        max_iter: int,
        bound: float,
        tol: float,
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
        backtrack : bool
            Whether to use backtracking line search.
        max_iter : int
            Maximum number of iterations.
        bound : float
            Bound for clipping provider effects.
        tol : float
            Tolerance for convergence.
        """
        self.data = data
        self.y_index = y_index
        self.X = X
        self.prov_index = prov_index
        self.n_prov = n_prov
        self.gamma_prov = gamma_prov
        self.beta = beta
        self.backtrack = backtrack
        self.max_iter = max_iter
        self.bound = bound
        self.tol = tol

        # Convert response column to integers and validate
        if self.data[:, self.y_index].dtype == object:
            self.data[:, self.y_index] = self.data[:, self.y_index].astype(int)
        y_col = self.data[:, self.y_index].astype(float)
        if np.isnan(y_col).any() or np.isinf(y_col).any():
            raise ValueError("Response variable contains NaN or infinite values")
        self.y = y_col.astype(int)

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
        return np.sum(self.y * linear - np.log1p(np.exp(linear)))
    

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
        unique, indices = np.unique(self.data[:, self.prov_index], return_inverse=True)
        score_gamma = np.bincount(indices, weights=(self.y - p))
        score_beta = self.X.T @ (self.y - p)
        info_gamma_inv = 1 / np.bincount(indices, weights=q)

        info_beta_gamma = np.array([
            np.bincount(indices, weights=q * self.X[:, i], minlength=len(unique))
            for i in range(self.X.shape[1])
        ])

        info_beta = self.X.T @ (q[:, None] * self.X)
        return score_gamma, score_beta, info_gamma_inv, info_beta_gamma, info_beta

    def _compute_deltas(self, score_gamma: np.ndarray, score_beta: np.ndarray, info_gamma_inv: np.ndarray, 
                        mat_tmp1: np.ndarray, mat_tmp2: np.ndarray, schur_inv: np.ndarray) -> tuple:
        """Compute the deltas for gamma and beta.

        The deltas are the changes in the parameters that will increase the log-likelihood.
        They are computed using the scores and the inverse of the information matrix.

        Parameters:
        ----------
        score_gamma (array): The score for gamma.
        score_beta (array): The score for beta.
        info_gamma_inv (array): The inverse of the information matrix for gamma.
        mat_tmp1 (array): The product of the inverse of the information matrix for gamma and the information matrix for beta-gamma.
        mat_tmp2 (array): The product of mat_tmp1 and the inverse of the Schur complement.
        schur_inv (array): The inverse of the Schur complement.

        Returns:
        -------
        tuple: The deltas for gamma and beta.
        """

        d_gamma_prov = info_gamma_inv * score_gamma + mat_tmp2 @ (mat_tmp1.T @ score_gamma - score_beta)
        d_beta = -mat_tmp2.T @ score_gamma + schur_inv @ score_beta
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
            gamma_obs = np.repeat(self.gamma_prov, self.n_prov)

            p = sigmoid(gamma_obs + self.X @ self.beta)
            q = p * (1 - p)

            score_gamma, score_beta, info_gamma_inv, info_beta_gamma, info_beta = self._compute_scores_and_info(p, q)

            mat_tmp1 = (info_gamma_inv * info_beta_gamma).T
            schur_inv = np.linalg.solve(info_beta - info_beta_gamma @ mat_tmp1, np.eye(info_beta.shape[0]))

            mat_tmp2 = mat_tmp1 @ schur_inv

            d_gamma_prov, d_beta = self._compute_deltas(score_gamma, score_beta, info_gamma_inv, mat_tmp1, mat_tmp2, schur_inv)

            v = 1
            loglkd = self._loglikelihood(np.repeat(self.gamma_prov, self.n_prov), self.beta)
            d_loglkd = self._loglikelihood(np.repeat(self.gamma_prov + v * d_gamma_prov, self.n_prov), self.beta + v * d_beta) - loglkd
            lambda_ = np.concatenate([score_gamma, score_beta]) @ np.concatenate([d_gamma_prov, d_beta])

            while d_loglkd < s * v * lambda_:
                v *= t
                d_loglkd = self._loglikelihood(np.repeat(self.gamma_prov + v * d_gamma_prov, self.n_prov), self.beta + v * d_beta) - loglkd

            self.gamma_prov += v * d_gamma_prov

            self.gamma_prov = np.clip(self.gamma_prov, np.median(self.gamma_prov) - self.bound, np.median(self.gamma_prov) + self.bound)
            beta_new = self.beta + v * d_beta

            self.beta_crit = np.linalg.norm(self.beta - beta_new,  ord=np.inf)

            self.beta = beta_new
            print(f"Inf norm of running diff in est reg parm is {self.beta_crit:.3e};")

    def _no_backtrack(self) -> None:
        """Single Newton step without line search.
        """
        while self.iter < self.max_iter and self.beta_crit > self.tol:
            self.iter += 1
            gamma_obs = np.repeat(self.gamma_prov, self.n_prov)
            p = sigmoid(gamma_obs + self.X @ self.beta)
            q = p * (1 - p)

            sc_g, sc_b, ig_inv, ibg, ib = self._compute_scores_and_info(p, q)
            mat1 = (ig_inv * ibg).T
            schur_inv = np.linalg.inv(ib - ibg @ mat1)
            mat2 = mat1 @ schur_inv
            d_gamma, d_beta = self._compute_deltas(sc_g, sc_b, ig_inv, mat1, mat2, schur_inv)
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
        gamma_obs = np.repeat(self.gamma_prov, self.n_prov)
        linear = gamma_obs + self.X @ self.beta
        p = sigmoid(linear)
        q = p * (1 - p)
        _, idx = np.unique(self.data[:, self.prov_index], return_inverse=True)
        score_gamma = np.bincount(idx, weights=(self.y - p))
        delta_gamma = score_gamma / np.bincount(idx, weights=q)
        return gamma_obs, p, q, score_gamma, delta_gamma

    def _update_beta(self, p: np.ndarray, q: np.ndarray) -> tuple:
        """Update regression coefficients given fixed gamma.

        Returns
        -------
        tuple
            (score_beta, delta_beta)
        """
        score_beta = self.X.T @ (self.y - p)
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
                    np.repeat(self.gamma_prov + v * d_g, self.n_prov),
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
            ll_old = self._loglikelihood(np.repeat(self.gamma_prov, self.n_prov), self.beta)
            while True:
                beta_cand = self.beta + v * d_b
                ll_new = self._loglikelihood(np.repeat(self.gamma_prov, self.n_prov), beta_cand)
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
