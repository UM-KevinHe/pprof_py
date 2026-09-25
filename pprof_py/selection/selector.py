"""CoxPHSelector: forward/backward/both-direction variable selection on
top of CoxPH, by AIC, BIC, or p-value.

This module fits many `CoxPH` models (one per candidate move considered)
and compares their own reported `log_likelihood_`, `n_features_in_`,
`n_events_`, and `p_values_` -- it never recomputes a partial
likelihood, a score, or a p-value itself. The entire selector is a loop
over `CoxPH(...).fit(...)` calls plus arithmetic on their outputs.

The AIC/BIC greedy algorithm is matched to R's `step()` on a `coxph`
object, verified directly (not assumed) against real R -- see
`docs/R_COMPATIBILITY.md`, "Variable selection", for the R session this
was checked against: at each step, every legal add-move (forward/both)
and remove-move (backward/both) is evaluated, plus staying put; the
option with the lowest resulting criterion wins; selection stops when
staying put is already the minimum. Forced variables are simply excluded
from the remove-candidate pool and are always part of the starting
model -- R's own mechanism for this is a `scope` with a non-trivial
`lower` formula, which does exactly the same thing.

p-value selection has no single canonical R equivalent to validate
against the way `step()` provides for AIC/BIC (the closest CRAN package,
`My.stepwise`, isn't available in this environment -- see
docs/R_COMPATIBILITY.md). It follows the classic textbook/SAS
`PROC PHREG`-style convention instead: forward adds the most significant
eligible candidate if its own p-value in the augmented model is below
`p_enter`; backward removes the least significant retained (non-forced)
variable if its p-value exceeds `p_remove`; "both" interleaves the two
(try an addition first, then re-check all retained variables for
removal) each step. The per-model p-values it acts on are themselves
fully R-validated (they come straight from `CoxPH.p_values_`); only the
selection *procedure* built on top of them lacks an R package to check
against directly, and that limitation is stated here rather than implied
away.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from ..base import ProviderModel

from ..models.survival.coxph import CoxPH
from .criteria import CRITERIA


class CoxPHSelector(ProviderModel):
    """Forward, backward, or bidirectional variable selection for CoxPH.

    Parameters
    ----------
    direction : {"forward", "backward", "both"}, default "forward"
    criterion : {"aic", "bic", "pvalue"}, default "aic"
    p_enter : float, default 0.05
        Used only when `criterion="pvalue"`: a candidate is added if its
        own p-value, in the model that would include it, is below this.
    p_remove : float, default 0.10
        Used only when `criterion="pvalue"`: a retained non-forced
        variable is removed if its p-value exceeds this. Conventionally
        `p_remove > p_enter`, so a variable that just entered isn't
        immediately eligible for removal on the same information used to
        admit it -- this is not enforced, but a warning is not raised
        either; an inverted configuration is a modeling choice, not
        something this class second-guesses.
    ties : {"breslow", "efron"}, default "breslow"
        Passed straight through to every `CoxPH` fit this selector runs.
    max_steps : int or None, default None
        Safety cap on the number of add/remove steps. Defaults to
        `4 * len(candidates) + 10`, generous for a well-behaved greedy
        search but still finite in case of pathological oscillation
        (most plausible with an inverted `p_enter`/`p_remove`).
    fit_intercept, max_iter, eps, confidence_level :
        Passed straight through to every `CoxPH` this selector fits --
        see `CoxPH`'s own docstring.

    Attributes (set by `fit`)
    -------------------------
    selected_variables_ : list of str
        Final variable set, forced variables included.
    selection_history_ : pandas.DataFrame
        One row per step: the action taken, the variable, and the
        criterion value(s) that drove the decision.
    final_model_ : CoxPH
        Fitted on `X[selected_variables_]` with the same
        strata/offset/weight/start-stop/ties as the search -- the actual
        object to read `coef_`, `standard_errors_`, `baseline_hazard_`,
        etc. from.
    n_steps_ : int
    """

    def __init__(
        self,
        direction: str = "forward",
        criterion: str = "aic",
        p_enter: float = 0.05,
        p_remove: float = 0.10,
        ties: str = "breslow",
        max_steps: Optional[int] = None,
        fit_intercept: bool = False,
        max_iter: int = 20,
        eps: float = 1e-9,
        confidence_level: float = 0.95,
    ):
        self.direction = direction
        self.criterion = criterion
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.ties = ties
        self.max_steps = max_steps
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        duration=None,
        start=None,
        stop=None,
        event=None,
        strata=None,
        offset=None,
        sample_weight=None,
        forced: Optional[Sequence[str]] = None,
        candidates: Optional[Sequence[str]] = None,
    ) -> "CoxPHSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CoxPHSelector requires a pandas DataFrame for X (not a bare array), "
                "since variables are selected by name."
            )
        if self.direction not in ("forward", "backward", "both"):
            raise ValueError(f"direction must be 'forward', 'backward', or 'both'; got {self.direction!r}")
        if self.criterion not in ("aic", "bic", "pvalue"):
            raise ValueError(f"criterion must be 'aic', 'bic', or 'pvalue'; got {self.criterion!r}")

        forced = list(forced) if forced is not None else []
        if candidates is None:
            candidates = [c for c in X.columns if c not in forced]
        else:
            candidates = [c for c in candidates if c not in forced]
        missing = [c for c in list(forced) + list(candidates) if c not in X.columns]
        if missing:
            raise ValueError(f"Column(s) not found in X: {missing}")

        self._shared_kwargs = dict(
            duration=duration, start=start, stop=stop, event=event,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        self._forced = forced
        self._candidates = candidates
        max_steps = self.max_steps if self.max_steps is not None else 4 * len(candidates) + 10

        history = []

        if self.direction == "backward":
            current = list(forced) + list(candidates)
        else:
            current = list(forced)

        current_model = self._fit_subset(X, current)
        current_score = self._score_or_nan(current_model)
        history.append(self._history_row(0, "start", None, current, current_score))

        step = 0
        while step < max_steps:
            step += 1
            if self.criterion == "pvalue":
                action, variable, new_current, new_model, new_score, detail = self._pvalue_step(
                    X, current, current_model
                )
            else:
                action, variable, new_current, new_model, new_score, detail = self._score_step(
                    X, current, current_model, current_score
                )

            if action == "stop":
                history.append(self._history_row(step, "stop", None, current, current_score, detail))
                break

            history.append(self._history_row(step, action, variable, new_current, new_score, detail))
            current, current_model, current_score = new_current, new_model, new_score
        else:
            history.append(self._history_row(step, "stop (max_steps reached)", None, current, current_score))

        self.selected_variables_ = current
        self.final_model_ = current_model
        self.selection_history_ = pd.DataFrame(history)
        self.n_steps_ = step

        return self

    # ------------------------------------------------------------------
    # AIC/BIC search
    # ------------------------------------------------------------------
    def _score(self, model) -> float:
        return CRITERIA[self.criterion](model)

    def _score_step(self, X, current, current_model, current_score):
        """One step of the R `step()`-equivalent greedy search: evaluate
        every legal move, keep the best if it beats staying put."""
        moves = []  # (score, action, variable, new_var_list)

        if self.direction in ("forward", "both"):
            for c in self._candidates:
                if c in current:
                    continue
                trial = current + [c]
                model = self._fit_subset(X, trial)
                moves.append((self._score(model), "add", c, trial, model))

        if self.direction in ("backward", "both"):
            for c in current:
                if c in self._forced:
                    continue
                trial = [v for v in current if v != c]
                model = self._fit_subset(X, trial)
                moves.append((self._score(model), "remove", c, trial, model))

        if not moves:
            return "stop", None, current, current_model, current_score, {"reason": "no legal moves"}

        best_score, best_action, best_var, best_vars, best_model = min(moves, key=lambda m: m[0])
        detail = {"candidates_evaluated": {v: s for s, a, v, _, _ in moves}}

        if best_score >= current_score:
            return "stop", None, current, current_model, current_score, detail

        return best_action, best_var, best_vars, best_model, best_score, detail

    # ------------------------------------------------------------------
    # p-value search
    # ------------------------------------------------------------------
    def _pvalue_of(self, model, variable: str) -> float:
        names = list(model.feature_names_in_)
        return float(model.p_values_[names.index(variable)])

    def _pvalue_step(self, X, current, current_model):
        """Classic add-then-check-remove p-value step (see module
        docstring for the exact convention and why it was chosen)."""
        if self.direction in ("forward", "both"):
            best_p, best_var, best_model, best_vars = np.inf, None, None, None
            for c in self._candidates:
                if c in current:
                    continue
                trial = current + [c]
                model = self._fit_subset(X, trial)
                p = self._pvalue_of(model, c)
                if p < best_p:
                    best_p, best_var, best_model, best_vars = p, c, model, trial
            if best_var is not None and best_p < self.p_enter:
                return "add", best_var, best_vars, best_model, self._score_or_nan(best_model), {"p_value": best_p}
            if self.direction == "forward":
                return "stop", None, current, current_model, self._score_or_nan(current_model), {"reason": "no candidate below p_enter"}

        if self.direction in ("backward", "both"):
            removable = [c for c in current if c not in self._forced]
            if removable:
                p_values = {c: self._pvalue_of(current_model, c) for c in removable}
                worst_var = max(p_values, key=p_values.get)
                if p_values[worst_var] > self.p_remove:
                    trial = [v for v in current if v != worst_var]
                    model = self._fit_subset(X, trial)
                    return "remove", worst_var, trial, model, self._score_or_nan(model), {"p_value": p_values[worst_var]}

        return "stop", None, current, current_model, self._score_or_nan(current_model), {"reason": "no move meets p_enter/p_remove"}

    def _score_or_nan(self, model) -> float:
        # p-value mode doesn't select on AIC/BIC, but the history table
        # still reports it for context, using whichever criterion is
        # configured (defaulting to AIC's formula) purely for display.
        try:
            return CRITERIA.get(self.criterion, CRITERIA["aic"])(model)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    def _fit_subset(self, X: pd.DataFrame, variables: List[str]) -> CoxPH:
        cols = list(dict.fromkeys(variables))  # de-dup, preserve order
        X_subset = X[cols] if cols else pd.DataFrame(index=X.index)
        model = CoxPH(
            ties=self.ties, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
            eps=self.eps, confidence_level=self.confidence_level,
        )
        return model.fit(X_subset, **self._shared_kwargs)

    def _history_row(self, step, action, variable, variables, score, detail=None) -> dict:
        if self.criterion == "pvalue":
            # Keep the "p_value" column's units consistent: only an
            # actual per-variable p-value belongs here, never the AIC-
            # equivalent `score` used purely to drive comparisons in
            # AIC/BIC mode (a "start" or "stop" row has no single
            # variable's p-value to report, so this is None, not a
            # number from a different scale entirely).
            display = detail.get("p_value") if detail and "p_value" in detail else None
        else:
            display = score
        row = {
            "step": step,
            "action": action,
            "variable": variable,
            "n_variables": len(variables),
            "variables": list(variables),
            (self.criterion if self.criterion != "pvalue" else "p_value"): display,
        }
        return row

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Coefficient table for `final_model_` -- see `CoxPH.summary()`."""
        return self.final_model_.summary()
